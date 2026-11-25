from copy import deepcopy
from .mechanism_program import Program, ProgramError, digest, source_schema
from .plan_memory import PROGRAM_ABI


def invocation_state(selection, changed, current, update):
    if changed and selection is not None:
        return deepcopy(selection['parameters'])
    return deepcopy(current if update is None else update)


GUIDANCE = '''
operations optionally adds/updates named reusable operations. Each item is
{name,source,description,evidence_ids}; source uses the same observe/decide ABI.
For local changes, use {name,base_version,edits,description,evidence_ids} instead.
base_version is an exact stored operation ID. edits is an ordered nonempty list
of {old,new} literal replacements. Each nonempty old must match exactly once in
the source after preceding edits; include unique surrounding code when needed.
An empty new deletes that text. The full resulting source is compiled and saved
as a new version; unchanged code is retained exactly. A different name creates
an alternative from that base. Full source remains available for new operations
or broader rewrites. Do not include source together with base_version/edits.
Only listed names change. Other operations, old versions and their evidence remain.
operation={name,parameters} selects a stored operation with fresh episode-local
state equal to parameters. Its observe/decide runs on every observation/step;
on a new named invocation, parameters owns that new state even if the same response
also includes state_update for the departing program. Put destination fields in
parameters. Without a new named invocation, state_update updates the current state;
with operation=null it can initialize the resumed main program.
delegate checks the measured objective and consumes the checked candidate plan.
actions=[] is sufficient when learned decide code can generate the required action
or perform a supported local transition.
To select another stored operation locally, return kind=continue with operation;
derive its parameters from current measurements/state. operation=null returns to
the main program. Switching clears the previous numeric goal and plan; the selected
operation derives its own goal. A route records execution order, not task success.
The active invocation pins its source version; explicitly select it again to use
an updated version. Parameters and scene state are never inherited next episode.
Review may update operations while keeping the main program. entry_operation names
an optional stored operation to enter on fresh observations next episode; null
uses the main program. Its rule must derive new scene parameters or ask for them.
Numeric completion, effect confirmation and choosing the next operation remain
separate. Teach useful partial operations without rewriting existing skills.
Review operation_versions={name:stored_version_id} restores a specific earlier
operation without deleting its evidence or rolling back unrelated operations.
'''

def updates_schema():
    common=dict(name=dict(type='string'),description=dict(type='string'),
        evidence_ids=dict(type='array',items=dict(type='string')))
    full=dict(common,source=source_schema())
    revision=dict(common,base_version=dict(type='string'),edits=dict(type='array',minItems=1,
        items=dict(type='object',additionalProperties=False,
            properties=dict(old=dict(type='string',minLength=1),new=dict(type='string')),
            required=['old','new'])))
    return dict(type='array',items=dict(anyOf=[
        dict(type='object',additionalProperties=False,properties=fields,required=list(fields))
        for fields in (full,revision)]))

def invocation_schema():
    return dict(anyOf=[dict(type='null'),dict(type='object',additionalProperties=False,
        properties=dict(name=dict(type='string'),parameters=dict(type='object')),
        required=['name','parameters'])])

def prepare_updates(proposals,known,episode,step,versions=()):
    if not isinstance(proposals,list):raise ProgramError('operations must be a list')
    records=[]
    for p in proposals:
        if (not isinstance(p,dict) or set(p) not in (
                {'name','source','description','evidence_ids'},
                {'name','base_version','edits','description','evidence_ids'})
                or not isinstance(p['name'],str) or not p['name'].strip()
                or not isinstance(p['description'],str) or not isinstance(p['evidence_ids'],list)
                or any(not isinstance(i,str) for i in p['evidence_ids'])):
            raise ProgramError('Operation requires name, description, evidence_ids and either source or base_version with edits')
        if not set(p['evidence_ids'])<=known:raise ProgramError('Unavailable operation evidence IDs')
        resolved=deepcopy(p)
        if 'base_version' in p:
            base=next((r for r in versions if r['id']==p['base_version']),None)
            if base is None:raise ProgramError('base_version must reference a stored operation version')
            edits=resolved.pop('edits')
            if not isinstance(edits,list) or not edits:
                raise ProgramError('Operation edits must be a nonempty list')
            source=base['source']
            for index,edit in enumerate(edits):
                if (not isinstance(edit,dict) or set(edit)!={'old','new'}
                        or not isinstance(edit['old'],str) or not edit['old']
                        or not isinstance(edit['new'],str)):
                    raise ProgramError(f'Operation edit {index+1} requires nonempty old and string new')
                matches=source.count(edit['old'])
                if matches!=1:
                    raise ProgramError(f'Operation edit {index+1} old matches {matches} times; include unique exact context')
                source=source.replace(edit['old'],edit['new'],1)
            resolved['source']=source
        program=Program(resolved['source'])
        records.append(dict(id=digest([p['name'],program.sha]),**resolved,abi=PROGRAM_ABI,
            origin_episode=episode,origin_step=step,status='provisional'))
    return records

class OperationKnowledge:
    def __init__(self,memory,episode):
        self.bank=memory.setdefault('operation_knowledge',dict(versions=[],selected={},entry=None,
            invocations=[],flows=[]))
        self.episode=episode;self.active=None;self.started=False

    def install(self,records):
        for record in records:
            if not any(r['id']==record['id'] for r in self.bank['versions']):
                self.bank['versions'].append(deepcopy(record))
            self.bank['selected'][record['name']]=record['id']

    def prepare_invocation(self,value,records=()):
        if value is None:return None
        names=set(self.bank['selected'])|{r['name'] for r in records}
        if (not isinstance(value,dict) or set(value)!={'name','parameters'}
                or not isinstance(value['name'],str) or value['name'] not in names
                or not isinstance(value['parameters'],dict)):
            raise ProgramError('operation must select a stored name and parameters object')
        digest(value)
        return deepcopy(value)

    def set_entry(self,name):
        if name is not None and (not isinstance(name,str) or name not in self.bank['selected']):
            raise ProgramError('entry_operation must name a stored operation or be null')
        self.bank['entry']=name

    def restore(self,selections):
        for name,version in selections.items():
            if not any(r['name']==name and r['id']==version for r in self.bank['versions']):
                raise ProgramError('operation_versions must reference a stored version of that name')
        self.bank['selected'].update(selections)

    def record(self):
        if self.active is None:return None
        return next(r for r in self.bank['versions'] if r['id']==self.active['version'])

    def select(self,value,ctx,source,reason):
        self.started=True
        version=self.bank['selected'][value['name']] if value is not None else None
        if self.active is not None and value is not None:
            if self.active['version']==version and self.active['parameters']==value['parameters']:return False
        elif self.active is None and value is None:return False
        previous=self.active
        if previous is not None:
            previous.update(end=ctx['id'],exit_reason=reason,
                control_at_exit=deepcopy(ctx.get('experience_control')),
                effects_at_exit=deepcopy(ctx.get('effects',[])))
        self.active=None
        if value is not None:
            self.active=dict(id=f'e{self.episode}:operation{len(self.bank["invocations"])+1}',
                episode=self.episode,name=value['name'],version=version,
                parameters=deepcopy(value['parameters']),start=ctx['id'],source=source,evidence_ids=[])
            self.bank['invocations'].append(self.active)
        flow=next((f for f in self.bank['flows'] if f['episode']==self.episode),None)
        if flow is None:
            flow=dict(id=f'e{self.episode}:operation_flow',episode=self.episode,invocations=[],connections=[]);self.bank['flows'].append(flow)
        if self.active is not None:flow['invocations'].append(self.active['id'])
        if previous is not None:
            flow['connections'].append(dict(before=previous['id'],after=self.active['id'] if self.active else None,
                observation=ctx['id'],evidence_ids=list(previous['evidence_ids']),source=source,reason=reason))
        return True

    def observe(self,transition):
        identity=transition.get('operation_invocation')
        if identity is None or not transition.get('executed'):return
        item=next(i for i in self.bank['invocations'] if i['id']==identity)
        if transition['id'] not in item['evidence_ids']:item['evidence_ids'].append(transition['id'])

    def finish(self,ctx):
        if self.active is not None:
            self.active.update(end=ctx['id'],exit_reason='episode_ended',
                control_at_exit=deepcopy(ctx.get('experience_control')),effects_at_exit=deepcopy(ctx.get('effects',[])))

    def context(self):
        selected=set(self.bank['selected'].values())
        if self.active is not None:selected.add(self.active['version'])
        return dict(entry=self.bank['entry'],selected=deepcopy(self.bank['selected']),active=deepcopy(self.active),
            operations=[deepcopy(r) for r in self.bank['versions'] if r['id'] in selected],
            flows=deepcopy(self.bank['flows']))
