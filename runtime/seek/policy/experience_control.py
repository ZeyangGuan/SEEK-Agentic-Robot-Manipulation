from copy import deepcopy
import math
from .mechanism_program import ProgramError, digest
from .feature_geometry import goal_errors


GUIDANCE = '''
Supply a measured control intent {kind:"control",name,goals,scales,requires};
reason is optional annotation. goals/scales name measured features and positive
target tolerances for either or both arms. Complete arm rx/ry/rz triples are
orientations; equivalent rotation vectors denote the same goal. Scalar features
and requires intervals retain their literal measured-coordinate meaning.
Measured role.<name>.surface_x/surface_y/surface_z are camera-depth surface
coordinates in environment meters, not an object center or holding label. They
are null when the tracked role or its depth is unavailable. RGB-only experience
does not invent depth. requires contains {feature,lower,upper} applicability
intervals. Choose current-scene parameters from observations and express required
holds in those bounds. Actual receipts record objective progress and conditions.
control_status() returns ask/fallback for the current objective. It never fits
recorded responses or generates an action. Learned decide code supplies actions
or uses cached/delegate to consume checked teacher candidates.
A control objective and task-effect confirmation are independent. Use effects
and observe for pending or unknown effects. An information-gathering intent may
be {kind:"probe",reason,control}, with control a numeric objective or null.
Numeric completion never confirms an effect. Use observe to assess evidence and
select the next intent locally or ask for the specific missing information.
A bare probe invents no objective and does not resume an older objective.
intent=null retains an existing intent; at fresh entry it permits checked teacher
candidates without an objective. Receipts are saved without inventing a goal or
success label. program=null keeps the current program.
A learned program may select the next measured intent via continue/delegate;
that invalidates old candidates. Repeating an identical objective retains its
invocation and receipt evidence. Goals, bindings and state reset each episode;
persist the parameter-derivation rule in code. Numeric completion is a local
control boundary, separate from task-effect confirmation and native task success.
'''


def intent_schema():
    def obj(properties):
        return dict(type='object', properties=properties, required=[k for k in properties if k!='reason'], additionalProperties=False)
    numeric = dict(type='object', additionalProperties=dict(type='number'))
    control = obj(dict(kind=dict(type='string', enum=['control']), name=dict(type='string'),
        goals=numeric, scales=numeric,
        requires=dict(type='array', items=obj(dict(feature=dict(type='string'), lower=dict(type='number'), upper=dict(type='number')))),
        reason=dict(type='string')))
    probe = obj(dict(kind=dict(type='string', enum=['probe']), reason=dict(type='string')))
    probe['properties']['control'] = dict(anyOf=[control, dict(type='null')])
    return dict(anyOf=[control, probe, dict(type='null')])


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def control_objective(spec):
    return spec.get('control') if spec.get('kind') == 'probe' else spec


def roles_used(spec):
    names = list(spec['goals']) + [r['feature'] for r in spec['requires']]
    return sorted({name.split('.')[1] for name in names if name.startswith('role.')})


def measured_features(snapshot):
    features = dict(snapshot.get('features', {}))
    for name, role in snapshot.get('roles', {}).items():
        surface = role.get('surface') or {}
        point = surface.get('point_environment_m') if role.get('valid') and surface.get('valid') else None
        for index, axis in enumerate('xyz'):
            key = 'role.' + name + '.surface_' + axis
            if key not in features:
                features[key] = float(point[index]) if point is not None else None
    return features


def condition(spec, snapshot, identities=None):
    features = measured_features(snapshot)
    for name in roles_used(spec):
        role = snapshot.get('roles', {}).get(name, {})
        if not role.get('valid'):
            return dict(status='unavailable', reason='role_measurement_unavailable', feature=name)
        if identities is not None and role.get('generation') != identities.get(name):
            return dict(status='unavailable', reason='role_binding_changed', feature=name)
    for bound in spec['requires']:
        value = features.get(bound['feature'])
        if not finite(value):
            return dict(status='unavailable', reason='condition_measurement_unavailable', feature=bound['feature'])
        if not bound['lower'] <= value <= bound['upper']:
            return dict(status='outside', reason='objective_condition_changed', feature=bound['feature'])
    for name in spec['goals']:
        if not finite(features.get(name)):
            return dict(status='unavailable', reason='goal_measurement_unavailable', feature=name)
    errors = goal_errors(features, spec['goals'], spec['scales'])
    return dict(status='complete' if all(abs(v)<=1 for v in errors.values()) else 'open',
        normalized_errors=errors, native_task_success=None)


class ExperienceControl:
    def __init__(self, memory, episode, frozen=False):
        self.frozen = frozen
        self.memory, self.episode = memory, episode
        self.bank = memory.setdefault('control_experience', dict(methods={}, invocations=[]))
        self.active = None
        self.last_decision = None

    def prepare(self, spec, ctx, *, has_actions, has_program):
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise ProgramError('Intent must be an object or null')
        spec=deepcopy(spec)

        if not isinstance(spec.get('reason'),str):spec['reason']=''
        if spec.get('kind') == 'probe':
            if set(spec)-{'control'} != {'kind','reason'}:
                raise ProgramError('Probe intent contains kind, reason and optional control')
            if spec.get('control') is not None:
                if not isinstance(spec['control'],dict) or spec['control'].get('kind')!='control':
                    raise ProgramError('Probe control must be a numeric control objective')
                spec['control']=self.prepare(spec['control'],ctx,has_actions=has_actions,has_program=has_program)
            return deepcopy(spec)
        fields={'kind','name','goals','scales','requires','reason'}
        if set(spec) not in (fields,fields|{'arm'}) or spec['kind'] != 'control':
            raise ProgramError('Control intent fields do not match the contract')
        if not isinstance(spec['name'],str) or not spec['name'].strip():
            raise ProgramError('Control intent requires a name')

        if 'arm' in spec and spec['arm'] not in ('left','right'):
            raise ProgramError('Legacy control arm must be left or right')
        goals, scales = spec['goals'], spec['scales']
        features = measured_features(ctx)
        if not isinstance(goals,dict) or not goals or not isinstance(scales,dict) or set(goals)!=set(scales):
            raise ProgramError('Goals and positive scales must name the same measured features')
        if any(name not in features or not finite(value) or not finite(scales[name]) or scales[name]<=0
               for name,value in goals.items()):
            raise ProgramError('Goals must use available feature names and finite values/positive tolerances')
        if not isinstance(spec['requires'],list):
            raise ProgramError('requires must be measured feature intervals')
        for item in spec['requires']:
            if (not isinstance(item,dict) or set(item)!={'feature','lower','upper'} or
                item['feature'] not in features or not finite(item['lower']) or not finite(item['upper']) or
                item['lower'] > item['upper']):
                raise ProgramError('Invalid measured applicability interval')
        return deepcopy(spec)

    def activate(self, spec, ctx, *, source='teacher', program=None):
        if spec is None:
            return False
        objective = control_objective(spec)
        identities = {name:ctx['roles'][name]['generation'] for name in roles_used(objective)} if objective is not None else {}
        if source=='mechanism' and self.active is not None:
            previous_spec = self.objective() if self.objective() is not None and objective is not None else self.active['spec']
            current_spec = objective if self.objective() is not None and objective is not None else spec
            previous = {k:v for k,v in previous_spec.items() if k!='reason'}
            current = {k:v for k,v in current_spec.items() if k!='reason'}
            same = previous==current
            if not same and previous.get('kind')==current.get('kind')=='control':
                numerical = ('goals','scales')
                same = ({k:v for k,v in previous.items() if k not in numerical}==
                        {k:v for k,v in current.items() if k not in numerical}) and all(
                    set(previous[field])==set(current[field]) and all(
                        math.isclose(previous[field][name],value,rel_tol=1e-12,abs_tol=1e-12)
                        for name,value in current[field].items()) for field in numerical)
            if same and self.active['roles']==identities:
                self.active['spec']=deepcopy(spec)
                return False
        if self.active is not None and self.active['status']=='active':
            self.active['status'] = 'replaced'
        method = None
        if objective is not None:
            definition = dict(name=objective['name'], features=sorted(objective['goals']),
                conditions=sorted(r['feature'] for r in objective['requires']),
                roles={name:ctx['roles'][name]['description'] for name in roles_used(objective)})
            if 'arm' in objective:definition['arm']=objective['arm']
            method = digest(definition)
            if not self.frozen:
                self.bank['methods'].setdefault(method, dict(id=method, definition=definition,
                    evidence_ids=[], source_episodes=[], status='unobserved'))
        self.active = dict(id=f'e{self.episode}:objective{len(self.bank["invocations"])+1}',
            episode=self.episode, spec=deepcopy(spec), method=method, anchor=ctx['id'],
            roles=identities, origin=dict(source=source,program=program,observation_id=ctx['id']),
            status='active', evidence_ids=[], assessments=[])
        self.bank['invocations'].append(self.active)
        return True

    def observe(self, transition):
        if not transition.get('executed'):
            return
        identity = transition.get('experience_invocation')
        invocation = next((i for i in self.bank['invocations'] if i['id']==identity), None)
        if invocation is None or transition['id'] in invocation['evidence_ids']:
            return
        invocation['evidence_ids'].append(transition['id'])
        if invocation['method'] is None:
            return
        if not self.frozen:
            method = self.bank['methods'][invocation['method']]
            method['evidence_ids'].append(transition['id'])
            method['source_episodes'] = sorted(set(method['source_episodes']+[transition['episode']]))
            method['status'] = 'observed'
        objective = control_objective(invocation['spec'])
        assessment = condition(objective, transition['after'], invocation['roles'])
        before = condition(objective, transition['before'])
        progress = None
        if 'normalized_errors' in before and 'normalized_errors' in assessment:
            progress = sum(v*v for v in before['normalized_errors'].values()) - sum(v*v for v in assessment['normalized_errors'].values())
        invocation['assessments'].append(dict(transition=transition['id'], source=transition['source'],
            observed_error_reduction=progress, **assessment))
        invocation['status'] = 'active' if assessment['status']=='open' else assessment['status']

    def finish(self):
        if self.active is not None and self.active['status']=='active':
            self.active['status']='episode_ended'

    def objective(self):
        return control_objective(self.active['spec']) if self.active else None

    def assess(self, ctx):
        if self.active is None:
            return dict(kind='fallback', reason='current_control_objective_missing')
        spec = self.objective()
        if spec is None:
            return dict(kind='fallback', reason='probe_control_objective_missing', missing=self.active['spec']['reason'])
        check = condition(spec, ctx, self.active['roles'])
        if check['status'] != 'open':
            return dict(kind='ask', reason='measured_objective_complete' if check['status']=='complete' else check['reason'],
                objective=self.active['id'], assessment=check)
        return dict(kind='fallback', reason='current_objective_requires_program_or_teacher_action',
            objective=self.active['id'])

    def context(self):
        active = self.active
        return dict(active=None if active is None else {k:deepcopy(active[k]) for k in
            ('id','spec','method','anchor','roles','status')}, last_decision=deepcopy(self.last_decision),
            methods=[dict(id=m['id'], definition=m['definition'], samples=len(m['evidence_ids']),
                source_episodes=m['source_episodes'], status=m['status']) for m in list(self.bank['methods'].values())[-8:]],
            objective=deepcopy(self.objective()),
            total_methods=len(self.bank['methods']), scope='Control objectives and effect assessments are independent; not native task truth')
