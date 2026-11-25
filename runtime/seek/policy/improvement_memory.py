from collections import Counter
from copy import deepcopy
from .experience_access import available_ids


LESSON_GUIDANCE = '''
task_lessons contains retained, evidence-linked experience for this task. Before
choosing the first or next operation, apply a relevant lesson to the CURRENT scene:
when -> do -> expected observation; if that observation does not occur, use the
recorded alternative or gather the missing evidence. These are revisable hypotheses,
not task truth, fixed coordinates, or extra action gates. Model assistance is useful.
Optional using_lessons lists the lesson IDs that actually inform this proposal;
receipts record that attribution, not proof that the lesson caused success.
The accepted proposal's references also apply to local actions within its current
operation invocation (or current program/objective when no named operation is active).
A new teacher proposal updates this attribution; a different local invocation does
not inherit the previous invocation's references. Saved receipts retain their own
proposal reference. Operation lesson_references is the attribution history, not a
count of executed actions; only executed receipts establish recorded use.
In review, lessons may add or revise a named lesson with when/do/expect/if_failed
and saved evidence_ids. Cite receipts, numerical queries, errors or programs by
their existing IDs and preserve their meaning: numerical feasibility is not
physical progress, and a saved proposal is not an executed result.
Retain an actionable starting strategy and useful causal
or recovery distinctions even when they cannot yet be compiled into an operation.
Update the same name to refine it; status=retired withdraws a disproven lesson.
Compile supported repeated behavior into existing named operations when useful.
An unresolved effect can motivate a discriminating experiment or a best-supported
continuation hypothesis. A useful failure lesson includes the changed approach to
try next, not only a prohibition or a demand for more certainty. Native task results
must inform whether this strategy helped; textual caution is not learned success.
Learning is better decisions and task outcomes; fewer calls are not required.
'''


def lesson_schema():
    fields={k:dict(type='string') for k in ('name','when','do','expect','if_failed')}
    fields.update(status=dict(type='string',enum=['active','retired']),
                  evidence_ids=dict(type='array',items=dict(type='string')))
    return dict(type='array',items=dict(type='object',properties=fields,
                required=list(fields),additionalProperties=False))


def audit_lessons(proposals,memory):
    items=proposals if isinstance(proposals,list) else []
    known=available_ids(memory) if items else set()
    valid=[];ignored=[];details=[]
    for index,item in enumerate(items):
        if (not isinstance(item,dict) or set(item)!=set(lesson_schema()['items']['properties'])
                or any(not isinstance(item[k],str) or not item[k].strip()
                       for k in ('name','when','do','expect','if_failed'))
                or item['status'] not in ('active','retired')
                or not isinstance(item['evidence_ids'],list)
                or not item['evidence_ids']
                or any(not isinstance(i,str) or i not in known for i in item['evidence_ids'])):
            ignored.append(index)
            ids=item.get('evidence_ids') if isinstance(item,dict) else None
            unknown=sorted({i for i in ids if isinstance(i,str)}-known) if isinstance(ids,list) else []
            details.append(dict(index=index,unknown_evidence_ids=unknown,
                reason='Unavailable evidence references' if unknown else 'Invalid lesson fields'))
        else:valid.append(deepcopy(item))
    return dict(accepted=valid,ignored_indices=ignored,ignored_details=details,requires_model_retry=False)


def lesson_context(memory):
    selected={}
    for review in memory.get('improvement_history',[]):
        for lesson in review.get('lessons',[]):selected[lesson['name']]=lesson
    return [deepcopy(item) for item in selected.values() if item['status']=='active']


GUIDANCE = '''
improvement_history links each review hypothesis to later actual receipts under
its selected program. Trace retained knowledge to subsequent action generation,
measurements and operation choices; use observed capability and native outcomes to
retain, revise or revert a change. Requests and handoffs are cost diagnostics, not
learning targets. Layouts/parameters can confound comparisons;
no execution means unobserved. Source counts describe execution ownership, not
autonomous perception. learning_update is optional rationale and a next-episode
check; accepted code and exercised code are separate from demonstrated benefit.
'''


def execution_evidence(transitions):
    groups = {}
    for transition in transitions:
        if transition.get('executed') is not True:
            continue
        key = (transition['episode'], transition.get('program'))
        groups.setdefault(key, []).append(transition)
    result = []
    for (episode, program), rows in groups.items():
        examples = []
        for index in sorted({0, len(rows) - 1}):
            row = rows[index]
            examples.append(dict(
                id=row['id'], source=row['source'], action=deepcopy(row['action']),
                before=deepcopy(row['before']['features']),
                after=deepcopy(row['after']['features']),
            ))
        result.append(dict(
            episode=episode, program=program, executed_steps=len(rows),
            sources=dict(Counter(row['source'] for row in rows)),
            transition_ids=[row['id'] for row in rows], examples=examples,
        ))
    return result


def append_review(before, after, proposal, audit, errors=(), *, handoffs=None):
    if audit.get('accepted') is not True:
        return
    episode = before['episodes'][-1]['episode']
    record = dict(
        id=f"e{episode}:review{len(before.get('improvement_history', [])) + 1}",
        episode=episode, parent_program=before.get('active'),
        selected_program=after.get('active'), decision=audit['decision'],
        rationale=proposal['reason'], rationale_status='hypothesis',
        validation_status=audit['status'],
        reference=execution_evidence(t for t in before['transitions'] if t['episode'] == episode),

        episode_errors=dict(
            counts=dict(Counter(e.get('kind', 'unknown') for e in errors)),
            examples=[{k: deepcopy(e[k]) for k in ('kind', 'step', 'program', 'detail') if k in e}
                      for e in list(errors)[-4:]],
        ),
    )
    if handoffs is not None:
        record['handoff_reference'] = deepcopy({k:v for k,v in handoffs.items() if k != 'reasons'})
    learning_audit=audit.get('learning_audit')
    learning_update=learning_audit.get('update') if learning_audit is not None else proposal.get('learning_update')
    if learning_audit is not None:
        record['learning_feedback'] = deepcopy({k:v for k,v in learning_audit.items() if k!='update'})
    if learning_update is not None:
        record['learning_update'] = deepcopy(learning_update)
        record['learning_validation'] = 'unverified_hypothesis'
    lesson_audit=audit.get('lesson_audit') or audit_lessons(proposal.get('lessons',[]),before)
    record['lessons']=[dict(item,id=f"{record['id']}:lesson{i+1}",episode=episode)
                       for i,item in enumerate(lesson_audit['accepted'])]
    record['lesson_feedback']={k:deepcopy(v) for k,v in lesson_audit.items() if k!='accepted'}
    old=before.get('operation_knowledge',{});new=after.get('operation_knowledge',{})
    record['operation_selection_before']=deepcopy(old.get('selected',{}))
    record['operation_selection_after']=deepcopy(new.get('selected',{}))
    record['entry_before']=old.get('entry');record['entry_after']=new.get('entry')
    record['changed_operations']=[dict(name=name,version=version)
        for name,version in new.get('selected',{}).items() if old.get('selected',{}).get(name)!=version]
    after.setdefault('improvement_history', []).append(record)


def improvement_context(memory, *, include_sources=False, limit=3):
    history = memory.get('improvement_history', [])
    result = []
    for index in range(max(0, len(history) - limit), len(history)):
        record = deepcopy(history[index])
        end = history[index + 1]['episode'] if index + 1 < len(history) else None
        selected = record['selected_program']
        probe = execution_evidence(
            t for t in memory['transitions']
            if t['episode'] > record['episode']
            and (end is None or t['episode'] <= end)
            and t.get('program') == selected
        )
        record.update(
            observation_window_end_episode=end,
            subsequent_selected_execution=probe,
            evidence_status='observed_uncontrolled' if probe else 'unobserved',
            causal_improvement_established=False,
        )

        versions={item['version'] for item in record.get('changed_operations',[])}
        invocations=[i for i in memory.get('operation_knowledge',{}).get('invocations',[])
                     if i['episode']>record['episode'] and (end is None or i['episode']<=end)
                     and i['version'] in versions]
        record['subsequent_operation_execution']=[dict(
            id=i['id'],name=i['name'],version=i['version'],episode=i['episode'],
            evidence_ids=deepcopy(i['evidence_ids'])) for i in invocations if i.get('evidence_ids')]
        lesson_ids={item['id'] for item in record.get('lessons',[])}
        used=[t for t in memory['transitions'] if t.get('executed') is True
              and t['episode']>record['episode'] and lesson_ids.intersection(t.get('using_lessons',[]))]
        record['subsequent_lesson_execution']=[dict(id=t['id'],episode=t['episode'],
            using_lessons=sorted(lesson_ids.intersection(t['using_lessons']))) for t in used]
        if include_sources:
            ids = {record['parent_program'], selected}
            ids.update(group['program'] for group in record['reference'])
            record['program_sources'] = [
                {key: deepcopy(program[key]) for key in ('id', 'source', 'description')}
                for program in memory['programs'] if program['id'] in ids
            ]
        result.append(record)
    return result
