from copy import deepcopy
from collections import Counter
import json

from .context_view import receipt
from .prompt_layout import restore_action_packet


GUIDANCE = '''
Learn the decision that connects operations, as well as the motion inside one.
boundary_evidence pairs actual PRE-teacher context/state with the teacher's answer
and subsequent executed receipts. call_route follows the entire accepted-response
route, including zero-step updates; call_span separates proposed candidate rows,
their execution, other local steps, and the next request. A high local-step share
can still mean the model chooses every short waypoint. Inspect what each call supplied.
When a repeated request is answered from quantities already measurable locally,
retain that parameter derivation, progression, stopping condition and next decision
in the existing observe/decide program. Use continue/intent/operation to cross a
supported boundary locally. A numeric endpoint needs the model only if the NEXT choice
requires missing scene information or an interpretation the learned code cannot make.
For recurring teacher action sequences, teach a parameterized operation that reads
each real receipt and can continue, stop, change method or request the specific gap.
Do not reduce this to another fixed-distance motion followed by an unconditional ask.
Use the current necessary response for teaching; no separate training call is needed.
Teacher actions, beliefs and later corrections are demonstrations, not physical labels.
The following teacher answer must never become an input to an earlier replay. Old
contexts not recorded remain unavailable. Derive new-layout parameters from current
observations; retain useful earlier behavior and explicit uncertainty about effects.
Call count and batch length are diagnostics, not optimization targets or quotas.
A genuinely new visual binding, unsupported recovery or unresolved interpretation
may require the model. The test is whether retained experience supports a better next
decision and eventual task completion, not whether the system avoids asking.
test_mechanism uses real saved contexts/images without motion or installation.
Use cases before and after the observed boundary to inspect a proposed continuation;
replay cannot certify the effect of an unexecuted action or future native success.
It shares the existing read-only response slots with recall_experience.
'''


def capture_boundary(ctx, state, reason, after_turn):
    return dict(id=f"{ctx['id']}:after{after_turn}", step=ctx['step'], after_turn=after_turn,
        context=deepcopy(ctx), state=deepcopy(state), reason=deepcopy(reason),
        scope='Actual context before asking; no future teacher values included')


def load_contexts(trace):
    path = trace/'seek_learning_contexts.jsonl'
    if path.exists():
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]

    result = []
    path = trace/'exchanges.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                try:
                    packet = restore_action_packet(row['request']['input'])
                    ctx, state = packet['context'], packet['mechanism_state']
                    record = capture_boundary(ctx, state, ctx.get('last_error'), row['turn']-1)
                except (KeyError, ValueError, TypeError):
                    continue
                record['scope'] = 'Actual pre-teacher model packet; historical context was compacted, current observation/state complete'
                result.append(record)
    return result


def boundary_evidence(episode, decisions, teachers, transitions, contexts=(), limit=8, invocations=()):
    actual = [t for t in transitions if t.get('episode') == episode and t.get('executed') is True]
    asks = [r for r in decisions if r.get('decision', {}).get('kind') == 'ask']
    teachers = sorted(enumerate(teachers), key=lambda p: (p[1]['step'], p[0]))
    cases = []
    assessments={a['transition']:a for invocation in invocations for a in invocation.get('assessments',[])}
    for index, (_, teacher) in enumerate(teachers):
        step, turn = teacher['step'], teacher['turn']
        entries = [c for c in contexts if c['step'] == step and c['after_turn'] < turn]
        saved = max(entries, key=lambda c: c['after_turn']) if entries else None
        reasons = [r['decision'].get('reason', '') for r in asks if r['step'] == step]
        end = teachers[index+1][1]['step'] if index+1 < len(teachers) else None
        following = [t for t in actual if t['start'] >= step and (end is None or t['start'] < end)]
        preceding = next((t for t in reversed(actual) if t['end'] == step), None)
        demonstration = {k: deepcopy(teacher[k]) for k in
            ('actions', 'program', 'intent', 'bindings', 'state_update', 'effects', 'candidate_guard', 'execution_note', 'operations', 'operation') if k in teacher}

        state = saved['state'] if saved else teacher.get('state_before')
        ctx = saved['context'] if saved else None
        identity=teacher.get('id',f'e{episode}:teacher{turn}')
        following_context=None
        if index+1<len(teachers):
            following_teacher=teachers[index+1][1]
            candidates=[c for c in contexts if c['step']==following_teacher['step'] and c['after_turn']<following_teacher['turn']]
            following_context=max(candidates,key=lambda c:c['after_turn']) if candidates else None
        assessments_before={e['id']:e for e in (ctx or {}).get('effects',[])}
        assessments_before.update((e['id'],e) for e in teacher.get('effects',[]))
        assessments_after={e['id']:e for e in
            ((following_context or {}).get('context') or {}).get('effects',[])}

        if index+1<len(teachers):
            assessments_after.update((e['id'],e) for e in teachers[index+1][1].get('effects',[]))
        assessment_changes=[dict(id=e['id'],name=e.get('name'),
            before=assessments_before.get(e['id'],{}).get('status'),after=e['status'],
            evidence_ids=deepcopy(e.get('evidence_ids',[])),reason=e.get('reason'),
            scope='Changed model assessment during this interval, not native truth or causal attribution')
            for e in assessments_after.values()
            if e['status']!=assessments_before.get(e['id'],{}).get('status')]
        candidate_execution=((following_context or {}).get('context',{}).get('teacher_execution') or {})
        if candidate_execution.get('teacher_proposal_id')!=identity:candidate_execution=None
        attributed=[t for t in following if t.get('teacher_proposal_id')==identity]
        call_span=dict(candidate_rows_proposed=len(teacher.get('actions',[])),
            teacher_candidate_receipts=[t['id'] for t in attributed],
            execution_sources=dict(Counter(t.get('source') for t in following)),
            physical_steps_until_next_teacher=len(following),
            next_request=deepcopy((following_context or {}).get('reason')),
            candidate_execution_at_next_request=deepcopy(candidate_execution),
            interval_complete=end is not None,
            scope='Actual receipts; other local actions are not executions of the proposed candidate rows')
        cases.append(dict(id=f'e{episode}:boundary{turn}', episode=episode, step=step,
            observation_id=f'e{episode}:s{step}', teacher_id=teacher.get('id', f'e{episode}:teacher{turn}'),
            request_reason=deepcopy(saved.get('reason')) if saved else None,
            ask_reasons=list(dict.fromkeys(reasons)), same_step_teacher_count=sum(t['step']==step for _, t in teachers),
            context_available=ctx is not None, context=deepcopy(ctx), state_before=deepcopy(state),
            context_scope=saved.get('scope') if saved else 'Pre-teacher context unavailable',
            teacher_demonstration=demonstration, proposed_actions=len(teacher.get('actions', [])),call_span=call_span,
            preceding_transition=receipt(preceding) if preceding else None,
            actual_following=[receipt(t) for t in following],
            numeric_assessments=[deepcopy(assessments[t['id']]) for t in following if t['id'] in assessments],
            effect_assessment_changes=assessment_changes,
            following_teacher=({k:deepcopy(teachers[index+1][1][k]) for k in
                ('id','step','turn','intent','bindings','execution_note','state_update','effects') if k in teachers[index+1][1]}
                if index+1 < len(teachers) else None),
            actual_following_ids=[t['id'] for t in following],
            next_teacher_step=end, execution_is_not_effect_confirmation=True))
    selected = select_operation_cases(cases, limit)
    route=[dict(id=c['id'],step=c['step'],teacher_id=c['teacher_id'],
        request=deepcopy(c['request_reason']),
        proposed_actions=c['proposed_actions'],executed_candidate_rows=len(c['call_span']['teacher_candidate_receipts']),
        physical_steps=c['call_span']['physical_steps_until_next_teacher'],
        next_teacher_step=c['next_teacher_step'],next_request=c['call_span']['next_request']) for c in cases]
    return dict(cases=selected, call_route=route,total_cases=len(cases), omitted_cases=len(cases)-len(selected),
        contexts_available=sum(c['context_available'] for c in cases),
        scope='Joined evidence, not success labels or proof that a next goal was independently inferred')


def operation_key(case):
    operation=case.get('teacher_demonstration',{}).get('operation')
    if isinstance(operation,dict) and operation.get('name'):
        return ('operation',operation['name'])
    intent=case.get('teacher_demonstration',{}).get('intent') or {}
    control=intent.get('control') if intent.get('kind')=='probe' else intent
    if control and control.get('goals'):
        return ('control',control.get('name'),control.get('arm'),tuple(sorted(control['goals'])))
    channels=tuple(sorted({i for t in case.get('actual_following',[])
        for i,value in enumerate(t.get('action',[])) if value != 0}))
    return (intent.get('kind'),channels)


def challenged_effect(case):
    return any(e['after']=='refuted'
        or (e['before'] in (None,'confirmed') and e['after']=='unknown')
        for e in case.get('effect_assessment_changes',[]))


def boundary_index(evidence):
    return [dict(id=c['id'],episode=c['episode'],step=c['step'],
        observation_id=c['observation_id'],teacher_id=c['teacher_id'],
        preceding_transition_id=(c.get('preceding_transition') or {}).get('id'),
        actual_following_ids=deepcopy(c['actual_following_ids']),
        next_teacher_step=c['next_teacher_step'],context_available=c['context_available'],
        scope='Recorded observation/proposal/receipt association; not an effect or success label')
        for c in evidence['cases']]


def selected_boundary_view(evidence, limit=8):
    selected=select_operation_cases(evidence['cases'],limit)
    view=dict(evidence,cases=selected,omitted_cases=evidence['total_cases']-len(selected))

    predecessors=assessment_predecessors(evidence['cases'])
    view['visual_assessment_predecessors']={c['id']:predecessors[c['id']]
        for c in selected if c['id'] in predecessors}
    return boundary_view(view)


def assessment_predecessors(cases):
    previous={};result={}
    for case in sorted(cases,key=lambda c:(c['episode'],c['step'])):
        candidates=[]
        for change in case.get('effect_assessment_changes',[]):
            key=(case['episode'],change['id'])
            earlier=previous.get(key)
            if (change.get('before')=='unknown' and change.get('after')=='refuted'
                    and earlier is not None and earlier[1].get('after')=='unknown'
                    and earlier[0].get('actual_following')):
                onset=earlier[0]
                end=max([onset['step']]+[t['end'] for t in onset['actual_following']
                    if type(t.get('end')) is int])
                if end<=case['step']:
                    candidates.append(dict(case_id=onset['id'],effect_id=change['id'],
                        episode=onset['episode'],start_step=onset['step'],end_step=end,
                        scope='Earlier recorded uncertainty for this event; not a causal label'))
            previous[key]=(case,change)
        if candidates:
            result[case['id']]=max(candidates,key=lambda c:c['start_step'])
    return result


def select_operation_cases(cases, limit=8, current=None):
    ordered=sorted(cases,key=lambda c:(c['episode'],c['step'],int(c['id'].rsplit('boundary',1)[1])))
    if not ordered or limit<=0:return []
    if current is None and len(ordered)<=limit:return ordered
    executed=[c for c in ordered if c['actual_following_ids']]
    selected=[]
    def add(case):
        if case not in selected and len(selected)<limit:selected.append(case)
    if current is not None:
        active=(current.get('experience_control') or {}).get('active') or {}
        operation=(current.get('operation_knowledge') or {}).get('active')
        key=operation_key(dict(teacher_demonstration=dict(intent=active.get('spec'),operation=operation)))
        matching=[c for c in executed if operation_key(c)==key]
        primary=(matching or executed or ordered)[-1];add(primary)
        support={i for t in primary['actual_following'] for fit in t.get('matched_fit_evidence',[])
            for i in fit.get('evidence_ids',[])}
        others=[c for c in executed if c is not primary]

        related=sorted(others,key=lambda c:(bool(support.intersection(c['actual_following_ids'])),
            operation_key(c)==operation_key(primary),c['episode']<primary['episode'],
            any(t.get('matched_fit_evidence') for t in c['actual_following']),c['episode'],c['step']),reverse=True)
        for case in related:add(case)
        if len(selected)<limit:
            for case in reversed(ordered):add(case)
    else:
        add(ordered[0]);add(ordered[-1])

        challenged=[c for c in executed if challenged_effect(c)]
        recurring={}
        for case in challenged:recurring.setdefault(operation_key(case),[]).append(case)
        groups=sorted(recurring.values(),key=lambda group:
            (len(group),group[-1]['episode'],group[-1]['step']),reverse=True)
        for group in groups[:2]:
            case=group[-1]
            add(case)
            peers=[c for c in executed if c is not case
                and operation_key(c)==operation_key(case)]
            if peers:
                peer=max(peers,key=lambda c:(c not in challenged,c['episode'],c['step']))
                add(peer)
        for case in executed:
            if any(t.get('matched_fit_evidence') for t in case['actual_following']):add(case)
        seen={operation_key(c) for c in selected}
        for case in reversed(executed):
            key=operation_key(case)
            if key not in seen:add(case);seen.add(key)
        for case in reversed(executed):add(case)
    return selected


def execution_sequence(rows):
    return [{k:deepcopy(row[k]) for k in ('id','episode','start','end','source','executed',
        'action','teacher_proposal_id','experience_invocation','operation_invocation','matched_fit_evidence',
        'controller_event','before','after') if k in row} for row in rows]


def boundary_view(evidence):
    result = deepcopy(evidence)
    for case in result['cases']:
        ctx = case.pop('context')
        if ctx is not None:
            case['pre_teacher_context'] = {k: ctx[k] for k in
                ('id', 'step', 'features', 'roles', 'last_execution', 'last_error',
                 'experience_control', 'effects', 'effect_counts', 'teacher_execution', 'operation_knowledge') if k in ctx}
        case['actual_following'] = execution_sequence(case['actual_following'])
    return result


def visual_step_candidates(evidence):
    cases=sorted(evidence['cases'],key=lambda c:c['step'])
    result=[]
    if not cases:return result
    def end(case):
        return max([case['step']]+[t['end'] for t in case.get('actual_following',[])
                                  if type(t.get('end')) is int])

    last=max(end(case) for case in cases)
    def add(step):
        if type(step) is int and 0<=step<=last and step not in result:result.append(step)
    add(last)

    for case in evidence['cases']:
        if not case.get('actual_following') or not challenged_effect(case):continue
        predecessor=evidence.get('visual_assessment_predecessors',{}).get(case['id'])
        if predecessor is not None:
            add(predecessor['start_step']);add(predecessor['end_step'])
        add(case['step']);add(end(case))
        peers=[c for c in cases if c is not case and c.get('actual_following')
               and operation_key(c)==operation_key(case)]
        if peers:
            peer=max(peers,key=lambda c:(not challenged_effect(c),any(e['after']=='confirmed'
                for e in c.get('effect_assessment_changes',[])),c['step']))
            add(peer['step']);add(end(peer))
        break
    add(cases[0]['step'])
    events={}
    for case in cases:
        ctx=case.get('context') or case.get('pre_teacher_context') or {}
        for effect in (ctx.get('effects') or [])+((case.get('teacher_demonstration') or {}).get('effects') or []):
            events[effect['id']]=(case,effect)
    for case,effect in sorted(events.values(),key=lambda p:(p[1].get('status') in ('pending','unknown'),p[0]['step']),reverse=True):
        if effect.get('status') not in ('pending','unknown'):continue
        episode=case.get('episode') or (case.get('context') or {}).get('episode')
        for identity in effect.get('evidence_ids',[]):
            prefix,sep,value=identity.partition(':s')
            if sep and value.isdigit() and (episode is None or prefix==f'e{episode}'):
                add(int(value))

    for case in list(reversed(cases)):
        add(case['step'])
        preceding=case.get('preceding_transition')
        if preceding:add(preceding.get('start'))
        for transition in case.get('actual_following',[])[:1]:add(transition.get('end'))
    return result


def selected_steps(evidence,limit=6):
    return visual_step_candidates(evidence)[:limit]


def read_observations(path, steps, observed_steps=None):
    wanted = set(steps)
    result = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                step=row.get('step')
                if observed_steps is not None and type(step) is int:observed_steps.add(step)
                if step in wanted:
                    result[step] = row
    return result


def image_parts(evidence, archive):
    parts, coverage = [], []
    for step in selected_steps(evidence):
        row = archive.get(step)
        if row is None:
            coverage.append(dict(step=step, status='missing'))
            continue
        cameras = []
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            if camera not in row.get('images', {}):
                continue
            parts.extend([dict(type='input_text', text=f'Historical real observation s{step}, {camera}; camera-local coordinates.'),
                          dict(type='input_image', image_url=row['images'][camera], detail='high')])
            cameras.append(camera)
        coverage.append(dict(step=step,cameras=cameras,
            missing_cameras=[c for c in ('cam_head','cam_left_wrist','cam_right_wrist') if c not in cameras],
            status='available' if cameras else 'missing'))
    coverage.extend(dict(step=s,status='omitted_budget') for s in visual_step_candidates(evidence) if s not in selected_steps(evidence))
    return parts, coverage


def learning_agenda(memory):
    result = []
    for review in memory.get('improvement_history', [])[-3:]:
        update = review.get('learning_update')
        if not update:
            continue
        result.append(dict(review_id=review['id'], episode=review['episode'], status=update['status'],
            unresolved=update['unresolved'], expected_check=update['expected_check'],
            observer_change=update['observer_change'], transition_change=update['transition_change']))
    return dict(items=result, request_policy='Use this already-needed response to teach a reusable partial rule or collect missing measurements; never ask solely for this agenda',
        reset_policy='Rules persist; current scene bindings, coordinates, counts and runtime state do not')


def online_learning_agenda(memory, episode, contexts, current):
    contexts=[c for c in contexts if c['context'].get('episode')==episode
              and c['step']<=current['step']]
    teachers=[t for t in memory.get('teacher_updates',[]) if t.get('episode')==episode
              and type(t.get('step')) is int and type(t.get('turn')) is int
              and t['step']<=current['step'] and any(c['step']==t['step']
                  and c['after_turn']<t['turn'] for c in contexts)]
    def reason_name(reason):
        detail=(reason or {}).get('detail','initial_scene')
        return detail if isinstance(detail,str) else json.dumps(detail,sort_keys=True,ensure_ascii=False)
    decisions=[dict(step=c['step'],decision=dict(kind='ask',
        reason=reason_name(c.get('reason')))) for c in contexts]
    transitions=[t for t in memory.get('transitions',[]) if t.get('episode')==episode
                 and t.get('end',current['step']+1)<=current['step']]
    evidence=boundary_evidence(episode,decisions,teachers,transitions,contexts,limit=len(teachers),
        invocations=memory.get('control_experience',{}).get('invocations',[]))

    cases=evidence['cases']
    prior_teachers=[t for t in memory.get('teacher_updates',[]) if
        t.get('episode',episode)<episode and type(t.get('step')) is int and type(t.get('turn')) is int]
    for prior in sorted({t['episode'] for t in prior_teachers}):
        teachers_at=[t for t in prior_teachers if t['episode']==prior]
        old=boundary_evidence(prior,[],teachers_at,memory.get('transitions',[]),
            limit=len(teachers_at),invocations=memory.get('control_experience',{}).get('invocations',[]))
        cases.extend(old['cases'])
    evidence['total_cases']=len(cases)

    recorded={r['id']:r for r in memory.get('learning_boundaries',[])}
    for record in boundary_index(evidence):
        if record['episode']==episode or record['id'] not in recorded:
            recorded[record['id']]=record
    memory['learning_boundaries']=list(recorded.values())
    chosen=select_operation_cases(cases,2,current)
    evidence.update(cases=chosen,omitted_cases=evidence['total_cases']-len(chosen))
    view=boundary_view(evidence)

    view.pop('call_route',None)
    def fields(values, maximum=2400):
        included={};omitted=[];used=0
        for key,value in (values or {}).items():
            size=len(json.dumps({key:value},ensure_ascii=False).encode())
            if size>800 or used+size>maximum:omitted.append(key)
            else:included[key]=deepcopy(value);used+=size
        return included,omitted
    compact=[]
    for case in view['cases']:
        pre=case.get('pre_teacher_context') or {};demo=case['teacher_demonstration']
        before=case.get('state_before') or {};after=demo.pop('state_update',None)
        state,omitted=fields(before)
        active=(pre.get('experience_control') or {}).get('active')
        pre={k:deepcopy(pre[k]) for k in ('id','step','features','roles') if k in pre}
        pre['roles']={name:{k:deepcopy(role[k]) for k in ('valid','generation','description') if k in role}
                      for name,role in pre.get('roles',{}).items()}
        if active:pre['control_intent']=deepcopy(active['spec'])
        if isinstance(after,dict):
            changed={k:deepcopy(v) for k,v in after.items() if k not in before or before[k]!=v}
            changes,missing=fields(changed)
            demo['belief_update_delta']=dict(changed=changes,removed=[k for k in before if k not in after],
                omitted_changed_keys=missing,native_truth=False)
        effects=demo.pop('effects',[])
        demo['effect_assessments']=[{k:deepcopy(e[k]) for k in ('id','name','status','evidence_ids') if k in e} for e in effects]
        following=case['actual_following']
        item={k:deepcopy(case[k]) for k in ('id','episode','step','observation_id','teacher_id','ask_reasons',
            'context_available','context_scope','actual_following_ids','proposed_actions','next_teacher_step',
            'numeric_assessments','following_teacher','call_span')}
        item.update(pre_teacher_context=pre,state_before=state,
            state_projection=dict(omitted_keys=omitted),
            teacher_demonstration=demo,actual_following=following)
        compact.append(item)

    view.update(cases=compact,omitted_cases=view['total_cases']-len(compact),
        online_case_bytes=sum(len(json.dumps(c,ensure_ascii=False).encode()) for c in compact),
        selection='Executed operation and related evidence, including prior episodes; at most two complete cases',
        scope='Teacher beliefs and actual receipts are distinct. Projected beliefs omit listed keys; full contexts remain the replay inputs.')
    return dict(episode=episode,current_reason=(current.get('last_error') or {}).get('detail'),
        boundary_evidence=view,
        request_policy='Use these operation sequences to retain or extend a reusable capability, or choose a useful information action. Learning is not measured by fewer handoffs; no separate training call is added.')
