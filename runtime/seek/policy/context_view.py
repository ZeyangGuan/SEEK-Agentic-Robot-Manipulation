from collections import Counter
from copy import deepcopy
import json

VIEW = 'seek-context'
PRIOR_LIMIT = 12
HISTORY_LIMIT = 8
REVIEW_LIMIT = 20

GUIDANCE = '''
MODEL INPUT VIEW seek-context (not the executable ctx ABI):
Current context.robot/roles/features, mechanism_state, active_program, buffered_actions,
last_execution and last_error are complete. Historical receipts are SELECTED excerpts:
up to 12 prior receipts and the last 8 current receipts. Review uses up to 20 receipts,
prioritizing error neighborhoods, execution-source/program changes and tracking validity
or generation changes. Selection is not a success label. Missing evidence is unknown.
Each historical before/after has exact features and role identity/validity; duplicate
robot vectors and tracking internals are omitted. action is the executed delta;
requested_action is the proposal, not proof of motion. state_change lists changed
beliefs (before/after values and removed/added keys), never verified physical effects.
Each selected receipt includes its complete historical state_before anchor, so
unchanged method/parameter beliefs survive selection gaps. execution_note and
teacher_proposal_id link a demonstrated action to its accepted proposal.
Historical instruction and duplicate absolute command are omitted.
SEEK plan_reference/intended_target identify the original intent; controller_event
and joint_response preserve execution deviations. Historical joint_response keeps
actual response and target residual vectors; its duplicate before/after/target joint
arrays are available in the original receipt via recall_experience. They are not task-effect labels.
evidence_selection reports coverage and omissions. improvement history keeps hypotheses,
source counts, evidence status and examples; long ID lists are excerpts. Review program
sources are in improvement_programs, once per ID, referenced by program_source_ids.
The local decide(ctx,state) STILL receives the original last 24 history and up to 48
prior receipts with full fields; complete saved memory remains available through recall_experience.
recall_experience can retrieve omitted original records by ID or query. Do not
infer an omitted field is absent at runtime or an unshown transition never occurred.
No separate cache preparation request is required for this compact view.
'''


def _pick(value, keys):
    return {k: deepcopy(value[k]) for k in keys if k in value}


def snapshot(value):
    result = _pick(value, ('id', 'episode', 'step', 'features'))
    if 'roles' in value:
        result['roles'] = {name: _pick(role, (
            'camera', 'valid', 'generation', 'description', 'method', 'identity_reason'))
            for name, role in value['roles'].items()}
    return result


def receipt(value):
    result = _pick(value, ('id', 'episode', 'start', 'end', 'action',
        'requested_action', 'source', 'program', 'executed', 'matched_fit_evidence', 'tracking',
        'plan_reference','intended_target','controller_event','motion_evidence_id','queried_intent','motion_request',
        'experience_invocation','operation_invocation','state_before','teacher_proposal_id','execution_note','using_lessons'))
    if 'execution_receipt' in value:
        raw=value['execution_receipt']
        result['execution_receipt']=_pick(raw,('schema','reference','resulting_step','executed','controller_event'))
        if 'joint_response' in raw:
            result['execution_receipt']['joint_response']=[_pick(row,('arm','response','residual')) for row in raw['joint_response']]
            result['execution_receipt']['joint_response_omitted_fields']=['before','after','accepted_target']
    for key in ('before', 'after'):
        if key in value:
            result[key] = snapshot(value[key])
    if 'state_before' in value and 'state_after' in value:
        before, after = value['state_before'], value['state_after']
        if isinstance(before, dict) and isinstance(after, dict):
            result['state_change'] = dict(
                changed={k: dict(before=deepcopy(before[k]), after=deepcopy(after[k]))
                         for k in sorted(before.keys() & after.keys()) if before[k] != after[k]},
                added={k: deepcopy(after[k]) for k in sorted(after.keys() - before.keys())},
                removed={k: deepcopy(before[k]) for k in sorted(before.keys() - after.keys())})
        else:
            result['state_change'] = dict(before=deepcopy(before), after=deepcopy(after))
    return result


def _identity(row):
    return (row.get('source'), row.get('program'))


def _role_state(row, side):
    return {name: (r.get('valid'), r.get('generation'))
            for name, r in row.get(side, {}).get('roles', {}).items()}


def select(rows, limit, errors=()):
    if len(rows) <= limit:
        return list(rows)
    indices = []

    def add(i):
        if 0 <= i < len(rows) and i not in indices and len(indices) < limit:
            indices.append(i)

    add(0)
    for i in range(len(rows) - 1, max(-1, len(rows) - 5), -1):
        add(i)

    for error in reversed(list(errors)):
        step = error.get('step')
        if isinstance(step, int):
            for i, row in enumerate(rows):
                if row.get('start') == step or row.get('end') == step:
                    add(i)

    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(_identity(row), []).append(i)
    for group in sorted(groups.values(), key=lambda g: (len(g), g[0])):
        add(group[-1])
    for i in range(len(rows) - 1, 0, -1):
        if (_identity(rows[i]) != _identity(rows[i-1]) or
                _role_state(rows[i], 'before') != _role_state(rows[i], 'after') or
                _role_state(rows[i-1], 'after') != _role_state(rows[i], 'before')):
            add(i-1)
            add(i)
    while len(indices) < limit:
        add(max((i for i in range(len(rows)) if i not in indices),
                key=lambda i: (min(abs(i-j) for j in indices), -i)))
    return [rows[i] for i in sorted(indices)]


def coverage(rows, chosen, method):
    return dict(available=len(rows), included=len(chosen), omitted=len(rows)-len(chosen),
        selection=method, source_counts=dict(Counter(r.get('source', 'unknown') for r in rows)),
        included_ids=[r['id'] for r in chosen if 'id' in r],
        scope='Counts describe available receipts, not progress or task success')


def improvements(history):
    result = deepcopy(history)
    programs = {}
    for record in result:
        for key in ('reference', 'subsequent_selected_execution'):
            for group in record.get(key, []):
                ids = group.get('transition_ids', [])
                if len(ids) > 4:
                    group.update(transition_ids=ids[:2]+ids[-2:],
                                 transition_ids_total=len(ids), transition_ids_omitted=len(ids)-4)
        if 'program_sources' in record:
            sources = record.pop('program_sources')
            record['program_source_ids'] = [p['id'] for p in sources]
            for program in sources:
                programs[program['id']] = program
    return result, list(programs.values())


def action_view(packet):
    result = deepcopy(packet)
    ctx = result['context']
    prior, history = ctx['experience'], ctx['history']
    chosen_prior, chosen_history = select(prior, PRIOR_LIMIT), history[-HISTORY_LIMIT:]
    ctx.update(experience=[receipt(t) for t in chosen_prior],
               history=[receipt(t) for t in chosen_history])
    if 'motion_evidence' in ctx:
        ctx['motion_evidence']=[motion_receipt(r) for r in ctx['motion_evidence']]
    if 'effects' in ctx:
        effects=ctx['effects'];pending=[e for e in effects if e['status'] in ('pending','unknown')]
        chosen=list({e['id']:e for e in effects[-8:]+pending[-8:]}.values())
        ctx['effects']=chosen
        ctx['effect_evidence_coverage']=dict(available=len(effects),included=len(chosen),
            omitted=len(effects)-len(chosen),scope='Counts use all event assessments; full effects remain in local decide context')
    result['input_view'] = VIEW
    result['evidence_selection'] = dict(
        experience=coverage(prior, chosen_prior, 'source/program coverage and temporal gaps within runtime sample'),
        history=coverage(history, chosen_history, 'most recent actual receipts'),
        raw_memory_preserved=True)
    result['improvement_history'], _ = improvements(result.get('improvement_history', []))
    return result


def motion_receipt(row):
    result=_pick(row,('id','episode','observation_id','step','source','executed','requested_action','action','mode','configuration_distance_rad','plan_reference','target','solver_tolerances','space','requested_joint_deltas','requested_grippers'))
    def checks(values):
        return [_pick(c,('arm','passed','error','limit_rad','max_current_delta_rad','max_target_delta_rad')) for c in values]
    if 'result' in row:
        r=row['result'];trials=r.get('trials',[])
        result['result']=_pick(r,('status','action','fraction','reason','gripper_change','gripper_reference','kinematics','collision_checked','checks_scope','tool_base_relations'))
        selected=trials if len(trials)<=3 else [trials[0],trials[-2],trials[-1]]
        result['result'].update(trials=[dict(fraction=t['fraction'],feasible=t['feasible'],checks=checks(t['checks']),**_pick(t,('solver_errors',))) for t in selected],
                                tested_candidates=len(trials),omitted_candidates=len(trials)-len(selected))
    if 'detail' in row:result['checks']=checks(row['detail'].get('checks',[]))
    return result


def _dictionary_delta(before, after):
    assigned = {}
    nested = {}
    for key, value in after.items():
        if key in before and json.dumps(before[key], sort_keys=True) == json.dumps(value, sort_keys=True):
            continue
        if key in before and isinstance(before[key], dict) and isinstance(value, dict):
            nested[key] = _dictionary_delta(before[key], value)
        else:
            assigned[key] = deepcopy(value)
    result = {}
    if assigned:result['set'] = assigned
    if nested:result['update'] = nested
    removed = [key for key in before if key not in after]
    if removed:result['remove'] = removed
    return result


def review_view(packet, transitions=None):
    result = deepcopy(packet)
    if 'motion_evidence' in result:
        rows=result['motion_evidence']
        result['motion_evidence']=[motion_receipt(r) for r in rows[-REVIEW_LIMIT:]]
        result['motion_evidence_coverage']=dict(available=len(rows),included=min(len(rows),REVIEW_LIMIT),selection='most recent numerical and rejection evidence')
    rows = packet['actual_transitions'] if transitions is None else transitions
    chosen = select(rows, REVIEW_LIMIT, packet.get('errors', []))
    result['actual_transitions'] = [receipt(t) for t in chosen]
    result['input_view'] = VIEW
    result['evidence_selection'] = coverage(rows, chosen, 'errors, sources/programs, tracking changes, temporal gaps')
    initial = result.get('initial_context')
    if initial is not None:
        for key in ('experience', 'history'):
            omitted = initial.pop(key, [])
            initial[key+'_omitted_count'] = len(omitted)
    result['improvement_history'], result['improvement_programs'] = improvements(
        result.get('improvement_history', []))

    active = result.get('active_program')
    if active is not None:
        result['improvement_programs'] = [p for p in result['improvement_programs'] if p['id'] != active['id']]
        result['active_program_source_location'] = 'active_program (also resolves matching program_source_ids)'
    return result
