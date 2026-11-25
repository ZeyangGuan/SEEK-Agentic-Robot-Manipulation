from collections import Counter
from copy import deepcopy
from hashlib import sha256

from .mechanism_program import ProgramError
from .experience_access import available_ids


GUIDANCE = '''
Review executed operation sequences, retained experience and later reuse, then retain,
revise or replace the method responsible for the failure. The scope may include
action generation, parameter derivation, operation composition and observe/decide.
Learn current-parameter derivation, applicability, event
measurements and next-intent selection from recorded observations and receipts.
Prioritize the recurring failure that prevents task progress. Follow the complete
chain from contact or method selection through execution and later effect changes,
including contrasting executions of the same operation. Earlier detection alone
does not solve a failing method. Connect a useful new measurement to a changed
action, parameter derivation or productive recovery, and explain which actual
consequence would distinguish improvement from another abort. A provisional
intervention grounded in recorded evidence can be learned before its physical
benefit is proven; mark that uncertainty rather than limiting every review to
bookkeeping or another condition that only asks for help. Do not add a separate
test or model call just to satisfy this guidance.
A missing final-effect label need not block a supported partial rule or an
information-gathering rule. Name the measurement needed for the unresolved branch.
Current layout coordinates and model explanations are not reusable physical laws.
Use continue/delegate for supported local transitions. A semantic ask needs model
input; blocked terminates the episode. Fresh entry may still need a scene binding.
learning_update is optional: describe the supported capability change, evidence,
remaining gap and what later behavior would demonstrate useful reuse. target_reason_id
may be null; a capability need not correspond to a handoff. Existing observer_change
and transition_change fields may describe numerical or information-gathering learning
without a code replacement. Invalid or absent annotations do not trigger a repair call.
Accepted changes are provisional; later actions and outcomes establish their usefulness.
Calls are cost diagnostics and need not decrease on each episode.
Use the existing read-only recall/test tools when useful, not to fill a checklist.
'''


def learning_schema():
    properties = {key: dict(type='string') for key in (
        'observer_change', 'transition_change', 'expected_check', 'unresolved')}
    properties.update(status=dict(type='string', enum=['candidate', 'deferred']),
                      target_reason_id=dict(type=['string', 'null']),
                      evidence_ids=dict(type='array', items=dict(type='string')))
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def handoff_evidence(episode, decisions, teachers, transitions, report, limit=6):
    actual = [t for t in transitions if t.get('episode') == episode and t.get('executed') is True]
    snapshots, before, after = {}, {}, {}
    for row in actual:
        before[row.get('end')] = row['id']
        after.setdefault(row.get('start'), row['id'])
        for side in ('before', 'after'):
            obs = row.get(side, {})
            if isinstance(obs.get('step'), int) and obs.get('id'):
                snapshots[obs['step']] = obs['id']
    teachers_at = {}
    for row in teachers:
        teachers_at.setdefault(row.get('step'), []).append(row)
    groups = {}
    for row in decisions:
        if row.get('decision', {}).get('kind') != 'ask':
            continue
        reason = row['decision'].get('reason') or '(unspecified)'
        groups.setdefault(reason, []).append(row)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    reasons = []
    for reason, events in ordered:
        entry = dict(reason_id=sha256(reason.encode()).hexdigest()[:16], reason=reason,
                     count=len(events), programs=dict(Counter(str(r.get('program')) for r in events)))
        if len(reasons) < limit:
            examples = []
            for index in sorted({0, len(events)-1}):
                event = events[index]; step = event.get('step'); responses = teachers_at.get(step, [])
                examples.append(dict(step=step, program=event.get('program'),
                    observation_id=snapshots.get(step), preceding_transition_id=before.get(step),
                    subsequent_transition_id=after.get(step),
                    teacher_updates_at_same_observation=[dict(
                        id=r.get('id') or f'e{episode}:teacher{r.get("turn", "unknown")}',
                        execution_note=r.get('execution_note', '')[:600],
                        state_update_keys=sorted((r.get('state_update') or {}).keys())) for r in responses[:2]],
                    teacher_updates_at_same_observation_total=len(responses)))
            entry['examples'] = examples
        reasons.append(entry)
    return dict(episode=episode, ask_decisions=sum(len(v) for v in groups.values()),
                decision_counts=dict(Counter(r.get('decision', {}).get('kind', 'unknown') for r in decisions)),
                accepted_teacher_updates=len(teachers),
                episode_api_requests=report.get('model_action_requests'),
                executed_steps=len(actual), program_steps=dict(Counter(str(t.get('program')) for t in actual)),
                native_result={k: report[k] for k in ('success', 'score', 'final_step') if k in report},
                reasons=reasons[:limit], reason_counts=[{k: r[k] for k in ('reason_id', 'count', 'programs')} for r in reasons],
                omitted_example_groups=max(0, len(reasons)-limit),
                scope='Actual ask counts; API requests include lookups/retries. Teacher beliefs and subsequent receipts do not establish successful effects or causal autonomy gains.')


def validate_learning_update(proposal, memory, handoffs, evidence_memory=None):
    update = proposal.get('learning_update')

    if update is None:
        return
    fields = set(learning_schema()['properties'])
    if not isinstance(update, dict) or set(update) != fields:
        raise ProgramError('SEEK learning_update must describe the capability hypothesis and later-behavior check')
    if update['status'] not in ('candidate', 'deferred'):
        raise ProgramError('Learning status must be candidate or deferred, never verified')
    for key in ('observer_change', 'transition_change', 'expected_check', 'unresolved'):
        if not isinstance(update[key], str) or (key != 'unresolved' and not update[key].strip()):
            raise ProgramError(f'Learning {key} must be explicit text')
    targets = {r['reason_id'] for r in handoffs.get('reasons', [])+handoffs.get('reason_counts', [])}
    target = update['target_reason_id']
    if target is not None and target not in targets:
        raise ProgramError('Optional handoff reference must be a listed reason or null')
    ids = update['evidence_ids']
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
        raise ProgramError('Learning evidence_ids must be actual observation/transition IDs')
    actual_ids = set()
    for row in memory['transitions']:
        if row.get('executed') is True:
            actual_ids.add(row['id'])
            actual_ids.update(row[s]['id'] for s in ('before', 'after') if row.get(s, {}).get('id'))
    known=available_ids(evidence_memory if evidence_memory is not None else memory)|actual_ids
    if not set(ids) <= known or (update['status']=='candidate' and not actual_ids.intersection(ids)):
        raise ProgramError('A capability hypothesis must cite executed evidence; other saved references may accompany it')
    if update['status'] == 'deferred' and not update['unresolved'].strip():
        raise ProgramError('Deferred learning must name missing evidence or capability')


def audit_learning_update(proposal, memory, handoffs, evidence_memory=None):
    try:
        validate_learning_update(proposal, memory, handoffs, evidence_memory)
    except (ProgramError, ValueError, TypeError, KeyError) as error:
        return dict(status='unresolved_metadata', detail=str(error), update=None,
                    requires_model_retry=False)
    update=proposal.get('learning_update')
    return dict(status='unverified_hypothesis' if update is not None else 'not_supplied',
                update=deepcopy(update), requires_model_retry=False)


def previous_review_check(memory, current):
    result = []
    for record in memory.get('improvement_history', [])[-3:]:
        reference = record.get('handoff_reference'); learning = record.get('learning_update')
        if not reference or not learning:
            continue
        target = learning['target_reason_id']; selected = str(record.get('selected_program'))
        old = next((r for r in reference['reason_counts'] if r['reason_id'] == target), None)
        new = next((r for r in current['reason_counts'] if r['reason_id'] == target), None)
        count = new['programs'].get(selected, 0) if new else None
        result.append(dict(review_id=record['id'], learning_update=deepcopy(learning),
            reference_episode=reference['episode'], observed_episode=current['episode'],
            reference_ask_count=old['count'] if old else None,
            current_ask_count_under_selected_program=count,
            selected_program_executed_steps=current['program_steps'].get(selected, 0),
            reference_api_requests=reference['episode_api_requests'], current_api_requests=current['episode_api_requests'],
            reference_native_result=deepcopy(reference['native_result']), current_native_result=deepcopy(current['native_result']),
            status=('capability_check_pending' if target is None else
                'target_handoff_recurs' if count else 'unresolved_not_proven_by_absent_or_renamed_reason'),
            causal_improvement_established=False))
    return result
