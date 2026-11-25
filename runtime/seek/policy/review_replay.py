from copy import deepcopy

from .boundary_learning import read_observations
from .effect_evidence import measure_patch, update_effects, current_effects, effect_counts
from .experience_access import available_ids, lookup
from .experience_control import ExperienceControl, measured_features
from .mechanism_program import Program, ProgramError, digest, source_schema
from .visual_measurements import measure_region
from .rgb_perception import measure_rgb, locate_rgb_point, segment_rgb
from .operation_knowledge import OperationKnowledge, prepare_updates, updates_schema, invocation_state


def replay_tool():
    return dict(type='function', name='test_mechanism', strict=False,
        description='Read-only observe/decide replay on any recorded boundary ID in boundary_evidence.call_route or cases, or fresh_entry. Shares the review lookup budget; no motion, no installation.',
        parameters=dict(type='object', additionalProperties=False,
            properties=dict(source=source_schema(),operations=updates_schema(),
                case_ids=dict(type='array', minItems=1, maxItems=4, items=dict(type='string'))),
            required=['source', 'case_ids']))


def replay(source, case_ids, evidence, memory, observation_path, initial_context=None, operations=None):
    if not isinstance(case_ids, list) or not 1 <= len(case_ids) <= 4 or any(not isinstance(v, str) for v in case_ids):
        raise ValueError('Replay needs 1..4 listed boundary IDs')
    main_program = Program(source)

    candidate_evidence_ids = available_ids(memory)
    cases = {c['id']: c for c in evidence['cases']}
    if initial_context is not None:
        cases['fresh_entry'] = dict(step=initial_context['step'], context=initial_context, state_before={})
    if any(key not in cases for key in case_ids):
        raise ValueError('Unknown boundary ID; use boundary_evidence.call_route, cases or fresh_entry')

    observed_steps=set()
    archive = read_observations(observation_path, {cases[key]['step'] for key in case_ids},observed_steps)
    loaded_steps=set(archive)
    results = []
    for key in case_ids:
        case = cases[key]
        ctx = deepcopy(case.get('context'))
        full_context = ctx is not None
        if not full_context:
            preceding = case.get('preceding_transition')
            ctx = deepcopy(preceding.get('after')) if preceding else None
        if ctx is None or case.get('state_before') is None:
            results.append(dict(case_id=key, status='unavailable_pre_teacher_context', executed=False))
            continue
        ctx['features'] = measured_features(ctx)
        program=main_program
        state = deepcopy(case['state_before'])
        step = case['step']
        measurements = []

        past = dict(transitions=list({t['id']:t for t in ctx.get('experience', [])+ctx.get('history', [])}.values()),
                    effects=deepcopy(ctx.get('effects', [])))
        def measure(name, function, *args):
            if len(measurements) >= 32:
                raise ProgramError('Replay measurement budget exhausted')
            if name=='measure_patch':
                if len(args)!=3:raise ValueError('measure_patch requires camera, bbox and reference_step')
                reference=args[2]
                if type(reference) is not int or reference<0 or reference>step:
                    raise ValueError('Reference must be an earlier real observation')
                if reference not in loaded_steps:
                    if len(loaded_steps)>=16:raise ProgramError('Replay retained-image budget exhausted')
                    archive.update(read_observations(observation_path,{reference}))
                    loaded_steps.add(reference)
            result = function(archive, step, *args)
            measurements.append(dict(operation=name, arguments=deepcopy(args), result=result))
            return result
        from .execution_policy import TOOL_GEOMETRY
        capabilities = dict(
            segment_rgb=lambda camera,text: measure('segment_rgb',segment_rgb,camera,text),
            measure_rgb=lambda views,z,pose=None: measure('measure_rgb',measure_rgb,views,z,pose,TOOL_GEOMETRY),
            locate_rgb_point=lambda *a: measure('locate_rgb_point',locate_rgb_point,*a),
            measure_region=lambda *a: measure('measure_region', measure_region, *a),
            measure_patch=lambda *a: measure('measure_patch', measure_patch, *a),
            recall_experience=lambda query, ids, limit, filters=None: lookup(past, query, ids, limit, filters=filters))
        observed, decisions = None, []
        try:
            def effects_for(updates):
                ids={ctx['id']}
                ids.update(f"e{ctx['episode']}:s{s}" for s in observed_steps if s<=step)
                for transition in past['transitions']:
                    for side in ('before','after'):
                        value=transition.get(side,{})
                        if value.get('episode')==ctx['episode'] and value.get('step',step+1)<=step and value.get('id'):
                            ids.add(value['id'])
                return update_effects(ctx.get('effects',[]),updates,ctx['episode'],ids,'offline_replay')
            def commit_effects(effects):
                ctx['effects']=current_effects(effects,ctx['episode'])
                ctx['effect_counts']=effect_counts(ctx['effects'])
            def observe():
                nonlocal state, observed
                observed = program.observe(ctx, state, capabilities)
                if observed is not None:
                    effects=effects_for(observed.get('effects',[]))
                    if observed.get('state_update') is not None:
                        state = deepcopy(observed['state_update'])
                    commit_effects(effects)
            observe()
            if not full_context:
                status = 'observer_only_legacy_snapshot'
            else:
                seen = set()
                control = ExperienceControl(deepcopy(past), ctx['episode'])
                control.active=deepcopy((ctx.get('experience_control') or {}).get('active'))
                capabilities['control_status']=lambda:control.assess(ctx)
                knowledge=OperationKnowledge({},ctx['episode'])
                view=ctx.get('operation_knowledge') or {}
                knowledge.install(view.get('operations',[]))
                if 'selected' in view:knowledge.bank['selected']=deepcopy(view['selected'])
                knowledge.active=deepcopy(view.get('active'))
                known={ctx['id']}|{t['id'] for t in past['transitions']}
                known.update(t[s]['id'] for t in past['transitions'] for s in ('before','after'))
                known.update(candidate_evidence_ids)
                knowledge.install(prepare_updates(operations or [],known,ctx['episode'],'replay_candidate',
                    memory.get('operation_knowledge',{}).get('versions',[])))

                status = 'observed_decision'
                for _ in range(16):
                    signature = digest([ctx, state])
                    if signature in seen:
                        raise ProgramError('Local continuation cycle on recorded evidence')
                    seen.add(signature)
                    decision = program.run(ctx, state, capabilities)
                    decisions.append(decision)

                    effects=effects_for(decision.get('effects',[]))
                    spec=control.prepare(decision.get('intent'),ctx,has_actions=False,has_program=True)
                    changed=False
                    selection=None
                    operation_changed=False
                    if 'operation' in decision:
                        value=knowledge.prepare_invocation(decision['operation'])
                        changed=knowledge.select(value,ctx,'mechanism',decision.get('reason',''))
                        selection=value
                        operation_changed=changed
                        if changed:
                            control.active=None
                            state=deepcopy(value['parameters']) if value else {}
                            record=knowledge.record()
                            program=Program(record['source']) if record else main_program
                            ctx['operation_knowledge']=knowledge.context()
                    changed=control.activate(spec,ctx,source='mechanism',program=program.sha) or changed
                    if changed:
                        if (ctx.get('candidate_plan') or {}).get('status')=='active':
                            ctx['buffered_actions']=[]
                            view=deepcopy(ctx['candidate_plan'])
                            view.update(status='objective_changed',remaining=0,event=dict(type='plan_objective_changed',
                                plan_id=view.get('id'),discarded_candidates=view.get('remaining',0),
                                reason='Local program selected a new objective; previous candidates are not evidence for it'))
                            for name in ('next_reference','target','proposed_increment'):view.pop(name,None)
                            ctx['candidate_plan']=view
                        ctx['experience_control']=control.context()
                    changed=changed or current_effects(effects,ctx['episode'])!=ctx.get('effects',[])
                    commit_effects(effects)
                    next_state=invocation_state(selection,operation_changed,state,decision.get('state_update'))
                    changed = changed or next_state != state
                    state = next_state
                    if decision['kind'] != 'continue':
                        break
                    if changed:
                        observe()
                else:
                    raise ProgramError('Local continuation budget exhausted')
            results.append(dict(case_id=key, status=status, observer=observed,
                decisions=decisions, state_after=state, measurements=measurements, executed=False))
        except (ProgramError, ValueError, TypeError, KeyError) as error:
            results.append(dict(case_id=key, status='unresolved', error=str(error),
                observer=observed, decisions=decisions, measurements=measurements, executed=False))
    return dict(source_sha256=main_program.sha, cases=results, executed=False, installed=False,
        scope='Historical diagnostic only. No live fitting/IK, action execution, task labels or proof of future autonomy. Missing capabilities are unresolved, not simulated success.')
