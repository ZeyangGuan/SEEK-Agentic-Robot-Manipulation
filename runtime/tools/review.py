from copy import deepcopy
import json,os
from pathlib import Path
from seek.policy.mechanism_program import ProgramError,digest,validate_candidate
from seek.policy.execution_policy import ABI,FEEDBACK_ABI,proposal_schema
from seek.policy.improvement_memory import (append_review,improvement_context,
    lesson_context,lesson_schema,audit_lessons,LESSON_GUIDANCE,GUIDANCE as IMPROVEMENT_GUIDANCE)
from seek.policy.review_learning import (GUIDANCE as LEARNING_GUIDANCE, handoff_evidence,
    learning_schema, audit_learning_update, previous_review_check)
from seek.policy.context_view import GUIDANCE as CONTEXT_GUIDANCE,review_view
from seek.policy.provider_compat import to_wire_payload
from tools.review_transport import send_review,save
from seek.policy.plan_memory import PROGRAM_ABI,accept_program,require_compatible
from seek.policy.experience_control import GUIDANCE as EXPERIENCE_GUIDANCE
from seek.policy.experience_access import (GUIDANCE as ACCESS_GUIDANCE,LOOKUP_LIMIT,CONTEXT_LOOKUP_LIMIT,
    recall_tool,lookup,validate_lookup_arguments,is_context_lookup,available_ids,error_records,attach_teacher_evidence)
from seek.policy.candidate_conditions import GUIDANCE as CONDITION_GUIDANCE
from seek.policy.dialogue_policy import canonical_tool_history
from seek.policy.robot_prompt import ARM_WORKSPACE_GUIDANCE, ARM_POSTURE_GUIDANCE, GRIPPER_GUIDANCE
from seek.policy.boundary_learning import (GUIDANCE as BOUNDARY_GUIDANCE, boundary_evidence,
    boundary_index, selected_boundary_view, load_contexts, selected_steps, read_observations, image_parts)
from seek.policy.review_replay import replay_tool, replay
from seek.policy.review_observations import (observation_tool, inspect_observations, lineage_archives,
    GUIDANCE as OBSERVATION_GUIDANCE)
from seek.policy.operation_knowledge import (OperationKnowledge, prepare_updates, updates_schema,
    GUIDANCE as OPERATION_GUIDANCE)
from seek.policy.context_window import ContextArchive,bound_payload
from seek.policy.manipulation_contract import REVIEW_ROLE, assemble_instructions


def review_prompt():
    return assemble_instructions(REVIEW_ROLE, [
        ('Review response and validation interface',
         'Review this actual SEEK episode using review_mechanism. official_task_prompt is the complete '
         'specification supplied to the actor; initial_context.instruction may be abbreviated. '
         'If the full specification is unavailable, retain that uncertainty. '
         'replace includes the complete program; keep/revert have program=null. '
         'Review may update selected operations without replacing unrelated methods. '
         'A new episode has empty state/roles: derive current parameters from its observations, '
         'or ask for the current-scene interpretation that local code cannot provide. '
         'The saved initial context is historical evidence, not the next layout. '
         'Compilation/capability validation does not establish physical success. '
         'An invalid replacement preserves the previous mechanism and raw experience. '
         'A blocked episode is a terminal policy outcome, not an infrastructure fault. '
         'Correct interface errors using the exact validation feedback.'),
        ('Robot planning strategies and gripper semantics',
         ARM_WORKSPACE_GUIDANCE+' '+ARM_POSTURE_GUIDANCE+' '+GRIPPER_GUIDANCE),
        ('Executable program interface', ABI),
        ('Feedback and measurement reference', FEEDBACK_ABI),
        ('Reusable operations and evidence memory',
         OPERATION_GUIDANCE+IMPROVEMENT_GUIDANCE+LESSON_GUIDANCE+EXPERIENCE_GUIDANCE),
        ('Review evidence and next-episode checks', LEARNING_GUIDANCE+BOUNDARY_GUIDANCE+OBSERVATION_GUIDANCE),
        ('Candidate batches', CONDITION_GUIDANCE),
        ('Evidence access and request layout', ACCESS_GUIDANCE+CONTEXT_GUIDANCE),
    ])

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

def actor_task_prompt(trace):
    path=trace/'requests.jsonl'
    if not path.exists():return None
    with path.open() as source:
        for line in source:
            request=json.loads(line)
            for message in request.get('payload',{}).get('input',[]):
                for part in message.get('content',[]):
                    if part.get('type')!='input_text':continue
                    try:packet=json.loads(part['text'])
                    except ValueError:continue
                    if isinstance(packet,dict) and 'official_task_prompt' in packet:
                        return packet['official_task_prompt']
    return None

def validate_review(memory,proposal,initial_context=None,evidence_memory=None,learning_evidence=None):
    if set(proposal)-{'learning_update','operations','entry_operation','lessons','operation_versions'}!={'decision','reason','program','revert_id'}:raise ProgramError('Review fields do not match interface')
    learning=(dict(learning_audit=audit_learning_update(proposal,memory,learning_evidence,evidence_memory))
              if learning_evidence is not None else {})
    decision=proposal['decision'];candidate=deepcopy(memory)
    if evidence_memory is not None and 'learning_boundaries' in evidence_memory:
        candidate['learning_boundaries']=deepcopy(evidence_memory['learning_boundaries'])
    if not isinstance(proposal['reason'],str) or not proposal['reason'].strip():raise ProgramError('Review reason required')
    known={t['id'] for t in memory['transitions']}|{t[s]['id'] for t in memory['transitions'] for s in ('before','after')}
    known.update(available_ids(memory))
    if evidence_memory is not None:known.update(available_ids(evidence_memory))
    operations=OperationKnowledge(candidate,memory['episodes'][-1]['episode'] if memory['episodes'] else 1)
    records=prepare_updates(proposal.get('operations',[]),known,operations.episode,'review',
        operations.bank['versions'])
    operations.install(records)
    operations.restore(proposal.get('operation_versions',{}))
    if 'entry_operation' in proposal:operations.set_entry(proposal['entry_operation'])
    learning.update(operation_updates=[r['id'] for r in records],
                    lesson_audit=audit_lessons(proposal.get('lessons',[]),
                        evidence_memory if evidence_memory is not None else memory))
    if decision=='keep':
        if proposal['program'] is not None or proposal['revert_id'] is not None:raise ProgramError('keep does not install a program')
        status=('operation_update' if records or 'entry_operation' in proposal or proposal.get('operation_versions')
                else 'knowledge_update' if learning['lesson_audit']['accepted'] else 'no_change')
        return candidate,dict(accepted=True,decision='keep',status=status,**learning)
    if decision=='revert':
        if proposal['program'] is not None or proposal['revert_id'] not in {p['id'] for p in memory['programs']}:raise ProgramError('revert requires a real stored program ID')
        require_compatible(memory,proposal['revert_id'])
        candidate['active']=proposal['revert_id']
        return candidate,dict(accepted=True,decision='revert',status='provisional',program=candidate['active'],**learning)
    if decision!='replace' or proposal['revert_id'] is not None:raise ProgramError('Use keep, replace or revert')
    p=proposal['program']
    if not isinstance(p,dict) or set(p)!={'source','description','evidence_ids'}:raise ProgramError('Replacement requires full program')
    known={t['id'] for t in memory['transitions']}|{t[s]['id'] for t in memory['transitions'] for s in ('before','after')}
    known.update(r['id'] for r in memory.get('motion_evidence',[]))
    known.update(r['observation_id'] for r in memory.get('motion_evidence',[]) if 'observation_id' in r)
    known.update(available_ids(memory))
    if evidence_memory is not None:known.update(available_ids(evidence_memory))
    if not set(p['evidence_ids'])<=known:raise ProgramError('Unknown source evidence ID')

    check=validate_candidate(p['source'],[])
    check['runtime_validation']='Each actual decide call; no hypothetical state replay used as an acceptance gate'
    check['initial_context_available']=initial_context is not None
    key=check['sha256']
    accept_program(candidate,dict(id=key,**p,abi=PROGRAM_ABI,origin_episode=memory['episodes'][-1]['episode'],origin_step='review',status='provisional',validation=check))
    return candidate,dict(accepted=True,decision='replace',status='provisional',program=key,validation=check,**learning)

def review(root,report):
    episode=report['episode'];path=root/f'memory_{episode}.json';memory=json.loads(path.read_text())
    directory=root/f'review_{episode}';directory.mkdir(exist_ok=False);save(directory/'input_memory.json',memory)
    trace=Path(report['trace']);transitions=[t for t in memory['transitions'] if t['episode']==episode]
    archives=memory.setdefault('observation_archives',{})
    archives.update(lineage_archives(path));archives[str(episode)]=str(trace/'observations.jsonl')
    teachers=rows(trace/'seek_teacher.jsonl')
    transitions=attach_teacher_evidence(transitions,teachers,episode)
    errors=error_records(rows(trace/'seek_errors.jsonl'),episode)
    lookup_memory=deepcopy(memory)
    teacher_rows={t['id']:t for t in lookup_memory.get('teacher_updates',[])}
    for item in teachers:
        key=f'e{episode}:teacher{item["turn"]}'
        teacher_rows.setdefault(key,dict(item,id=key,episode=episode))
    lookup_memory['teacher_updates']=list(teacher_rows.values())
    error_rows={e['id']:e for e in lookup_memory.get('errors',[])}
    error_rows.update({e['id']:e for e in errors})
    lookup_memory['errors']=list(error_rows.values())

    joined={t['id']:t for t in transitions}
    lookup_memory['transitions']=[joined.get(t['id'],t) for t in memory['transitions']]
    indices=sorted(set([*range(0,len(transitions),max(1,len(transitions)//30)),*range(max(0,len(transitions)-8),len(transitions))]))
    examples=[transitions[i] for i in indices]
    initial_path=trace/'seek_initial_context.json'
    initial_context=json.loads(initial_path.read_text()) if initial_path.exists() else None
    active=next((p for p in memory['programs'] if p['id']==memory.get('active')),None)
    handoffs=handoff_evidence(episode,rows(trace/'seek_decisions.jsonl'),teachers,transitions,report)

    accepted=[teacher_rows.get(f'e{episode}:teacher{t["turn"]}',t) for t in teachers]
    boundaries=boundary_evidence(episode,rows(trace/'seek_decisions.jsonl'),accepted,transitions,load_contexts(trace),
        limit=len(accepted),invocations=memory.get('control_experience',{}).get('invocations',[]))
    save(directory/'boundary_contexts.json',boundaries)

    boundary_records={r['id']:r for r in lookup_memory.get('learning_boundaries',[])}
    boundary_records.update((r['id'],r) for r in boundary_index(boundaries))
    lookup_memory['learning_boundaries']=list(boundary_records.values())
    displayed_boundaries=selected_boundary_view(boundaries)
    images,visual_coverage=image_parts(displayed_boundaries,read_observations(trace/'observations.jsonl',selected_steps(displayed_boundaries)))
    packet=dict(native_result=report,active_program=active,previous_results=memory['episodes'],initial_context=initial_context,
        visual_observation_episodes=sorted(int(e) for e in archives),
        official_task_prompt=actor_task_prompt(trace),
        task_lessons=lesson_context(memory),
        operation_knowledge=OperationKnowledge(deepcopy(memory),episode).context(),
        operation_versions=[{k:r[k] for k in ('id','name','description','evidence_ids')}
                            for r in memory.get('operation_knowledge',{}).get('versions',[])],
        task_effect_assessments=[e for e in memory.get('effects',[]) if e['episode']==episode],
        motion_evidence=[e for e in memory.get('motion_evidence',[]) if e['episode']==episode],
        policy_outcome=report.get('policy_outcome'),
        improvement_history=improvement_context(memory,include_sources=True),
        control_experience=dict(methods=[dict(id=m['id'],definition=m['definition'],
            samples=len(m['evidence_ids']),evidence_ids=m['evidence_ids'][-4:],source_episodes=m['source_episodes'],status=m['status'])
            for m in list(memory.get('control_experience',{}).get('methods',{}).values())[-8:]],
            invocations=[dict({k:v for k,v in item.items() if k not in ('evidence_ids','assessments')},
                samples=len(item['evidence_ids']),recent_assessments=item['assessments'][-2:])
                for item in memory.get('control_experience',{}).get('invocations',[]) if item['episode']==episode][-20:],
            full_evidence_preserved=True),
        migration=memory.get('migration'),program_abis=memory.get('program_abis',{}),
        actual_transitions=examples,errors=errors,
        program_versions=[{k:p.get(k) for k in ('id','description','origin_episode','origin_step','status')} for p in memory['programs']],
        handoff_evidence=handoffs,previous_review_check=previous_review_check(memory,handoffs),
        boundary_evidence=displayed_boundaries,visual_evidence_coverage=visual_coverage)
    save(directory/'evidence.json',packet)
    model_packet=review_view(packet,transitions)
    save(directory/'model_evidence.json',model_packet)
    properties=dict(decision=dict(type='string',enum=['keep','replace','revert']),reason=dict(type='string'),
                    program=proposal_schema()['properties']['program'],revert_id=dict(type=['string','null']),
                    learning_update=learning_schema(),operations=updates_schema(),
                    entry_operation=dict(type=['string','null']),lessons=lesson_schema(),
                    operation_versions=dict(type='object',additionalProperties=dict(type='string')))
    tool=dict(type='function',name='review_mechanism',strict=False,parameters=dict(type='object',properties=properties,required=[k for k in properties if k not in ('learning_update','operations','entry_operation','lessons','operation_versions')],additionalProperties=False))
    prompt=review_prompt()
    feedback=[];attempts=[];outcome=None;reviews=0;lookups=0;context_lookups=0;probes=0;visual_reads=0;lookup_history=[];call_ids=set()
    profile_path=root/'profile.json'
    profile=json.loads(profile_path.read_text()) if profile_path.exists() else {}
    effort=profile.get('harness_overrides',{}).get('reasoning_effort','xhigh')
    if effort not in ('low','medium','high','xhigh'):raise ProgramError('Unsupported review reasoning effort')
    os.environ['SEEK_REVIEW_BUDGET_DIR']=str(trace)
    try:
        while reviews<3:
            folder=directory/f'attempt_{len(attempts)+1}';folder.mkdir();receipt={}
            payload=dict(model=profile.get('model','gpt-6-astra'),instructions=prompt,reasoning=dict(effort=effort),store=False,stream=True,
                input=[dict(role='user',content=[dict(type='input_text',text=json.dumps(dict(evidence=model_packet,validation_feedback=feedback),separators=(',',':')))]+deepcopy(images))]+deepcopy(lookup_history),
                tools=[tool,recall_tool(),replay_tool(),observation_tool()],parallel_tool_calls=False,
                tool_choice='required' if lookups<LOOKUP_LIMIT or context_lookups<CONTEXT_LOOKUP_LIMIT else dict(type='function',name='review_mechanism'))
            payload,context_metrics=bound_payload(payload,ContextArchive(directory/'context_index'))
            save(folder/'context_budget.json',context_metrics)
            payload=to_wire_payload(payload,'responses')
            save(folder/'request.json',payload)
            token_env=os.environ.get('SEEK_API_TOKEN_ENV') or 'SEEK_MODEL_KEY'
            token=os.environ.get(token_env) or os.environ.get('ASTRA_API_KEY')
            if not token:raise RuntimeError('Set SEEK_MODEL_KEY before starting the review')
            endpoint=profile.get('endpoint',os.environ.get('SEEK_MODEL_ENDPOINT') or os.environ.get('ASTRA_ENDPOINT','https://api.openai.com/v1/responses'))
            response=send_review(endpoint,token,payload,receipt,folder)
            save(folder/'response.json',response);proposal=None;review_counted=False
            try:
                calls=[c for c in response['output'] if c.get('type')=='function_call']
                if len(calls)==1 and calls[0].get('name') in ('recall_experience','test_mechanism','inspect_observations'):
                    call=calls[0]
                    call_id=call.get('call_id')
                    if not isinstance(call_id,str) or not call_id or call_id in call_ids:
                        raise ProgramError('Experience lookup requires a new call_id')
                    call_ids.add(call_id)
                    try:query=json.loads(call['arguments'])
                    except (ValueError,TypeError):query=None
                    context_read=call['name']=='recall_experience' and is_context_lookup(query)
                    if context_read:
                        if context_lookups>=CONTEXT_LOOKUP_LIMIT:raise ProgramError('Deferred-context read allowance exhausted; use returned evidence')
                        context_lookups+=1
                    else:
                        if lookups>=LOOKUP_LIMIT:raise ProgramError('Experience/probe allowance exhausted; deferred-context reads have a separate allowance')
                        lookups+=1
                    visual_images=[]
                    try:
                        if call['name']=='recall_experience':
                            query=validate_lookup_arguments(query)
                            if 'context_ref' in query or 'context_reads' in query:
                                found=ContextArchive(directory/'context_index').read_request(**query)
                            else:
                                found=lookup(lookup_memory,errors=errors,**query)
                        elif call['name']=='inspect_observations':
                            visual_reads+=1
                            found,visual_images=inspect_observations(query,episode,trace/'observations.jsonl',archives=archives)
                        else:
                            probes+=1
                            if not isinstance(query,dict) or set(query)-{'operations'}!={'source','case_ids'}:
                                raise ValueError('test_mechanism needs source and case_ids')
                            found=replay(query['source'],query['case_ids'],boundaries,lookup_memory,
                                trace/'observations.jsonl',initial_context,operations=query.get('operations'))
                    except (ValueError,TypeError,KeyError,IndexError,FileNotFoundError) as error:found=dict(error=str(error),executed=False)
                    lookup_history.extend(canonical_tool_history([call]))
                    lookup_history.append(dict(type='function_call_output',call_id=call_id,output=json.dumps(found,ensure_ascii=False)))
                    if visual_images:lookup_history.append(dict(role='user',content=visual_images))
                    audit=dict(decision='read_only_context' if context_read else 'read_only_experience' if call['name']=='recall_experience' else 'read_only_observations' if call['name']=='inspect_observations' else 'read_only_program_probe',executed=False,result=found)
                    save(folder/'audit.json',audit);attempts.append(dict(receipt=receipt,audit=audit));continue
                reviews+=1;review_counted=True
                if len(calls)!=1 or calls[0].get('name')!='review_mechanism':raise ProgramError('Expected one review_mechanism call')
                proposal=json.loads(calls[0]['arguments']);save(folder/'proposal.json',proposal)
                candidate,audit=validate_review(memory,proposal,initial_context=initial_context,evidence_memory=lookup_memory,learning_evidence=handoffs)
                if any(digest(candidate.get(k,[]))!=digest(memory.get(k,[])) for k in
                       ('transitions','episodes','motion_evidence','effects','episode_outcomes','control_experience','teacher_updates','errors')):
                    raise RuntimeError('Review attempted to modify raw evidence')
                append_review(memory,candidate,proposal,audit,packet['errors'],handoffs=handoffs)
                save(path,candidate);outcome=audit
            except (ValueError,TypeError,KeyError) as error:
                if not review_counted:reviews+=1
                audit=dict(accepted=False,decision='repair_required',error=str(error),proposal=proposal)
                feedback.append(audit)
            save(folder/'audit.json',audit);attempts.append(dict(receipt=receipt,audit=audit))
            if outcome:break
        result=dict(attempts=attempts,outcome=outcome or dict(accepted=False,decision='retain_previous',status='unresolved_interface'),
                    requests=sum(a['receipt']['requests'] for a in attempts),experience_lookup_responses=lookups-probes-visual_reads,
                    program_probe_responses=probes,visual_lookup_responses=visual_reads,context_read_responses=context_lookups,read_only_responses=lookups+context_lookups,
                    review_attempts=reviews,raw_memory_preserved=True)
        save(directory/'result.json',result);return result
    finally:os.environ.pop('SEEK_REVIEW_BUDGET_DIR',None)
