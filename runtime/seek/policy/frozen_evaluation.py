from copy import deepcopy
from .mechanism_program import ProgramError

GUIDANCE = '''
FROZEN EXPERIENCE EVALUATION
The incoming reviewed experience is fixed for this episode. Read and execute its
existing programs and named operations. Do not author, replace, or revise code:
act.program must be null and act.operations must be an empty array. Existing
operation invocations, current-scene parameters, visual bindings, control intents,
direct actions, live observations and effect assessments remain available.
Current-episode receipts support feedback and objective assessment, but do not
change the incoming programs, named operations, or reusable objective records. There is no post-episode learning review. Current execution logs are kept
separately; the reusable output memory is byte-for-byte the incoming memory.
This evaluation constraint takes precedence over general guidance to author code.
'''

EPISODE_ADAPTATION_GUIDANCE = '''
WITHIN-EPISODE PROGRAM ADAPTATION — NO CROSS-EPISODE CARRYOVER
Each episode starts from the same incoming reviewed memory. You may author,
replace and revise the executable task program and named operations through
act.program and act.operations during this episode. Existing code, observations,
actual execution receipts and current-scene parameters remain available.
When current evidence shows that an existing method is unsuitable, you may revise
its executable logic rather than only invoke it with different parameters.
All normal program validation, evidence provenance and execution checks still apply.
The foundation model weights, simulator physics, task assets, native score and
step budget are unchanged. This permission concerns task programs and operations.
Current receipts provide feedback and evidence for local code revisions.
Actions are supplied by executable programs or checked model candidates.
There is no post-episode learning review. Local revisions and execution evidence
are saved only in the episode audit trace. The reusable output memory is restored
byte-for-byte from the input. The next episode receives none of these changes.
This episode-local persistence rule overrides general claims that code persists
across episodes. It does not require a revision when existing code is suitable.
'''


def knowledge(memory):
    operations = memory.get('operation_knowledge', {})
    return deepcopy(dict(
        **{key: memory.get(key) for key in (
            'programs', 'program_abis', 'active', 'episodes',
            'improvement_history')},
        operations={key: operations.get(key) for key in ('versions', 'selected', 'entry')},
        methods=memory.get('control_experience', {}).get('methods', {})))


def validate_proposal(proposal):
    if proposal.get('program') is not None or proposal.get('operations'):
        raise ProgramError('Frozen evaluation: program must be null and operations must be empty; invoke existing operations or use direct actions')


def persist(model, atomic_json):
    current, incoming = knowledge(model.seek_bank), knowledge(model.seek_base)
    if getattr(model, 'seek_episode_adaptation', False):
        for key in ('episodes', 'improvement_history', 'methods'):
            if current[key] != incoming[key]:
                raise RuntimeError('Episode adaptation changed fixed knowledge: '+key)
    elif current != incoming:
        raise RuntimeError('Frozen reusable knowledge changed')
    atomic_json(model.seek_output.with_name('evaluation_trace_memory.json'), model.seek_bank)
    model.seek_output.write_bytes(model.seek_input.read_bytes())
    if getattr(model, 'seek_episode_adaptation', False):
        prior_programs = {p['id'] for p in incoming['programs']}
        prior_operations = {p['id'] for p in incoming['operations']['versions'] or []}
        atomic_json(model.seek_output.with_name('episode_adaptation_audit.json'), dict(
            episode=model.seek_episode, step=model.step, carryover='none',
            input_memory_sha256=model.seek_input_sha,
            new_program_ids=[p['id'] for p in current['programs'] if p['id'] not in prior_programs],
            new_operation_ids=[p['id'] for p in current['operations']['versions'] or [] if p['id'] not in prior_operations],
            revised_operation_names=[name for name, version in (current['operations']['selected'] or {}).items()
                if (incoming['operations']['selected'] or {}).get(name) != version],
            output_memory_restored_from_input=True))
