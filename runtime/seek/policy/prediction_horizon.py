GUIDANCE = '''
Choose the number of action rows separately on EVERY act response, within
context.prediction_budget.max_actions_this_call. This is a ceiling, not a fixed
length or a minimum of five. Old program reason text mentioning five rows does not
override this budget. Choose the next meaningful attempt and the observation that
can change its next decision. For a known method, implement supported continuation
in observe/decide. For exploration, a bounded teacher batch may end at a new visual
interpretation without first generalizing the maneuver. An uncertain contact can
need per-step measurement without needing per-step model interpretation. Use teacher rows for a path that
the selected program will consume; use actions=[] for a locally generated method.
Choose longer or shorter candidates from evidence, without padding or step quotas.
Zero rows may update intent, mechanism, binding or current evidence. Proposed
actions never establish their effects or permission to relax physical limits.
The local mechanism reads a fresh observation and receipt before EVERY physical
step. It can consume the next buffered candidate, replace it with a local action,
or request the model. A rejected row discards the unexecuted suffix; a newly accepted
proposal replaces the old buffer. Only executed receipts become experience.
'''


def prediction_horizon(step_limit):
    if type(step_limit) is not int or step_limit <= 0:
        raise ValueError('A positive native step limit is required')
    for limit, horizon in ((400, 5), (700, 10), (1100, 15), (1600, 20)):
        if step_limit <= limit:
            return horizon
    return 25


def prediction_budget(horizon, step, step_limit):
    remaining = None if step_limit is None else max(0, step_limit - step)
    return dict(episode_max=horizon, remaining_steps=remaining,
                max_actions_this_call=horizon if remaining is None else min(horizon, remaining),
                length_selected_by='model_from_current_evidence', physical_feedback_every=1)


def prediction_contract(horizon):
    return (f'Predict 0..{horizon} steps per act (zero only for an intent, mechanism, binding or current-evidence update); '
            'dispatch one physical step at a time with feedback and interruption; local experience control may continue beyond the candidate batch. ')
