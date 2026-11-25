import json
from pathlib import Path


def finish_policy_episode(env, outcome):
    if outcome.get('kind') != 'episode_outcome' or outcome.get('termination') != 'blocked':
        raise ValueError('Expected a policy blocked outcome')

    if env.num_envs != 1 or getattr(env, 'unstable_envs', None):
        raise RuntimeError('Cannot score a policy outcome in an unhealthy environment')
    rewards = env.reward_manager.get_reward(final_check=True)
    env.success[0] = bool(rewards[0] > 1 - 1e-3)
    env.end_flag[0] = True
    env.get_obs_batch(env_idx_list=[0], last_frame=True)
    record = dict(outcome, executed_steps=int(env.take_action_cnt[0]),
                  score_kind='native_current_state_at_policy_terminal')
    env.eval_result['policy_outcome'] = record
    (Path(env.save_dir)/'_policy_outcome.json').write_text(json.dumps(record, ensure_ascii=False)+'\n')
    return record
