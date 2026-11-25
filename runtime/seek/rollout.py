from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from seek.policy.action_feedback import TrackingGuard, tracking_receipt
from seek.policy.bounded_policy import MAX_RETRY_AFTER_S
from seek.policy.camera_readiness import blank_cameras
from seek.policy.dialogue_policy import DialogueConfig, load_config
from seek.policy.eval_failure import PolicyServiceError, policy_call
from seek.policy.head_calibration_capture import capture_head_calibration, capture_camera_reference
from seek.policy import transport_retry
from seek.policy.motion_query import measure_motion,execute_measured_motion
from seek.policy.episode_outcome import finish_policy_episode
from seek.policy.execution_receipt import execution_receipt


def _rpc_timeout_s(config: DialogueConfig) -> float | None:
    if transport_retry.enabled() or config.max_transport_attempts is None:
        return None
    retry_delay = sum(
        max(min(2 ** (attempt - 1), 8), MAX_RETRY_AFTER_S)
        for attempt in range(1, config.max_transport_attempts)
    )
    return (config.max_action_corrections + 1) * (
        config.max_transport_attempts * (config.request_timeout_s + 2) + retry_delay
    ) + 60


def _plain(action: dict) -> dict:
    return {key: np.asarray(value).tolist() for key, value in action.items()}


def _ready_observation(env):
    attempts = []
    for attempt in range(31):
        observation = env.get_obs()
        bad = blank_cameras({0: observation})
        if not bad:
            for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                camera = observation['vision'][name]
                camera.pop('depth', None)
                camera.pop('depth_calibration', None)
                camera['rgb_calibration'] = capture_camera_reference(env, observation, name)
            return observation
        attempts.append({"attempt": attempt, "blank_cameras": bad})
    env._policy_failure = {
        "method": "camera_capture",
        "classification": "camera_capture_failure",
        "official_score": None,
        "attempts": attempts,
    }
    raise PolicyServiceError("Camera capture remained blank after 30 render-only retries")


def eval_one_episode(env, model_client) -> None:
    config = load_config(os.environ.get("ASTRA_HARNESS_CONFIG"))

    env.ik_continuity_guard = None
    if config.ik_continuity_guard:
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("IK continuity guard requires one environment per model session")
        from seek.policy.observed_ik_continuity import IKContinuityError, IKContinuityGuard

        env.ik_continuity_guard = IKContinuityGuard(
            config.max_joint_increment_rad,
            log_path=Path(env.save_dir) / "_ik_continuity.jsonl",
        )
    if hasattr(model_client, "_client"):
        model_client._client.config.request_timeout_s = _rpc_timeout_s(config)
    policy_call(env, model_client, "reset")
    step = 0
    turns = 0
    observation = _ready_observation(env)
    observation["model_meta"] = {
        "step": step,
        "max_episode_steps": int(env.step_lim),
        "motion_query_available": env.ik_continuity_guard is not None,
        "tracking_guard_enabled": os.environ.get('ASTRA_TRACKING_GUARD','1') == '1',
    }
    if config.include_head_calibration:
        observation["model_meta"]["head_calibration_context"] = capture_head_calibration(
            env, observation
        )
    policy_call(env, model_client, "update_obs", obs=observation)

    motion_solutions = {}
    while not env.is_episode_end():
        if config.max_turns is not None and turns >= config.max_turns:
            raise PolicyServiceError("Harness action dialogue reached max_turns")
        actions = policy_call(env, model_client, "get_action")

        while isinstance(actions,dict) and actions.get('kind')=='motion_query':
            if actions.get('step') != step or env.ik_continuity_guard is None:
                raise ValueError('Motion query does not match the current execution state')
            report, solved = measure_motion(env, observation, actions)
            motion_solutions[actions['id']] = (step, solved)
            policy_call(env, model_client, 'notify_motion_result', obs=report)
            actions = policy_call(env, model_client, 'get_action')
        solution = None
        reference = None
        if isinstance(actions, dict):
            if actions.get('kind') == 'episode_outcome':
                policy_call(env, model_client, 'finalize')
                finish_policy_episode(env, actions)
                return
            if actions.get('kind') not in ('planned_action','probed_action'):
                raise ValueError('Unknown policy response kind')
            if actions['kind']=='planned_action':reference=deepcopy(actions['reference'])
            if 'query_id' in actions:
                proof_step, solution = motion_solutions[actions['query_id']]
                if proof_step != step or not solution:
                    raise ValueError('A dispatch needs a feasible measurement from this physical state')
            actions = [actions['action']]
        turns += 1

        tracking = TrackingGuard() if reference is None and os.environ.get("ASTRA_TRACKING_GUARD", "1") == "1" else None
        for action_index, action in enumerate(actions):
            if env.is_episode_end():
                break
            before = deepcopy(observation['state'])
            if env.ik_continuity_guard is None:
                env.take_action(action)
            else:
                try:
                    if solution is None:
                        env.take_action(action)
                    else:
                        execute_measured_motion(env, action, solution)
                except IKContinuityError as exc:
                    event = dict(exc.event, rejected_command=_plain(action))
                    if reference is None:
                        event.update(action_index=action_index,discarded_commands=len(actions)-action_index)
                    policy_call(
                        env,
                        model_client,
                        "notify_action_rejection",
                        obs={"step": step, "event": event, "reference": reference},
                    )
                    break
            step += 1
            motion_solutions.clear()
            observation = _ready_observation(env)
            observation["model_meta"] = {
                "step": step,
                "max_episode_steps": int(env.step_lim),
                "executed_action": _plain(action),
                "motion_query_available": env.ik_continuity_guard is not None,
                "tracking_guard_enabled": os.environ.get('ASTRA_TRACKING_GUARD','1') == '1',
                "commanded_grippers": {arm:float(action[arm+'_ee_joint_state'][0]) for arm in ('left','right')},
            }
            checks = getattr(env, "_last_ik_execution_checks", {}).get(0)
            if checks is not None:
                observation["model_meta"]["ik_execution_checks"] = deepcopy(checks)
            interrupted = tracking is not None and tracking.observe(
                tracking_receipt(action, observation["state"])
            )
            if interrupted:
                observation["model_meta"]["controller_event"] = {
                    "type": "chunk_interrupted_tracking_error",
                    "discarded_commands": len(actions) - action_index - 1,
                    "reason": "Two consecutive large arm target residuals; reobserve and replan.",
                }
            if reference is not None:
                observation['model_meta']['execution_receipt']=execution_receipt(
                    reference,step,_plain(action),before,observation['state'],checks,
                    observation['model_meta'].get('controller_event'))
            feedback = policy_call(env, model_client, "update_obs", obs=observation)
            if interrupted or (isinstance(feedback, dict) and feedback.get("stop_chunk") is True):
                break
    policy_call(env, model_client, "finalize")
