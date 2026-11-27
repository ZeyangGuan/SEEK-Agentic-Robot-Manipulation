from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from seek.policy import transport_retry
from seek.policy.camera_readiness import CAMERAS, RENDER_RETRIES, blank_cameras, retry_until_visible
from seek.policy.dialogue_policy import load_config
from seek.policy.episode_outcome import finish_policy_episode
from seek.policy.eval_failure import PolicyServiceError, policy_call
from seek.policy.execution_receipt import execution_receipt
from seek.policy.head_calibration_capture import capture_camera_reference, capture_head_calibration
from seek.policy.motion_query import execute_measured_motion, measure_motion
from seek.policy.provider_response import MAX_RETRY_AFTER_S


ARMS = ('left', 'right')
RPC_MARGIN_S = 60


def rpc_timeout(config):
    attempts = config.max_transport_attempts
    if transport_retry.enabled() or attempts is None:
        return None
    waits = sum(max(min(2 ** retry, 8), MAX_RETRY_AFTER_S) for retry in range(attempts - 1))
    per_action = attempts * (config.request_timeout_s + 2) + waits
    return (config.max_action_corrections + 1) * per_action + RPC_MARGIN_S


def plain_action(action):
    return {key: np.asarray(value).tolist() for key, value in action.items()}


def _ready_observation(env):
    def render():
        observation = env.get_obs()
        return observation, blank_cameras({0: observation})

    observation, attempts = retry_until_visible(render)
    if observation is None:
        env._policy_failure = dict(method="camera_capture", classification="camera_capture_failure",
                                   official_score=None, attempts=attempts)
        raise PolicyServiceError(f"Cameras stayed blank through {RENDER_RETRIES} extra renders")
    for name in CAMERAS:
        camera = observation['vision'][name]
        camera.pop('depth', None)
        camera.pop('depth_calibration', None)
        camera['rgb_calibration'] = capture_camera_reference(env, observation, name)
    return observation


def _install_guard(env, config):
    env.ik_continuity_guard = None
    if not config.ik_continuity_guard:
        return ()
    if getattr(env, "num_envs", 1) != 1:
        raise ValueError("The IK continuity guard needs a single environment")
    from seek.policy.ik_continuity import IKContinuityError, IKContinuityGuard

    env.ik_continuity_guard = IKContinuityGuard(
        config.max_joint_increment_rad, log_path=Path(env.save_dir) / "_ik_continuity.jsonl")
    return IKContinuityError


class Episode:
    def __init__(self, env, client):
        self.env, self.client = env, client
        self.config = load_config(os.environ.get("ASTRA_HARNESS_CONFIG"))
        self.rejection = _install_guard(env, self.config)
        self.step = self.turns = 0
        self.solutions = {}
        self.observation = None

    def call(self, method, **arguments):
        return policy_call(self.env, self.client, method, **arguments)

    def _flags(self):
        return dict(motion_query_available=self.env.ik_continuity_guard is not None,
                    tracking_guard_enabled=os.environ.get('ASTRA_TRACKING_GUARD', '1') == '1')

    def _begin(self):
        if hasattr(self.client, "_client"):
            self.client._client.config.request_timeout_s = rpc_timeout(self.config)
        self.call("reset")
        self.observation = observation = _ready_observation(self.env)
        observation["model_meta"] = meta = dict(step=self.step, max_episode_steps=int(self.env.step_lim),
                                                **self._flags())
        if self.config.include_head_calibration:
            meta["head_calibration_context"] = capture_head_calibration(self.env, observation)
        self.call("update_obs", obs=observation)

    def _next_packet(self):
        packet = self.call("get_action")
        while isinstance(packet, dict) and packet.get('kind') == 'motion_query':
            if packet.get('step') != self.step or self.env.ik_continuity_guard is None:
                raise ValueError('The motion query does not belong to the current step')
            report, solved = measure_motion(self.env, self.observation, packet)
            self.solutions[packet['id']] = (self.step, solved)
            self.call('notify_motion_result', obs=report)
            packet = self.call('get_action')
        return packet

    def _measured_solution(self, packet):
        if 'query_id' not in packet:
            return None
        measured_at, solution = self.solutions[packet['query_id']]
        if measured_at != self.step or not solution:
            raise ValueError('The action cites a measurement that is not feasible at the current step')
        return solution

    def _execute(self, packet):
        reference, solution = deepcopy(packet['reference']), self._measured_solution(packet)
        action = packet['action']
        self.turns += 1
        if self.env.is_episode_end():
            return
        before = deepcopy(self.observation['state'])
        try:
            if solution is None:
                self.env.take_action(action)
            else:
                execute_measured_motion(self.env, action, solution)
        except self.rejection as error:
            event = dict(error.event, rejected_command=plain_action(action))
            self.call("notify_action_rejection", obs=dict(step=self.step, event=event, reference=reference))
            return
        self.step += 1
        self.solutions.clear()
        self.observation = observation = _ready_observation(self.env)
        observation["model_meta"] = meta = dict(
            step=self.step, max_episode_steps=int(self.env.step_lim), executed_action=plain_action(action),
            **self._flags(),
            commanded_grippers={arm: float(action[f'{arm}_ee_joint_state'][0]) for arm in ARMS})
        checks = getattr(self.env, "_last_ik_execution_checks", {}).get(0)
        if checks is not None:
            meta["ik_execution_checks"] = deepcopy(checks)
        meta['execution_receipt'] = execution_receipt(
            reference, self.step, plain_action(action), before, observation['state'], checks, None)
        self.call("update_obs", obs=observation)

    def run(self):
        self._begin()
        while not self.env.is_episode_end():
            if self.config.max_turns is not None and self.turns >= self.config.max_turns:
                raise PolicyServiceError("The episode reached max_turns")
            packet = self._next_packet()
            kind = packet.get('kind') if isinstance(packet, dict) else None
            if kind == 'episode_outcome':
                self.call('finalize')
                finish_policy_episode(self.env, packet)
                return
            if kind != 'planned_action':
                raise ValueError(f'Unknown policy response kind: {kind}')
            self._execute(packet)
        self.call("finalize")


def run_episode(env, client):
    Episode(env, client).run()
