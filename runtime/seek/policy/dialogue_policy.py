from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from .action_validation import chunk_commands
from .model_policy import CAMERAS
from .bounded_policy import Model as BoundedModel, BudgetExceeded, RequestTransportError
from .model_request_budget import request_budget
from .provider_compat import provider_cache_usage, to_wire_payload
from .observation_images import camera_input_parts, image_encoding, prompt_image, validate_prompt_image


ARM_WORKSPACE_GUIDANCE = (
    "Arm-selection strategy: prefer the left arm for the robot's left workspace and the right arm for its right workspace. "
    'Keep the arms uncrossed. This does not define a fixed x-coordinate cutoff or a midline exclusion band; '
    'assess the current arm and tool geometry.'
)
ARM_POSTURE_GUIDANCE = (
    'Posture strategy: prefer the selected elbow-up configuration and room to maneuver before full extension. '
    'Adapt the approach or posture using current geometry and motion receipts when that strategy obstructs progress. '
    'Maintain the executor\'s joint and IK continuity limits.'
)
GRIPPER_GUIDANCE = (
    'Grippers use normalized opening targets: 0 fully closed, 1 fully open; keep 0 while holding. '
    'This is position control, not a force command. gripper_aperture is the first measured finger joint normalized to its range; '
    'use both measured gripper_joint_positions for achieved opening and asymmetry. '
    'gripper_target is the executor target, also retained as the legacy ee_joint_state alias. '
    'Neither a target nor an aperture alone confirms grasp success; judge retention from observations.'
)


DEFAULT_REQUEST_TIMEOUT_S = 600


@dataclass(frozen=True)
class DialogueConfig:
    predict_horizon: int = 24
    fixed_predict_horizon: bool = False
    execute_horizon: int = 24
    experience_horizon_from_budget: bool = False
    feedback_frames: int = 4
    max_turns: int | None = None
    max_model_requests: int | None = 150
    action_space: str = 'ee_delta'
    delta_frame: str = 'world'
    max_translation_m: float = .02
    max_rotation_rad: float = .15
    max_joint_increment_rad: float = .15
    ik_continuity_guard: bool = False
    max_output_tokens: int | None = None

    output_token_reservation: int = 16384
    reasoning_effort: str = "xhigh"
    max_total_tokens: int | None = None
    max_seconds: int | None = None
    max_context_text_bytes: int | None = None
    max_context_images: int | None = None
    max_context_wire_bytes: int | None = None
    max_context_reserved_tokens: int | None = None
    cache_mode: str = 'implicit'
    official_task_documentation_version: str = ''
    official_task_documentation_source: str = ''
    official_task_prompt: str = ''
    image_profile: str = 'thumbnail_jpeg'
    image_detail: str = 'low'
    image_layout: str = 'separate'
    include_head_calibration: bool = False
    max_transport_attempts: int | None = 1

    max_action_corrections: int = 0
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S

    def __post_init__(self):
        for field, low, high in (
            ('predict_horizon', 1, 64), ('execute_horizon', 1, 64),
            ('feedback_frames', 1, 8), ('max_turns', 1, 80), ('max_model_requests', 1, 150),
            ('max_output_tokens', 256, 65536), ('output_token_reservation', 256, 65536), ('max_total_tokens', 1, 20000000),
            ('max_seconds', 1, 7200), ('max_context_text_bytes', 1024, 2048000),
            ('max_context_images', 3, 512), ('max_context_wire_bytes', 100000, 32000000),

            ('max_context_reserved_tokens', 1024, 2048000 + 512 * 1024 + 65536),
            ('max_transport_attempts', 1, 3),
            ('max_action_corrections', 0, 3),
            ('request_timeout_s', 1, 900),
        ):
            value = getattr(self, field)
            if field in ('max_transport_attempts', 'max_model_requests', 'max_turns', 'max_total_tokens', 'max_seconds', 'max_context_text_bytes', 'max_context_images', 'max_context_wire_bytes', 'max_context_reserved_tokens', 'max_output_tokens') and value is None:
                continue
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{field} must be an integer in [{low},{high}]')
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning_effort")
        if self.execute_horizon > self.predict_horizon:
            raise ValueError('Cannot execute more steps than predicted')
        if self.action_space not in ('ee_delta', 'joint') or self.delta_frame != 'world':
            raise ValueError('Supported: world ee_delta, or absolute joint. No implicit frame conversion.')
        if self.cache_mode not in ('implicit', 'omit'):
            raise ValueError('cache_mode must be implicit or omit for unsupported gateways')
        image_encoding(self.image_profile, self.image_detail, image_layout=self.image_layout)
        if self.image_layout == 'triptych' and self.feedback_frames != 1:
            raise ValueError('Triptych requires all three cameras and one current observation per feedback')
        for field in ('max_translation_m', 'max_rotation_rad', 'max_joint_increment_rad'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value <= 1:
                raise ValueError(f'Invalid {field}')
        if any(type(value) is not bool for value in (self.experience_horizon_from_budget, self.include_head_calibration, self.fixed_predict_horizon, self.ik_continuity_guard)):
            raise ValueError('Boolean fields require boolean values')
        for field in ('official_task_documentation_version',
                      'official_task_documentation_source', 'official_task_prompt'):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f'{field} must be a string')
        if bool(self.official_task_prompt) != bool(self.official_task_documentation_version):
            raise ValueError('Official task prompt and documentation version must be set together')
        if bool(self.official_task_prompt) != bool(self.official_task_documentation_source):
            raise ValueError('Official task prompt and documentation source must be set together')


def load_config(path=None):
    raw = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    return DialogueConfig(**raw)


def canonical_tool_history(outputs):
    items = deepcopy(outputs)
    for item in items:
        if item.get('type') == 'function_call':
            item.pop('id', None)
            item.pop('status', None)
    return items


def without_reasoning_replay(payload):
    recovered = deepcopy(payload)
    inputs = recovered.get('input')
    if not isinstance(inputs, list):
        return recovered, 0
    kept = [item for item in inputs
            if not isinstance(item, dict) or item.get('type') != 'reasoning']
    removed = len(inputs) - len(kept)
    recovered['input'] = kept
    return recovered, removed


class Model(BoundedModel):
    def __init__(self, cfg):
        self.dialogue_config = load_config(os.environ.get('ASTRA_HARNESS_CONFIG'))
        super().__init__(cfg)
        c = self.dialogue_config
        self.max_chunk = c.predict_horizon
        self.max_output = c.max_output_tokens or c.output_token_reservation
        self.effort = os.environ.get("ASTRA_REASONING_EFFORT", c.reasoning_effort)
        self.max_requests = None if c.max_turns is None or c.max_transport_attempts is None else c.max_turns * c.max_transport_attempts
        self.max_total_tokens = c.max_total_tokens
        self.max_seconds = c.max_seconds
        self.request_timeout_s = c.request_timeout_s
        prediction_contract = (
            f'Predict exactly {c.predict_horizon} steps per act; execute at most {c.execute_horizon}. '
            if c.fixed_predict_horizon else
            f'Predict 1..{c.predict_horizon} steps per act; execute at most {c.execute_horizon}. '
        )
        action_contract = (
            'action_space=ee_delta: each arm6=[dx,dy,dz,rx,ry,rz], world translation in meters '
            'and world rotation-vector increments in radians, not Euler angles. Start from the '
            'latest observed EE pose; within a chunk accumulate from the previous requested '
            'target: p_next=p_previous+dxyz, q_next=exp(rotvec)*q_previous. '
            f'Per step, translation norm <= {c.max_translation_m}m and rotation norm <= {c.max_rotation_rad}rad. '
            if c.action_space == 'ee_delta' else
            'action_space=joint: each arm6 is an absolute joint target in radians. '
            f'Per step, joint change <= {c.max_joint_increment_rad}rad. '
        )
        self.prompt = (
            'Control dual ARX X5 from RGB, task instruction and proprioception using exactly one act call per turn. '
            'Reason internally; execution_note is a brief intent/uncertainty note. '
            'Do not access simulator object poses, rewards, task code or layouts.\n\n'
            'EE means link6, not fingertip/TCP. EE poses use environment-origin positions with '
            'world-aligned axes, meters, and quaternions [w,x,y,z]. '
            + ARM_WORKSPACE_GUIDANCE + ' ' + ARM_POSTURE_GUIDANCE + ' ' +
            GRIPPER_GUIDANCE + '\n\n'
            'Each action row has 14 numbers: [left6,left_gripper,right6,right_gripper]. '
            'Use at most six decimal places. '
            + action_contract + prediction_contract +
            'Commands run at 25Hz. After a tracking-guard interruption, replan from the latest observation. '
            'Tool receipts identify executed steps and discarded suffixes; rejected acts execute zero steps. '
            'Small tracking residuals do not prove grasping or completion. '
            + ('Tool images show the current observation after the executed prefix. '
               if c.feedback_frames == 1 else 'Tool images show sampled, already executed observations. ') +
            'Continue until the environment ends the episode.\n\n'
            'Earlier examples are separate trajectories. Learn progression and action scale, then adapt to current RGB; '
            'do not blindly replay coordinates or assume matching object layouts. '
            'Plans are not evidence of execution or success.'
        )
        if c.ik_continuity_guard:
            self.prompt += (
                f'\n\nBefore each EE action, the controller rejects IK solutions with a joint '
                f'change greater than {c.max_joint_increment_rad} rad from the observed joints '
                'or the preceding accepted IK target. If measured tracking drift makes those two '
                'bounds incompatible, the controller records a reference rebase to observed joints; '
                'the current-to-target limit remains unchanged. A rejected action moves neither arm nor gripper; '
                'the remaining chunk is discarded. Use the receipt to distinguish the executed prefix '
                'from rejected actions, and replan with smaller increments or another approach.'
            )

    def reset(self):
        super().reset()
        self.turns = 0
        self.stable_prefix_hash = None
        self.cache_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.cache_reported_responses = 0
        self.cache_hit_responses = 0
        self.seen_call_ids = set()
        self.returned_reasoning_items = []
        self.reasoning_replay_disabled = False

    def _observations(self, steps, cameras=CAMERAS):
        content = []
        for step in steps:
            entry = self.archive[step]
            state = {'observation_step': step, 'robot_state': entry['state']}
            content.append({'type': 'input_text', 'text': json.dumps(state, ensure_ascii=False)})
            c = self.dialogue_config
            images = {camera: prompt_image(entry['images'][camera], c.image_profile) for camera in cameras}
            content.extend(camera_input_parts(step, images, cameras, profile=c.image_profile, detail=c.image_detail,
                                              image_layout=c.image_layout))
        return content

    def payload_metrics(self, payload):
        c = self.dialogue_config
        images = []
        def scrub(value):
            if isinstance(value, dict):
                if value.get('type') == 'input_image':
                    if value.get('detail') != c.image_detail:
                        raise ValueError('Image detail differs from the configured image contract')
                    validate_prompt_image(value['image_url'], profile=c.image_profile, detail=c.image_detail,
                                          image_layout=c.image_layout)
                    images.append(value)
                    return {'type': 'input_image', 'detail': value['detail']}
                return {k: scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v) for v in value]
            return value
        text_bytes = len(json.dumps(scrub(payload), ensure_ascii=False).encode())
        wire_bytes = len(json.dumps(payload).encode())
        reserve = text_bytes + len(images)*1024 + self.max_output
        if ((c.max_context_text_bytes is not None and text_bytes > c.max_context_text_bytes) or
                (c.max_context_images is not None and len(images) > c.max_context_images) or
                (c.max_context_wire_bytes is not None and wire_bytes > c.max_context_wire_bytes) or
                (c.max_context_reserved_tokens is not None and reserve > c.max_context_reserved_tokens)):
            raise BudgetExceeded('Dialogue context cap reached; no truncation, no paid compaction, no request')
        return {'text_bytes': text_bytes, 'image_count': len(images), 'wire_bytes': wire_bytes,
                'reserved_tokens': reserve, 'reservation_note': 'Estimate assumes no cache discount; not monetary guarantee'}

    def _log_reasoning_replay(self, payload):
        outgoing = [item for item in payload['input'] if item.get('type') == 'reasoning']
        expected = self.returned_reasoning_items
        if self.reasoning_replay_disabled and not outgoing:
            status = 'disabled_after_provider_rejection'
        elif not expected and not outgoing:
            status = 'no_reasoning_items_provided'
        elif outgoing == expected:
            status = 'exact_replay'
        else:
            status = 'replay_mismatch'
        self._log('reasoning_replay.jsonl', {
            'turn': self.turns+1, 'expected_items': len(expected), 'outgoing_items': len(outgoing),
            'status': status,
            'expected_item_sha256': [_item_digest(item) for item in expected],
            'outgoing_item_sha256': [_item_digest(item) for item in outgoing],
            'note': 'Opaque items compared without decryption; no reasoning item is not evidence of reasoning replay.'})

    def _retry_delay(self, error, attempt, limit):
        if error.recovery is not None:
            return 0.0
        if error.retry_after_s is not None:
            return error.retry_after_s
        return min(2 ** min(attempt-1, 6), 60 if limit is None else 8)

    def _ensure_time_for_retry(self, error, attempt, delay):
        if self.max_seconds is None:
            return
        run_remaining = self.max_seconds - self._active_elapsed()
        if delay < run_remaining:
            return
        self._log('network_recovery.jsonl', {
            'turn': self.turns+1, 'failed_attempt': attempt,
            'retry_skipped': 'run_deadline', 'retry_delay_s': delay,
            'run_remaining_s': max(run_remaining, 0.0),
            'error_code': error.error_code, 'robot_action_replayed': False,
        })
        raise BudgetExceeded('harness_timeout: insufficient time for transport retry') from error

    def _log_retry(self, error, attempt, payload, removed, delay):
        wire = to_wire_payload(payload, self.api_mode)
        if os.environ.get('ASTRA_STREAM', '0') == '1':
            wire['stream'] = True
        self._log('network_recovery.jsonl', {
            'turn': self.turns+1, 'failed_attempt': attempt, 'next_attempt': attempt+1,
            'retained_failed_reservation_tokens': self.pending_reserved_tokens,
            'error_code': error.error_code, 'recovery': error.recovery or 'same_payload',
            'removed_reasoning_items': removed, 'retry_delay_s': delay,
            'next_payload_sha256': hashlib.sha256(json.dumps(wire).encode()).hexdigest(),
            'possible_duplicate_inference_cost': error.recovery is None,
            'robot_action_replayed': False})

    def _request(self, payload):
        self._log_reasoning_replay(payload)
        limit = self.dialogue_config.max_transport_attempts
        capped_failures = {}
        self.transport_logical_id = str(uuid4())
        attempt = 0
        while True:
            attempt += 1
            self.transport_attempt = attempt
            try:
                response = self._request_once(payload)
                break
            except RequestTransportError as error:
                request_budget(self).check_available()
                if not error.retryable or (limit is not None and attempt >= limit):
                    raise
                if error.max_attempts is not None:
                    key = (error.error_code, error.recovery)
                    capped_failures[key] = capped_failures.get(key, 0) + 1
                    if capped_failures[key] >= error.max_attempts:
                        raise
                removed = 0
                if error.recovery == 'drop_reasoning_items':
                    payload, removed = without_reasoning_replay(payload)
                    if not removed:
                        raise
                    self.reasoning_replay_disabled = True
                delay = self._retry_delay(error, attempt, limit)
                self._ensure_time_for_retry(error, attempt, delay)
                self._log_retry(error, attempt, payload, removed, delay)
                if error.recovery is None:
                    time.sleep(delay)
                self.budget_poisoned = False
        self._log_cache(response)
        return response

    def _log_cache(self, response):
        usage = provider_cache_usage(response.get('usage', {}))
        total, cached = usage['input_tokens'], usage['cached_tokens']
        written, valid = usage['cache_write_tokens'], usage['cache_reporting_valid']
        if valid:
            self.cache_reported_responses += 1
            self.cache_hit_responses += cached > 0
            self.cache_input_tokens += total
            self.cache_read_tokens += cached
            if type(written) is int and written >= 0:
                self.cache_write_tokens += written
        self._log('cache.jsonl', {
            'turn': self.turns+1, 'stable_prefix_sha256': self.stable_prefix_hash, **usage,
            'server_diagnostics': response.get('prompt_cache_diagnostics'),
            'request_has_cache_hit': cached > 0 if valid else None,
            'cached_input_fraction': cached/total if valid and total else None,
            'cumulative_request_hit_rate': self.cache_hit_responses/self.cache_reported_responses
            if self.cache_reported_responses else None,
            'cumulative_cached_input_fraction': self.cache_read_tokens/self.cache_input_tokens
            if self.cache_input_tokens else None,
            'note': 'Provider report only; not direct KV inspection or verified invoice'})

    def _convert_tensor(self, rows, state):
        return chunk_commands(rows, state, self.dialogue_config, self.max_chunk)


def _item_digest(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
