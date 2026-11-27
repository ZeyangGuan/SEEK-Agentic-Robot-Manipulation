from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .action_validation import chunk_commands
from .model_gateway import BudgetExceeded, ModelGateway
from .model_policy import CAMERAS, Model as EpisodeRecord
from .model_request_budget import ModelRequestBudget, request_budget
from .observation_images import camera_input_parts, check_prompt_image, image_rule, prompt_image
from .robot_prompt import control_prompt


DEFAULT_REQUEST_TIMEOUT_S = 600
IMAGE_TOKEN_RESERVE = 1024
EFFORTS = ("low", "medium", "high", "xhigh", "max")
WHOLE_RANGES = dict(
    predict_horizon=(1, 64), execute_horizon=(1, 64), feedback_frames=(1, 8), max_turns=(1, 80),
    max_model_requests=(1, 150), max_output_tokens=(256, 65536), output_token_reservation=(256, 65536),
    max_total_tokens=(1, 20000000), max_seconds=(1, 7200), max_context_text_bytes=(1024, 2048000),
    max_context_images=(3, 512), max_context_wire_bytes=(100000, 32000000),
    max_context_reserved_tokens=(1024, 2048000 + 512 * 1024 + 65536),
    max_transport_attempts=(1, 3), max_action_corrections=(0, 3), request_timeout_s=(1, 900),
)
OPTIONAL = ('max_turns', 'max_model_requests', 'max_output_tokens', 'max_total_tokens', 'max_seconds',
            'max_context_text_bytes', 'max_context_images', 'max_context_wire_bytes',
            'max_context_reserved_tokens', 'max_transport_attempts')
STEP_LIMITS = ('max_translation_m', 'max_rotation_rad', 'max_joint_increment_rad')
SWITCHES = ('experience_horizon_from_budget', 'include_head_calibration', 'fixed_predict_horizon',
            'ik_continuity_guard')
DOCUMENT_TEXT = ('official_task_documentation_version', 'official_task_documentation_source',
                 'official_task_prompt')


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
        self._check_counts()
        self._check_choices()
        self._check_limits()
        self._check_document()

    def _check_counts(self):
        for name, (low, high) in WHOLE_RANGES.items():
            value = getattr(self, name)
            if value is None and name in OPTIONAL:
                continue
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} is an integer from {low} to {high}')

    def _check_choices(self):
        if self.reasoning_effort not in EFFORTS:
            raise ValueError(f'reasoning_effort is one of {EFFORTS}')
        if self.execute_horizon > self.predict_horizon:
            raise ValueError('execute_horizon cannot exceed predict_horizon')
        if self.action_space not in ('ee_delta', 'joint') or self.delta_frame != 'world':
            raise ValueError('action_space is ee_delta or joint, and delta_frame is world')
        if self.cache_mode not in ('implicit', 'omit'):
            raise ValueError('cache_mode is implicit or omit')
        image_rule(self.image_profile, self.image_detail, self.image_layout)
        if self.image_layout == 'triptych' and self.feedback_frames != 1:
            raise ValueError('The triptych layout shows one observation, so feedback_frames must be 1')

    def _check_limits(self):
        for name in STEP_LIMITS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value <= 1:
                raise ValueError(f'{name} is a number in (0, 1]')
        if any(type(getattr(self, name)) is not bool for name in SWITCHES):
            raise ValueError(f'{SWITCHES} are true or false')

    def _check_document(self):
        version, source, prompt = (getattr(self, name) for name in DOCUMENT_TEXT)
        for name in DOCUMENT_TEXT:
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'{name} is a string')
        if bool(prompt) != bool(version) or bool(prompt) != bool(source):
            raise ValueError('The task prompt comes with its documentation version and source')


def load_config(path=None):
    return DialogueConfig(**(json.loads(Path(path).read_text(encoding='utf-8')) if path else {}))


def canonical_tool_history(outputs):
    history = deepcopy(outputs)
    for item in history:
        if item.get('type') == 'function_call':
            for volatile in ('id', 'status'):
                item.pop(volatile, None)
    return history


class Model(EpisodeRecord):
    def __init__(self, cfg):
        settings = self.dialogue_config = load_config(os.environ.get('ASTRA_HARNESS_CONFIG'))
        self.gateway = ModelGateway(
            cfg, settings,
            trace=lambda name, row: self._log(name, row),
            budget=lambda: request_budget(self),
            measure=lambda payload: self.payload_metrics(payload))
        self.max_chunk = settings.predict_horizon
        self.max_output = settings.max_output_tokens or settings.output_token_reservation
        self.effort = os.environ.get("ASTRA_REASONING_EFFORT", settings.reasoning_effort)
        self.prompt = control_prompt(settings)
        super().__init__(cfg)

    def reset(self):
        super().reset()
        self.model_request_budget = ModelRequestBudget(self.dialogue_config.max_model_requests, self.episode_dir)
        self.gateway.new_episode()
        self.turns = 0
        self.stable_prefix_hash = None
        self.seen_call_ids = set()

    def _request(self, payload):
        return self.gateway.send(payload, turn=self.turns + 1, prefix_hash=self.stable_prefix_hash)

    def _convert_tensor(self, rows, state):
        return chunk_commands(rows, state, self.dialogue_config, self.max_chunk)

    def _observations(self, steps, cameras=CAMERAS):
        settings = self.dialogue_config
        parts = []
        for step in steps:
            entry = self.archive[step]
            state = dict(observation_step=step, robot_state=entry['state'])
            parts.append(dict(type='input_text', text=json.dumps(state, ensure_ascii=False)))
            images = {camera: prompt_image(entry['images'][camera], settings.image_profile) for camera in cameras}
            parts += camera_input_parts(step, images, cameras, profile=settings.image_profile,
                                        detail=settings.image_detail, image_layout=settings.image_layout)
        return parts

    def payload_metrics(self, payload):
        settings = self.dialogue_config
        images = []

        def without_pixels(node):
            if isinstance(node, list):
                return [without_pixels(item) for item in node]
            if not isinstance(node, dict):
                return node
            if node.get('type') != 'input_image':
                return {key: without_pixels(item) for key, item in node.items()}
            if node.get('detail') != settings.image_detail:
                raise ValueError('Image detail differs from the configured image contract')
            check_prompt_image(node['image_url'], settings.image_profile, settings.image_detail,
                               settings.image_layout)
            images.append(node)
            return dict(type='input_image', detail=node['detail'])

        text_bytes = len(json.dumps(without_pixels(payload), ensure_ascii=False).encode())
        measured = dict(text_bytes=text_bytes, image_count=len(images),
                        wire_bytes=len(json.dumps(payload).encode()),
                        reserved_tokens=text_bytes + IMAGE_TOKEN_RESERVE * len(images) + self.max_output)
        caps = dict(text_bytes=settings.max_context_text_bytes, image_count=settings.max_context_images,
                    wire_bytes=settings.max_context_wire_bytes,
                    reserved_tokens=settings.max_context_reserved_tokens)
        if any(cap is not None and measured[name] > cap for name, cap in caps.items()):
            raise BudgetExceeded('Dialogue context cap reached; no truncation, no paid compaction, no request')
        return measured
