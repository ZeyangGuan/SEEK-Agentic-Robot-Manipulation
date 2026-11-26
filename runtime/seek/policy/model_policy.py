from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from .action_feedback import tracking_receipt
from .provider_compat import validate_api_mode, validate_endpoint

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = (
    "left_arm_joint_state", "left_ee_joint_state", "left_ee_pose",
    "right_arm_joint_state", "right_ee_joint_state", "right_ee_pose",
    "left_gripper_target", "right_gripper_target",
    "left_gripper_aperture", "right_gripper_aperture",
    "left_gripper_joint_positions", "right_gripper_joint_positions",
)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def image_url(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("Camera must supply an HWC RGB/RGBA array")
    buf = BytesIO()
    Image.fromarray(array[:, :, :3].astype(np.uint8)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Model:
    def __init__(self, model_cfg):
        self.api_mode = validate_api_mode(
            os.environ.get("ASTRA_API_MODE") or model_cfg.get("api_mode", "responses"))
        self.endpoint = validate_endpoint(
            self.api_mode,
            os.environ.get("ASTRA_ENDPOINT") or model_cfg.get("endpoint")
            or "https://api.openai.com/v1/responses",
        )
        self.model_name = model_cfg.get("model", "gpt-6-astra")
        self.api_key = os.environ.get("ASTRA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set SEEK_MODEL_KEY before starting the policy")
        self.effort = os.environ.get("ASTRA_REASONING_EFFORT", model_cfg.get("reasoning_effort", "xhigh"))
        self.max_chunk = int(model_cfg.get("max_chunk", 32))
        configured_output = model_cfg.get("max_output_tokens")
        self.max_output = int(configured_output) if configured_output is not None else None
        self.log_root = Path(os.environ.get("ASTRA_LOG_DIR", "eval_result/model_traces"))
        self.reset()

    def reset(self):
        self.archive = {}
        self.head_calibration_context = None
        self.ledger = []
        self.step = None
        self.episode_id = uuid4().hex
        self.episode_dir = self.log_root / self.episode_id

    def _log(self, name, value):
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        with (self.episode_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + "\n")

    def update_obs(self, obs):
        meta = obs.get("model_meta", {})
        step = int(meta.get("step", len(self.archive)))
        if step in self.archive:
            self.step = step
            return
        if meta.get('head_calibration_context') is not None:
            if self.head_calibration_context is not None or self.archive:
                raise ValueError('Head calibration can only initialize an empty episode')
            self.head_calibration_context = plain(meta['head_calibration_context'])
            self._log('head_calibration_context.jsonl', self.head_calibration_context)
        entry = {
            "step": step,
            "state": {k: plain(obs.get("state", {})[k]) for k in STATE_KEYS if k in obs.get("state", {})},
            "instruction": str(obs.get("instruction", "")),
            "images": {name: image_url(obs["vision"][name]["color"]) for name in CAMERAS},
        }

        entry['camera_calibration'] = {name: plain(obs['vision'][name]['rgb_calibration'])
            for name in CAMERAS if 'rgb_calibration' in obs['vision'][name]}
        entry['perception_mode'] = 'rgb_only'
        if meta.get('max_episode_steps') is not None:
            entry['max_episode_steps'] = int(meta['max_episode_steps'])
        if (meta.get('executed_action') is not None
                and meta.get('ik_execution_checks') is not None):
            entry['ik_execution_checks'] = plain(meta['ik_execution_checks'])
        if meta.get('execution_receipt') is not None:
            entry['execution_receipt'] = plain(meta['execution_receipt'])
        self.archive[step] = entry
        self.step = step
        if meta.get("executed_action") is not None:
            submitted = plain(meta["executed_action"])
            self.ledger.append({"resulting_step": step, "executed_action": submitted,
                                "state_after": entry["state"],
                                "tracking": tracking_receipt(submitted, entry["state"])})
            if meta.get("controller_event"):
                self.ledger[-1]["controller_event"] = plain(meta["controller_event"])
            if meta.get('execution_receipt') is not None:
                self.ledger[-1]['execution_receipt'] = plain(meta['execution_receipt'])
        self._log("observations.jsonl", entry)
