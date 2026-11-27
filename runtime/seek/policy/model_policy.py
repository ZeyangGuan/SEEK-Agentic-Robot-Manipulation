import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .action_feedback import tracking_receipt
from .observation_images import jpeg_url


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = (
    "left_arm_joint_state", "left_ee_joint_state", "left_ee_pose",
    "right_arm_joint_state", "right_ee_joint_state", "right_ee_pose",
    "left_gripper_target", "right_gripper_target",
    "left_gripper_aperture", "right_gripper_aperture",
    "left_gripper_joint_positions", "right_gripper_joint_positions",
)
ARCHIVE_QUALITY = 90
DEFAULT_TRACE_ROOT = "eval_result/model_traces"


def plain(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def frame_url(frame):
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
        raise ValueError("A camera frame is an HxWx3 or HxWx4 array")
    return jpeg_url(Image.fromarray(pixels[:, :, :3].astype(np.uint8)), quality=ARCHIVE_QUALITY)


class Model:
    def __init__(self, model_cfg):
        self.model_name = model_cfg.get("model", "gpt-6-astra")
        self.log_root = Path(os.environ.get("ASTRA_LOG_DIR", DEFAULT_TRACE_ROOT))
        self.reset()

    def reset(self):
        self.archive = {}
        self.ledger = []
        self.step = None
        self.head_calibration_context = None
        self.episode_id = uuid4().hex
        self.episode_dir = self.log_root / self.episode_id

    def _log(self, name, value):
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        row = json.dumps(plain(value), ensure_ascii=False, allow_nan=False)
        with (self.episode_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(row + "\n")

    def update_obs(self, obs):
        meta = obs.get("model_meta", {})
        step = int(meta.get("step", len(self.archive)))
        known = step in self.archive
        if not known:
            self._accept_calibration(meta.get("head_calibration_context"))
            self.archive[step] = self._entry(step, obs, meta)
        self.step = step
        if not known:
            self._record_execution(step, meta)
            self._log("observations.jsonl", self.archive[step])

    def _accept_calibration(self, context):
        if context is None:
            return
        if self.head_calibration_context is not None or self.archive:
            raise ValueError("Head calibration arrives once, before the first observation")
        self.head_calibration_context = plain(context)
        self._log("head_calibration_context.jsonl", self.head_calibration_context)

    def _entry(self, step, obs, meta):
        state, vision = obs.get("state", {}), obs["vision"]
        entry = dict(
            step=step,
            state={key: plain(state[key]) for key in STATE_KEYS if key in state},
            instruction=str(obs.get("instruction", "")),
            images={name: frame_url(vision[name]["color"]) for name in CAMERAS},
            camera_calibration={name: plain(vision[name]["rgb_calibration"]) for name in CAMERAS
                                if "rgb_calibration" in vision[name]},
            perception_mode="rgb_only",
        )
        if meta.get("max_episode_steps") is not None:
            entry["max_episode_steps"] = int(meta["max_episode_steps"])
        if meta.get("executed_action") is not None and meta.get("ik_execution_checks") is not None:
            entry["ik_execution_checks"] = plain(meta["ik_execution_checks"])
        if meta.get("execution_receipt") is not None:
            entry["execution_receipt"] = plain(meta["execution_receipt"])
        return entry

    def _record_execution(self, step, meta):
        if meta.get("executed_action") is None:
            return
        command, after = plain(meta["executed_action"]), self.archive[step]["state"]
        row = dict(resulting_step=step, executed_action=command, state_after=after,
                   tracking=tracking_receipt(command, after))
        if meta.get("controller_event"):
            row["controller_event"] = plain(meta["controller_event"])
        if meta.get("execution_receipt") is not None:
            row["execution_receipt"] = plain(meta["execution_receipt"])
        self.ledger.append(row)
