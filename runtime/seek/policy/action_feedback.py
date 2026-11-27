import math

import numpy as np


ARMS = ("left", "right")


def _pair(target, observed, key, size):
    pair = [np.asarray(source[key], dtype=float) for source in (target, observed)]
    if any(value.shape != (size,) or not np.isfinite(value).all() for value in pair):
        raise ValueError(f"Tracking receipt needs {size} finite values for {key}")
    return pair


def _pose_error(desired, actual):
    scale = np.linalg.norm(desired[3:]) * np.linalg.norm(actual[3:])
    if scale < 1e-12:
        raise ValueError("Tracking receipt got a zero quaternion")
    cosine = np.clip(abs(np.dot(desired[3:], actual[3:])) / scale, 0, 1)
    return dict(position_error_m=float(np.linalg.norm(desired[:3] - actual[:3])),
                orientation_error_rad=float(2 * math.acos(cosine)))


def _arm_receipt(arm, target, observed):
    entry = {}
    pose, joints, gripper = f"{arm}_ee_pose", f"{arm}_arm_joint_state", f"{arm}_ee_joint_state"
    if pose in target and pose in observed:
        entry.update(_pose_error(*_pair(target, observed, pose, 7)))
    if joints in target and joints in observed:
        desired, actual = _pair(target, observed, joints, 6)
        entry["max_joint_error_rad"] = float(np.max(np.abs(desired - actual)))
    aperture = observed.get(f"{arm}_gripper_aperture")
    if aperture is not None and gripper in target:
        entry["gripper_aperture_error"] = abs(float(target[gripper][0]) - float(aperture[0]))
    return entry


def tracking_receipt(target, observed):
    receipt = {arm: _arm_receipt(arm, target, observed) for arm in ARMS}
    return {arm: entry for arm, entry in receipt.items() if entry}


class TrackingGuard:
    def __init__(self, position_m=.05, orientation_rad=.5, joint_rad=.5, consecutive=2):
        self.limits = dict(position_error_m=position_m, orientation_error_rad=orientation_rad,
                           max_joint_error_rad=joint_rad)
        if min(self.limits.values()) <= 0 or consecutive < 1:
            raise ValueError("Tracking limits and the consecutive count must be positive")
        self.required = consecutive
        self.count = 0

    def _exceeded(self, entry):
        return any(entry[key] > limit for key, limit in self.limits.items() if key in entry)

    def observe(self, receipt):
        self.count = self.count + 1 if any(self._exceeded(entry) for entry in receipt.values()) else 0
        return self.count >= self.required
