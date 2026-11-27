import numpy as np


ARMS = ('left', 'right')
MODES = ('joint', 'ee', 'ee_delta')
ROW_SIZE = 14
TOLERANCE = 1e-6


def _compose(first, second):
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def _check_gripper(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("gripper commands must be finite values in [0,1]")


def _apply_delta(previous, delta):
    if previous.shape != (7,) or not np.isfinite(previous).all():
        raise ValueError("ee_delta requires an observed EE pose; do not mix joint then delta in one chunk")
    angle = float(np.linalg.norm(delta[3:]))
    scale = np.sin(angle/2)/angle if angle > 1e-10 else 0.5
    rotation = np.r_[np.cos(angle/2), delta[3:] * scale]
    return np.r_[previous[:3] + delta[:3], _compose(rotation, previous[3:])].astype(np.float32)


def _normalize_pose(pose):
    norm = float(np.linalg.norm(pose[3:]))
    if norm < 1e-6 or abs(norm - 1) > 0.05:
        raise ValueError("EE quaternion must be unit length, ordered w,x,y,z")
    pose[3:] /= norm
    return pose


def _command(item, targets):
    mode = item.get("mode")
    if mode not in MODES:
        raise ValueError("mode must be joint, ee or ee_delta")
    size = 7 if mode == "ee" else 6
    command = {}
    for arm in ARMS:
        values = np.asarray(item[arm], dtype=np.float32)
        gripper = item[f"{arm}_gripper"]
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"{arm} requires {size} finite numbers for {mode}")
        _check_gripper(gripper)
        if mode == "joint":
            targets[arm] = np.array([])
            command[f"{arm}_arm_joint_state"] = values
        else:
            if mode == "ee_delta":
                values = _apply_delta(targets[arm], values)
            values = _normalize_pose(values)
            targets[arm] = values.copy()
            command[f"{arm}_ee_pose"] = values
        command[f"{arm}_ee_joint_state"] = np.asarray([gripper], dtype=np.float32)
    return command


def validate_actions(raw, max_chunk=32, state=None):
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_chunk:
        raise ValueError(f"actions must contain 1..{max_chunk} control steps")
    observed = state or {}
    targets = {arm: np.asarray(observed.get(f"{arm}_ee_pose", []), dtype=np.float64).copy()
               for arm in ARMS}
    return [_command(item, targets) for item in raw]


def _check_rows(rows, config, max_chunk):
    if config.fixed_predict_horizon and (
            not isinstance(rows, list) or len(rows) != config.predict_horizon):
        raise ValueError(f'actions must contain exactly {config.predict_horizon} control steps; no padding or truncation')
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_chunk:
        raise ValueError('Invalid prediction horizon')
    if any(not isinstance(row, list) or len(row) != ROW_SIZE
           or any(type(value) not in (int, float) for value in row) for row in rows):
        raise ValueError('actions must have shape [T,14] with numeric values')
    tensor = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(tensor).all():
        raise ValueError('Nonfinite actions')
    return tensor


def _check_step(index, arm, vector, config):
    translation = float(np.linalg.norm(vector[:3]))
    rotation = float(np.linalg.norm(vector[3:]))
    if translation > config.max_translation_m+TOLERANCE or rotation > config.max_rotation_rad+TOLERANCE:
        raise ValueError(f'actions[{index}].{arm}: translation L2={translation:.8f}m '
            f'(limit {config.max_translation_m}m), rotation L2={rotation:.8f}rad '
            f'(limit {config.max_rotation_rad}rad); bounds are norms, not per-axis limits')


def _check_joint_step(previous, vector, config):
    if previous.shape != (6,) or np.max(np.abs(vector-previous)) > config.max_joint_increment_rad+TOLERANCE:
        raise ValueError('Absolute joint target change exceeds bound')


def chunk_commands(rows, state, config, max_chunk):
    tensor = _check_rows(rows, config, max_chunk)
    joints = {arm: np.asarray(state.get(f'{arm}_arm_joint_state', [])) for arm in ARMS}
    raw = []
    for index, row in enumerate(tensor):
        item = {'mode': config.action_space, 'left': row[:6], 'left_gripper': float(row[6]),
                'right': row[7:13], 'right_gripper': float(row[13])}
        for arm in ARMS:
            if config.action_space == 'ee_delta':
                _check_step(index, arm, item[arm], config)
            else:
                _check_joint_step(joints[arm], item[arm], config)
                joints[arm] = item[arm]
        raw.append(item)
    return validate_actions(raw, config.predict_horizon, state)
