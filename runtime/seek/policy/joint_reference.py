import numpy as np


def continuity_reference(current, previous, limit):
    if previous is None or limit is None:
        return previous, None
    current, previous = np.asarray(current, float), np.asarray(previous, float)
    if (current.shape != previous.shape or not np.isfinite(current).all()
            or not np.isfinite(previous).all() or not np.isfinite(limit) or limit <= 0):
        raise ValueError('Invalid measured joint continuity reference')
    drift = float(np.max(np.abs(current - previous)))
    if drift <= 2 * limit + 2e-6:
        return previous, None
    return current.copy(), dict(reason='observed_tracking_discontinuity',
        previous_commanded_target=previous.tolist(), observed_joints=current.tolist(),
        max_reference_drift_rad=drift, limit_rad=limit,
        incompatible_previous_bound=True, motion_executed=False)
