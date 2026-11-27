from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .joint_reference import continuity_reference


TURN = 2 * np.pi
TOLERANCE = 1e-6
PERIODIC_ROBOTS = ('x5',)
REJECTION = ('IK targets failed joint continuity checks before dispatch. '
             'Neither arm nor gripper executed this action. Replan from '
             'the latest observed pose using smaller increments or another approach; '
             'do not compensate for the unexecuted suffix.')


def nearest_legal_angles(target, reference, limits, periodic):
    lower, upper = limits[:, 0], limits[:, 1]
    first = np.ceil((lower - target) / TURN)
    last = np.floor((upper - target) / TURN)
    reachable = first <= last
    turns = np.minimum(np.maximum(np.rint((reference - target) / TURN), first), last).astype(np.int64)
    normalized = np.where(periodic & reachable, target + turns * TURN, target)
    legal = np.where(periodic, reachable, (lower <= target) & (target <= upper))
    return normalized, legal


def arm_joint_limits(manager, robot, env_idx=0):
    articulation = manager.robot_key[manager.robot_list.index(robot)]
    return np.array(articulation.data.joint_pos_limits[env_idx][robot.arm_joint_indices].cpu())


def ik_candidate(manager, robot, solved, env_idx=0):
    return dict(status=solved['status'],
                current=manager.get_joint(robot, env_idx_list=[env_idx])[env_idx],
                target=solved.get('joint_value'),
                joint_limits=arm_joint_limits(manager, robot, env_idx),
                periodic=getattr(robot, 'robot_name', None) in PERIODIC_ROBOTS)


class IKContinuityError(ValueError):
    def __init__(self, checks):
        self.checks = deepcopy(checks)
        self.event = dict(type='action_rejected_ik_continuity', rejected_action_executed=False, reason=REJECTION,
                          checks=deepcopy(self.checks),
                          violations=[deepcopy(check) for check in self.checks if not check['passed']])
        super().__init__(REJECTION)


def _shown(value):
    try:
        vector = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None
    return vector.tolist() if vector.ndim == 1 and vector.size and np.isfinite(vector).all() else None


def _out_of_bounds(check, legal):
    check['error'] = 'joint_target_out_of_bounds'
    check['out_of_bounds_joint_index'] = int(np.flatnonzero(~legal)[0])
    return check['error']


def _failure(check, candidate, previous, limit):
    if not isinstance(candidate, dict):
        return 'invalid_candidate'
    for source, field in (('current', 'current_joints'), ('target', 'target_joints')):
        shown = _shown(candidate.get(source))
        if shown is not None:
            check[field] = shown
    if candidate.get('status') != 'Success':
        return 'invalid_candidate' if candidate.get('status') is None else 'ik_failed'
    try:
        current = np.asarray(candidate['current'], dtype=float)
        target = np.asarray(candidate['target'], dtype=float)
    except (ValueError, TypeError):
        return 'invalid_joint_vector'
    if (current.ndim != 1 or not current.size or current.shape != target.shape
            or not np.isfinite(current).all() or not np.isfinite(target).all()):
        return 'invalid_joint_vector'
    try:
        limits = np.asarray(candidate['joint_limits'], dtype=float)
        periodic = np.broadcast_to(np.asarray(candidate['periodic'], dtype=bool), target.shape)
    except (KeyError, ValueError, TypeError):
        return 'invalid_joint_metadata'
    if (limits.shape != (target.size, 2) or not np.isfinite(limits).all()
            or np.any(limits[:, 0] > limits[:, 1])):
        return 'invalid_joint_metadata'
    normalized, legal = nearest_legal_angles(target, current, limits, periodic)
    check.update(current_joints=current.tolist(), target_joints=target.tolist(),
                 normalized_target_joints=normalized.tolist())
    if not legal.all():
        return _out_of_bounds(check, legal)
    step = np.abs(normalized - current)
    check.update(max_current_delta_rad=float(step.max()), current_delta_joint_index=int(step.argmax()))
    if previous is not None:
        if previous.shape != target.shape:
            return 'joint_dimension_changed'
        continued, legal = nearest_legal_angles(target, previous, limits, periodic)
        if not legal.all():
            return _out_of_bounds(check, legal)
        jump = np.abs(continued - previous)
        check.update(previous_target=previous.tolist(), max_target_delta_rad=float(jump.max()),
                     target_delta_joint_index=int(jump.argmax()))
    check['passed'] = (check['max_current_delta_rad'] <= limit + TOLERANCE
                       and check.get('max_target_delta_rad', 0) <= limit + TOLERANCE)
    return None if check['passed'] else 'joint_target_jump'


def _annotate(checks, rebases):
    for check in checks:
        if check['arm'] in rebases:
            check['reference_rebase'] = rebases[check['arm']]
    return checks


class IKContinuityGuard:
    def __init__(self, max_joint_increment_rad=.15, log_path=None):
        limit = max_joint_increment_rad
        if (isinstance(limit, bool) or not isinstance(limit, (int, float))
                or not np.isfinite(limit) or limit <= 0):
            raise ValueError('The IK joint increment limit is a positive finite number')
        self.limit = float(limit)
        self.previous_targets = {}
        self.log_path = Path(log_path) if log_path else None

    def _assess(self, candidates, references):
        if not isinstance(candidates, dict):
            raise ValueError('IK candidates are given as a mapping from arm to candidate')
        checks = []
        for arm, candidate in candidates.items():
            previous = references.get(arm)
            check = dict(arm=arm, passed=False, limit_rad=self.limit, current_joints=None, target_joints=None,
                         normalized_target_joints=None,
                         previous_target=None if previous is None else previous.tolist())
            checks.append(check)
            error = _failure(check, candidate, previous, self.limit)
            if error is not None:
                check['error'] = error
        if not checks:
            raise ValueError('No IK candidates were given')
        if not all(check['passed'] for check in checks):
            raise IKContinuityError(checks)
        return checks

    def _rebased(self, candidates, checks):
        references, rebases = dict(self.previous_targets), {}
        for check in checks:
            if check.get('error') != 'joint_target_jump':
                continue
            arm = check['arm']
            if check.get('previous_target') is None:
                continue
            candidate = candidates[arm]
            current = np.asarray(check['current_joints'], float)
            previous, legal = nearest_legal_angles(
                np.asarray(check['previous_target'], float), current,
                np.asarray(candidate['joint_limits'], float),
                np.broadcast_to(np.asarray(candidate['periodic'], bool), current.shape))
            if not legal.all():
                continue
            reference, evidence = continuity_reference(current, previous, self.limit)
            if evidence is not None:
                references[arm], rebases[arm] = reference, evidence
        return references, rebases

    def check(self, candidates):
        try:
            return self._assess(candidates, self.previous_targets)
        except IKContinuityError as error:
            references, rebases = self._rebased(candidates, error.checks)
            if not rebases:
                raise
        try:
            checks = self._assess(candidates, references)
        except IKContinuityError as error:
            raise IKContinuityError(_annotate(error.checks, rebases)) from error
        return _annotate(checks, rebases)

    def commit(self, checks):
        if not checks or any(not check.get('passed') for check in checks):
            raise ValueError('Cannot commit rejected or empty IK checks')
        for check in checks:
            self.previous_targets[check['arm']] = np.asarray(check['normalized_target_joints'], dtype=float).copy()

    def sync_joint_targets(self, targets):
        for arm, target in (targets or {}).items():
            try:
                vector = np.asarray(target, dtype=float)
            except (TypeError, ValueError):
                continue
            if vector.ndim == 1 and vector.size and np.isfinite(vector).all():
                self.previous_targets[arm] = vector.copy()

    def record(self, step, checks, accepted):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(attempted_step=step, accepted=accepted, checks=checks), allow_nan=False) + '\n')

    def screen(self, manager, env_idx, step, solved):
        candidates = {arm: ik_candidate(manager, robot, result, env_idx) for arm, (robot, result) in solved.items()}
        try:
            return self.check(candidates)
        except IKContinuityError as error:
            self.record(step, error.checks, accepted=False)
            raise

    def accept(self, step, checks):
        self.record(step, checks, accepted=True)
        self.commit(checks)
        return deepcopy(checks)
