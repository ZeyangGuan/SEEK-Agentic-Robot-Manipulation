from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .action_feedback import TrackingGuard


def command_delta(command, robot):
    result = []
    for arm in ('left', 'right'):
        old = np.asarray(robot[arm+'_ee_pose'], float)
        new = np.asarray(command[arm+'_ee_pose'], float)
        rotation = Rotation.from_quat(new[[4,5,6,3]]) * Rotation.from_quat(old[[4,5,6,3]]).inv()
        result.extend((new[:3]-old[:3]).tolist()+rotation.as_rotvec().tolist()+
                      [float(command[arm+'_ee_joint_state'][0])])
    return result


class CandidatePlan:
    def __init__(self, identity, anchor, rows, targets, origin, tracking_enabled=True, guard=None):
        self.id, self.anchor, self.origin = identity, anchor, origin
        self.rows, self.targets = deepcopy(rows), deepcopy(targets)
        self.cursor = 0
        self.status = 'active' if targets else 'completed'
        self.event = None
        self.tracking = TrackingGuard() if tracking_enabled else None
        self.guard = deepcopy(guard)

    @property
    def active(self):
        return self.status == 'active'

    def reference(self):
        return dict(plan_id=self.id, candidate_id=f'{self.id}:a{self.cursor}',
                    index=self.cursor, anchor_observation_id=self.anchor, origin=self.origin)

    def actions(self, robot):
        if not self.active:
            return []

        before = robot
        result = []
        for target in self.targets[self.cursor:]:
            result.append(command_delta(target, before))
            before = target
        return result

    def view(self):
        result = dict(id=self.id, anchor_observation_id=self.anchor, origin=self.origin,
                      status=self.status, next_index=self.cursor,
                      remaining=len(self.targets)-self.cursor if self.active else 0,
                      event=deepcopy(self.event))
        result['candidate_guard'] = deepcopy(self.guard)
        if self.active:
            result.update(next_reference=self.reference(), target=deepcopy(self.targets[self.cursor]),
                          proposed_increment=deepcopy(self.rows[self.cursor]))
        return result

    def snapshot(self):
        return dict(self.view(), proposed_rows=deepcopy(self.rows), resolved_targets=deepcopy(self.targets))

    def close(self, status, reason):
        self.status = status
        self.event = dict(type='plan_'+status, plan_id=self.id, reason=reason,
                          discarded_candidates=len(self.targets)-self.cursor)
        return deepcopy(self.event)

    def consume(self, receipt):
        if not self.active or receipt['reference'] != self.reference():
            raise ValueError('Execution receipt does not belong to the pending candidate')
        target = self.targets[self.cursor]
        if any(not np.array_equal(target[k], receipt['command'][k]) for k in target):
            raise ValueError('Execution receipt command differs from the dispatched target')
        interrupted = self.tracking is not None and self.tracking.observe(receipt['tracking'])
        self.cursor += 1
        event = receipt.get('controller_event')
        if event is not None or interrupted:
            self.close('interrupted', 'Execution feedback requires a new plan')
            self.event = deepcopy(event) if event is not None else dict(
                type='plan_interrupted_tracking_error', plan_id=self.id,
                consecutive=self.tracking.count, limits=deepcopy(self.tracking.limits),
                discarded_candidates=len(self.targets)-self.cursor,
                reason='Consecutive large arm target residuals in this plan; reobserve and replan')
        elif self.cursor == len(self.targets):
            self.status = 'completed'
        return deepcopy(self.event)
