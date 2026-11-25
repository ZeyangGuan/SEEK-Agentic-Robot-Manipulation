import numpy as np

from src.eval_client.ik_continuity import (
    IKContinuityError, IKContinuityGuard as BaseGuard, _nearest_legal_angles)
from .joint_reference import continuity_reference


class IKContinuityGuard(BaseGuard):
    def commit(self, checks):
        if not checks or any(not row.get('passed') for row in checks):
            raise ValueError('Cannot commit rejected or empty IK checks')
        native_commit = getattr(super(), 'commit', None)
        if native_commit is not None:
            return native_commit(checks)
        return self.sync_joint_targets({row['arm']: row['normalized_target_joints'] for row in checks})

    def check(self, candidates):
        try:
            return super().check(candidates)
        except IKContinuityError as error:
            rebases = {}
            references = dict(self.previous_targets)

            for check in error.checks:
                if check.get('error') != 'joint_target_jump':
                    continue
                arm = check['arm']
                if check.get('previous_target') is None:
                    continue
                candidate = candidates[arm]
                current = np.asarray(check['current_joints'], float)
                previous, legal = _nearest_legal_angles(
                    np.asarray(check['previous_target'], float), current,
                    np.asarray(candidate['joint_limits'], float),
                    np.broadcast_to(np.asarray(candidate['periodic'], bool), current.shape))
                if not legal.all():
                    continue
                reference, evidence = continuity_reference(current, previous, self.limit)
                if evidence is not None:
                    references[arm], rebases[arm] = reference, evidence
            if not rebases:
                raise

        original = self.previous_targets
        self.previous_targets = references
        try:
            try:
                checks = super().check(candidates)
            except IKContinuityError as error:
                checks = error.checks
                for check in checks:
                    if check['arm'] in rebases:
                        check['reference_rebase'] = rebases[check['arm']]
                raise IKContinuityError(checks) from error
            for check in checks:
                if check['arm'] in rebases:
                    check['reference_rebase'] = rebases[check['arm']]
            return checks
        finally:
            self.previous_targets = original
