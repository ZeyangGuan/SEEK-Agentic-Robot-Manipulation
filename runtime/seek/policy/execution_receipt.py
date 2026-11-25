from copy import deepcopy
import numpy as np
from .action_feedback import tracking_receipt

SCHEMA = 'seek-execution-receipt'
ROBOT_KEYS = tuple(arm+suffix for arm in ('left','right')
                   for suffix in ('_ee_pose','_ee_joint_state','_arm_joint_state',
                                  '_gripper_target','_gripper_aperture','_gripper_joint_positions'))


def execution_receipt(reference, step, command, before, after, checks=None, controller_event=None):
    robot = lambda state: {key:deepcopy(state[key]) for key in ROBOT_KEYS if key in state}
    joints = []
    for check in checks or []:
        key = check['arm']
        target = check.get('target_joints')
        if target is not None and key in after:
            actual = np.asarray(after[key], float)
            joints.append(dict(arm=key, accepted_target=deepcopy(target),
                before=deepcopy(before[key]), after=actual.tolist(),
                response=(actual-np.asarray(before[key],float)).tolist(),
                residual=(np.asarray(target,float)-actual).tolist()))
    return dict(schema=SCHEMA, reference=deepcopy(reference), resulting_step=step, executed=True,
                command=deepcopy(command), state_before=robot(before), state_after=robot(after),
                ik_execution_checks=deepcopy(checks), joint_response=joints,
                tracking=tracking_receipt(command, after), controller_event=deepcopy(controller_event))
