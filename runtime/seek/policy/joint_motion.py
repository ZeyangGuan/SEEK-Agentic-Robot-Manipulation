from copy import deepcopy
from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .model_policy import plain
from .candidate_plan import command_delta
from .tool_geometry import tool_base_motion_relation


@lru_cache(maxsize=16)
def chain(path, names, base, tip):
    by_child = {j.find('child').get('link'): j for j in ET.parse(path).getroot().findall('joint')}
    result = []
    while tip != base:
        joint = by_child[tip]
        origin, axis = joint.find('origin'), joint.find('axis')
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz',
                np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        direction = np.fromstring(axis.get('xyz', '1 0 0') if axis is not None else '1 0 0', sep=' ')
        result.append((joint.get('name'), joint.get('type'), transform, direction))
        tip = joint.find('parent').get('link')
    result.reverse()
    if {name for name, kind, _, _ in result if kind != 'fixed'} != set(names):
        raise ValueError('Robot arm joint names do not match the URDF chain')
    return result


def forward(robot, joints, base_pose):
    values = dict(zip(robot.arm_joints_name, joints, strict=True))
    base = np.asarray(base_pose, float)
    transform = np.eye(4)
    transform[:3, 3] = base[:3]
    transform[:3, :3] = Rotation.from_quat(base[[4, 5, 6, 3]]).as_matrix()
    for name, kind, origin, axis in chain(str(robot.urdf_path), tuple(robot.arm_joints_name),
                                        robot.base_link, robot.ee_link_name):
        motion = np.eye(4)
        if kind in ('revolute', 'continuous'):
            motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
        elif kind == 'prismatic':
            motion[:3, 3] = axis * values[name]
        elif kind != 'fixed':
            raise ValueError('Unsupported URDF joint type: ' + kind)
        transform = transform @ origin @ motion
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], quat[[3, 0, 1, 2]]]


def validate_proposal(deltas, grippers):
    if not isinstance(deltas, dict) or set(deltas) != {'left', 'right'}:
        raise ValueError('Joint increments require explicit left and right arms')
    if not isinstance(grippers, dict) or set(grippers) != {'left', 'right'}:
        raise ValueError('Gripper targets require explicit left and right arms')
    for arm in ('left', 'right'):
        vector = np.asarray(deltas[arm], float)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError('Each arm needs six finite joint increments in radians')
        grip = grippers[arm]
        if type(grip) not in (int, float) or not np.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError('Gripper targets must be finite numbers in [0,1]')


def measure_joint_motion(env, observation, query):
    from .motion_query import command
    from .observed_ik_continuity import IKContinuityError

    validate_proposal(query['joint_deltas'], query['grippers'])
    manager, state = env.robot_manager, observation['state']
    candidates, targets, geometry = {}, {}, {}
    for robot in manager.robot_list:
        if robot.type != 'target':
            continue
        arm = robot.arm_name.split('_')[0]
        current = np.asarray(manager.get_joint(robot, env_idx_list=[0])[0], float)
        target = current + np.asarray(query['joint_deltas'][arm], float)
        base = manager.get_link_pose(robot, link_name=robot.base_link,
                                    env_idx_list=[0], is_relative=True)[0]
        now, goal = forward(robot, current, base), forward(robot, target, base)
        measured = np.asarray(state[arm+'_ee_pose'], float)
        geometry[arm] = dict(current_pose=now.tolist(), target_pose=goal.tolist(),
            current_position_residual_m=float(np.linalg.norm(now[:3]-measured[:3])),
            current_orientation_residual_rad=float((Rotation.from_quat(now[[4,5,6,3]]) *
                Rotation.from_quat(measured[[4,5,6,3]]).inv()).magnitude()),
            current_joints=current.tolist(), target_joints=target.tolist())
        targets[arm+'_ee_pose'] = goal
        targets[arm+'_ee_joint_state'] = [query['grippers'][arm]]
        candidates[manager.process_name(robot.arm_name)] = dict(status='Success', current=current,
            target=target, joint_limits=manager.get_joint_limits(robot, env_idx_list=[0])[0],
            periodic=getattr(robot, 'robot_name', None) == 'x5')
    action = command_delta(targets, state)
    packed = command(action, state)

    limits = query['limits']
    in_bounds = all(np.linalg.norm(action[offset:offset+3]) <= limits['translation_m']+1e-6 and
                    np.linalg.norm(action[offset+3:offset+6]) <= limits['rotation_rad']+1e-6
                    for offset in (0, 7))
    try:
        checks = env.ik_continuity_guard.check(candidates)
        feasible, reason = in_bounds, None if in_bounds else 'Configured EE action bounds exceeded'
    except IKContinuityError as error:
        checks, feasible, reason = error.checks, False, 'Joint proposal failed the existing execution guard'
    result = dict(status='feasible' if feasible else 'unresolved',
        action=action if feasible else None, fraction=1.0 if feasible else None,
        trials=[dict(fraction=1.0, action=action, feasible=feasible, checks=checks)],
        kinematics=geometry, collision_checked=False,
        tool_base_relations=tool_base_motion_relation(state, packed),
        checks_scope='Joint continuity, joint limits and configured EE increment bounds; no IK or collision solve')
    if feasible:
        result['command'] = plain(packed)
    else:
        result['reason'] = reason
    report = dict(id=query['id'], observation_id=query['observation_id'], step=query['step'],
        requested_action=action, requested_joint_deltas=deepcopy(query['joint_deltas']),
        requested_grippers=deepcopy(query['grippers']), space='joint', mode='exact',
        result=plain(result), source='robot_forward_kinematics', executed=False)
    accepted = dict(command=packed, candidates=deepcopy(candidates), space='joint') if feasible else {}
    return report, accepted
