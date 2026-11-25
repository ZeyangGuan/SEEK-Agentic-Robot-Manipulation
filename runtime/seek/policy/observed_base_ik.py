from copy import deepcopy
from types import MethodType


def solve_ik(manager, target_pose, env_idx, robot, trans='world'):
    joints = manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
    target = deepcopy(target_pose)
    if trans == 'relative':
        target = manager._trans_from_endlink_to_gripper(target, robot)
    base = manager.get_link_pose(robot, link_name=robot.base_link,
        env_idx_list=[env_idx], is_relative=True)[env_idx]
    return manager.ik_solver[robot.robot_name].solve_ik_to_joint(
        joints, target, real_robot_pose=base)


def install_observed_base_ik(manager):
    manager.solve_ik = MethodType(solve_ik, manager)
