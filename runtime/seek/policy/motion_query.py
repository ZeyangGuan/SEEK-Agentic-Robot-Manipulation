from copy import deepcopy
import numpy as np

from .model_policy import validate_actions,plain
from .tool_geometry import tool_base_motion_relation


class CapabilityRequest(Exception):
    def __init__(self, query):
        self.query = query


def selected_motion_request(queries, action, observation_id, query_id=None):
    matches=[]
    for query in queries:
        result=query['result']
        if (query['observation_id']!=observation_id or query.get('space')=='joint'
                or result['status']!='feasible'
                or (query_id is not None and query['id']!=query_id)):
            continue
        candidate=np.asarray(result['action'],float)
        exact=np.array_equal(candidate,action)
        rounded=np.array_equal([round(float(v),6) for v in candidate],action)
        if exact or rounded:
            matches.append(dict(query_id=query['id'],mode=query['mode'],
                fraction=result['fraction'],requested_action=deepcopy(query['requested_action']),
                association='exact_action' if exact else 'rounded_action',
                diagnostic_only=True))
    if not matches or any(m['requested_action']!=matches[-1]['requested_action'] for m in matches):
        return None
    return matches[-1]


def command(action, robot):
    return validate_actions([dict(mode='ee_delta', left=action[:6], left_gripper=action[6],
        right=action[7:13], right_gripper=action[13])], 1, robot)[0]


def search_prefix(action, robot, evaluate, mode, gripper_targets=None):
    if mode not in ('exact', 'prefix'):
        raise ValueError('Motion query mode must be exact or prefix')
    a = np.asarray(action, dtype=float)
    if a.shape != (14,) or not np.isfinite(a).all():
        raise ValueError('Motion query requires fourteen finite numbers')
    command(a.tolist(), robot)
    moving = [*range(6), *range(7, 13)]

    references = {arm:(gripper_targets or {}).get(arm,robot[arm+'_ee_joint_state'][0]) for arm in ('left','right')}
    grip_change = any(abs(a[i] - references[arm]) > 1e-6
                      for i, arm in ((6, 'left'), (13, 'right')))
    fraction, trials = 1., []
    while True:
        candidate = a.copy()
        candidate[moving] *= fraction
        result = evaluate(candidate.tolist())
        trials.append(dict(fraction=fraction, action=candidate.tolist(), **result))
        if result['feasible']:
            return dict(status='feasible', action=candidate.tolist(), fraction=fraction,
                        trials=trials, gripper_change=grip_change,gripper_reference=references)
        if mode == 'exact' or grip_change or max(abs(candidate[moving])) <= 1e-6:
            return dict(status='unresolved', action=None, fraction=None, trials=trials,
                        reason='No tested command passed; change intent or supply a different candidate',
                        gripper_change=grip_change,gripper_reference=references)
        fraction /= 2


def measure_motion(env, observation, query):
    if query.get('space') == 'joint':
        from .joint_motion import measure_joint_motion
        return measure_joint_motion(env, observation, query)
    from .observed_ik_continuity import IKContinuityError
    robot_state = observation['state']
    guard = env.ik_continuity_guard
    accepted = {}
    solver_tolerances = {}
    def evaluate(action):
        packed = (deepcopy(query['target']) if 'target' in query and
                  np.array_equal(action,query['action']) else command(action, robot_state))
        candidates = {}
        solver_errors = {}
        for robot in env.robot_manager.robot_list:
            if robot.type != 'target':
                continue
            side = robot.arm_name.split('_')[0]
            name = env.robot_manager.process_name(robot.arm_name)
            config = env.robot_manager.ik_solver[robot.robot_name].ik_solver.config
            solver_tolerances[side] = dict(position_m=float(config.position_tolerance),
                                           orientation_rad=float(config.orientation_tolerance))
            solved = env.robot_manager.solve_ik(target_pose=packed[side+'_ee_pose'], env_idx=0, robot=robot)
            solver_errors[side] = {key: solved[key] for key in ('position_error_m', 'orientation_error_rad') if key in solved}
            candidates[name] = dict(status=solved['status'],
                current=env.robot_manager.get_joint(robot, env_idx_list=[0])[0],
                target=solved.get('joint_value'),
                joint_limits=env.robot_manager.get_joint_limits(robot, env_idx_list=[0])[0],
                periodic=getattr(robot, 'robot_name', None) == 'x5')
        try:
            checks = guard.check(candidates)
        except IKContinuityError as error:
            return dict(feasible=False, checks=error.checks, solver_errors=solver_errors)
        accepted.update(command=packed, candidates=deepcopy(candidates))
        return dict(feasible=True, checks=checks, solver_errors=solver_errors)
    targets=observation.get('model_meta',{}).get('commanded_grippers')
    result = search_prefix(query['action'], robot_state, evaluate, query['mode'],targets)

    last_action = result['trials'][-1]['action']
    candidate = (query['target'] if 'target' in query and
                 np.array_equal(last_action, query['action']) else command(last_action, robot_state))
    result['tool_base_relations'] = tool_base_motion_relation(robot_state, candidate)
    if result['status']=='feasible':result['command']=plain(accepted['command'])
    report = dict(id=query['id'], observation_id=query['observation_id'], step=query['step'],
        requested_action=query['action'], mode=query['mode'], result=result,
        source='robot_kinematics', executed=False, solver_tolerances=solver_tolerances)
    for key in ('plan_reference','target'):
        if key in query:report[key]=deepcopy(query[key])

    return report, accepted


def execute_measured_motion(env, action, solution):
    original = env.robot_manager.solve_ik
    def solve(target_pose, env_idx, robot, **kwargs):
        side = robot.arm_name.split('_')[0]
        key = env.robot_manager.process_name(robot.arm_name)
        candidate = solution.get('candidates', {}).get(key)
        expected = solution.get('command', {}).get(side+'_ee_pose')
        current = env.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
        if (candidate is not None and expected is not None and env_idx == 0
                and np.array_equal(np.asarray(target_pose), expected)
                and np.allclose(current, candidate['current'], rtol=0, atol=1e-6)):
            return dict(status='Success', joint_value=deepcopy(candidate['target']))
        if solution.get('space') == 'joint':
            from .observed_ik_continuity import IKContinuityError
            raise IKContinuityError([dict(arm=key, passed=False, error='stale_joint_query',
                reason='Joint intent requires a new query at the current physical state')])
        return original(target_pose=target_pose, env_idx=env_idx, robot=robot, **kwargs)
    env.robot_manager.solve_ik = solve
    try:
        return env.take_action(action)
    finally:
        env.robot_manager.solve_ik = original
