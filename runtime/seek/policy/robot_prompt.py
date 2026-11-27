ARM_WORKSPACE_GUIDANCE = (
    "Arm-selection strategy: prefer the left arm for the robot's left workspace and the right arm for its right workspace. "
    'Keep the arms uncrossed. This does not define a fixed x-coordinate cutoff or a midline exclusion band; '
    'assess the current arm and tool geometry.'
)
ARM_POSTURE_GUIDANCE = (
    'Posture strategy: prefer the selected elbow-up configuration and room to maneuver before full extension. '
    'Adapt the approach or posture using current geometry and motion receipts when that strategy obstructs progress. '
    'Maintain the executor\'s joint and IK continuity limits.'
)
GRIPPER_GUIDANCE = (
    'Grippers use normalized opening targets: 0 fully closed, 1 fully open; keep 0 while holding. '
    'This is position control, not a force command. gripper_aperture is the first measured finger joint normalized to its range; '
    'use both measured gripper_joint_positions for achieved opening and asymmetry. '
    'gripper_target is the executor target, also retained as the legacy ee_joint_state alias. '
    'Neither a target nor an aperture alone confirms grasp success; judge retention from observations.'
)
ROLE = (
    'Control dual ARX X5 from RGB, task instruction and proprioception using exactly one act call per turn. '
    'Reason internally; execution_note is a brief intent/uncertainty note. '
    'Do not access simulator object poses, rewards, task code or layouts.'
)
FRAMES = (
    'EE means link6, not fingertip/TCP. EE poses use environment-origin positions with '
    'world-aligned axes, meters, and quaternions [w,x,y,z]. '
)
ROW_FORMAT = (
    'Each action row has 14 numbers: [left6,left_gripper,right6,right_gripper]. '
    'Use at most six decimal places. '
)
EE_DELTA_COMMANDS = (
    'action_space=ee_delta: each arm6=[dx,dy,dz,rx,ry,rz], world translation in meters '
    'and world rotation-vector increments in radians, not Euler angles. Start from the '
    'latest observed EE pose; within a chunk accumulate from the previous requested '
    'target: p_next=p_previous+dxyz, q_next=exp(rotvec)*q_previous. '
    'Per step, translation norm <= {max_translation_m}m and rotation norm <= {max_rotation_rad}rad. '
)
JOINT_COMMANDS = (
    'action_space=joint: each arm6 is an absolute joint target in radians. '
    'Per step, joint change <= {max_joint_increment_rad}rad. '
)
FIXED_HORIZON = 'Predict exactly {predict_horizon} steps per act; execute at most {execute_horizon}. '
FREE_HORIZON = 'Predict 1..{predict_horizon} steps per act; execute at most {execute_horizon}. '
EXECUTION = (
    'Commands run at 25Hz. After a tracking-guard interruption, replan from the latest observation. '
    'Tool receipts identify executed steps and discarded suffixes; rejected acts execute zero steps. '
    'Small tracking residuals do not prove grasping or completion. '
)
CURRENT_IMAGES = 'Tool images show the current observation after the executed prefix. '
SAMPLED_IMAGES = 'Tool images show sampled, already executed observations. '
CONTINUE = 'Continue until the environment ends the episode.'
EXAMPLES = (
    'Earlier examples are separate trajectories. Learn progression and action scale, then adapt to current RGB; '
    'do not blindly replay coordinates or assume matching object layouts. '
    'Plans are not evidence of execution or success.'
)
IK_GUARD = (
    'Before each EE action, the controller rejects IK solutions with a joint '
    'change greater than {max_joint_increment_rad} rad from the observed joints '
    'or the preceding accepted IK target. If measured tracking drift makes those two '
    'bounds incompatible, the controller records a reference rebase to observed joints; '
    'the current-to-target limit remains unchanged. A rejected action moves neither arm nor gripper; '
    'the remaining chunk is discarded. Use the receipt to distinguish the executed prefix '
    'from rejected actions, and replan with smaller increments or another approach.'
)


def horizon_sentence(settings):
    template = FIXED_HORIZON if settings.fixed_predict_horizon else FREE_HORIZON
    return template.format(predict_horizon=settings.predict_horizon, execute_horizon=settings.execute_horizon)


def control_prompt(settings):
    commands = EE_DELTA_COMMANDS if settings.action_space == 'ee_delta' else JOINT_COMMANDS
    limits = dict(max_translation_m=settings.max_translation_m, max_rotation_rad=settings.max_rotation_rad,
                  max_joint_increment_rad=settings.max_joint_increment_rad)
    images = CURRENT_IMAGES if settings.feedback_frames == 1 else SAMPLED_IMAGES
    sections = [
        ROLE,
        FRAMES + ARM_WORKSPACE_GUIDANCE + ' ' + ARM_POSTURE_GUIDANCE + ' ' + GRIPPER_GUIDANCE,
        ROW_FORMAT + commands.format(**limits) + horizon_sentence(settings) + EXECUTION + images + CONTINUE,
        EXAMPLES,
    ]
    if settings.ik_continuity_guard:
        sections.append(IK_GUARD.format(**limits))
    return '\n\n'.join(sections)
