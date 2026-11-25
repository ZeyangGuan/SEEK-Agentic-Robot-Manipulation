from .rigid_geometry import box_relation


TOOL_GEOMETRY = dict(
    frame='link6', units='meters', finger_direction=[1, 0, 0], opening_axis=[0, 1, 0],
    inner_pad_x=[0.142570, 0.157570], inner_pad_z=[-0.010605, 0.007724],
    inner_pad_y_at_zero_joint=[0.000402, -0.000406],
    finger_joint_range=[0.0, 0.044],
    outer_bounds={'base': [[-1e-08, -0.087, -0.03], [0.084, 0.087, 0.0303048]], 'fingers_at_zero_joint': [[[0.07157, 0.00040158, -0.03064009], [0.15757001, 0.03775159, 0.03035992]], [[0.07157, -0.03775559, -0.03064015], [0.15757001, -0.00040558, 0.03035986]]], 'finger_joint_axes': [[0, 1, 0], [0, -1, 0]]},
)


def tool_base_relation(robot):
    poses = [[float(value) for value in robot[arm+'_ee_pose']] for arm in ('left', 'right')]
    bounds = TOOL_GEOMETRY['outer_bounds']['base']
    return dict(box_relation(poses[0], bounds, poses[1], bounds),
                basis='fixed_tool_base_envelopes_and_supplied_ee_poses',
                scope='Tool bases only; excludes fingers, arm links, objects and swept paths. '
                      'Envelope overlap is not a contact measurement or an execution prohibition.')


def tool_base_motion_relation(current, candidate):
    before, after = tool_base_relation(current), tool_base_relation(candidate)
    return dict(current=before, candidate=after,
                axis_gap_change_m=after['axis_gap_m']-before['axis_gap_m'])
