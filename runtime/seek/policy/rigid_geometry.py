import math


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'Expected a {size}-element numeric vector')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError('Geometry requires finite numeric values')
    return [float(x) for x in value]


def quat_matrix(quaternion):
    q = _vector(quaternion, 4)
    length = math.hypot(*q)
    if not math.isfinite(length) or length < 1e-12:
        raise ValueError('A rotation quaternion must have a finite nonzero norm')
    w, x, y, z = [v/length for v in q]
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def matvec(matrix, vector):
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        raise ValueError('Expected a 3 by 3 matrix')
    rows = [_vector(row, 3) for row in matrix]
    v = _vector(vector, 3)
    result = [sum(row[j]*v[j] for j in range(3)) for row in rows]
    if not all(math.isfinite(x) for x in result):
        raise ValueError('Nonfinite geometry result')
    return result


def quat_rotate(quaternion, vector):
    return matvec(quat_matrix(quaternion), vector)


def transform_points(pose, points, inverse=False):
    p = _vector(pose, 7)
    if type(inverse) is not bool:
        raise ValueError('inverse must be a boolean')
    if not isinstance(points, (list, tuple)) or len(points) > 2048:
        raise ValueError('Use a list of at most 2048 XYZ points')
    rotation = quat_matrix(p[3:])
    if inverse:
        rotation = [list(row) for row in zip(*rotation)]
    result = []
    for point in points:
        xyz = _vector(point, 3)
        if inverse:
            result.append(matvec(rotation, [xyz[i]-p[i] for i in range(3)]))
        else:
            rotated = matvec(rotation, xyz)
            result.append(_vector([rotated[i]+p[i] for i in range(3)], 3))
    return result


def box_corners(bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('Bounds must contain minimum and maximum XYZ vectors')
    lo, hi = [_vector(row, 3) for row in bounds]
    if any(lo[i] > hi[i] for i in range(3)):
        raise ValueError('Bounds are inverted')
    return [[x, y, z] for x in (lo[0], hi[0])
            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def box_relation(pose_a, bounds_a, pose_b, bounds_b):
    corners = [transform_points(pose, box_corners(bounds)) for pose, bounds in
               ((pose_a, bounds_a), (pose_b, bounds_b))]
    frames = [list(zip(*quat_matrix(pose[3:]))) for pose in (pose_a, pose_b)]
    axes = list(frames[0]) + list(frames[1])
    for a in frames[0]:
        for b in frames[1]:
            cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
            length = math.hypot(*cross)
            if length > 1e-12:
                axes.append([value/length for value in cross])
    best = None
    for axis in axes:
        intervals = []
        for points in corners:
            values = [sum(axis[i]*point[i] for i in range(3)) for point in points]
            intervals.append([min(values), max(values)])
        forward = intervals[1][0]-intervals[0][1]
        reverse = intervals[0][0]-intervals[1][1]
        gap = max(forward, reverse)
        if best is None or gap > best['axis_gap_m']:
            best = dict(axis_gap_m=gap, axis_environment=[float(v) for v in axis],
                        projection_intervals_m=intervals)
    scale = max(1., *(abs(value) for points in corners for point in points for value in point))
    epsilon = 32 * math.ulp(scale)
    best['relation'] = ('separated' if best['axis_gap_m'] > epsilon else
                        'overlap' if best['axis_gap_m'] < -epsilon else 'touching')
    best['numerical_tolerance_m'] = epsilon
    return best


GUIDANCE = '''
Pure geometry helpers are available to observe/decide: quat_matrix(q),
quat_rotate(q, xyz), matvec(matrix, xyz), box_corners([lo_xyz, hi_xyz]), and
transform_points(pose, xyz_points, inverse). The third positional boolean is
optional and defaults to False. Quaternions are [w,x,y,z];
poses are [x,y,z,w,x,y,z]. Quaternion roundoff is normalized. transform_points
accepts at most 2048 points; pass True to map environment points into that pose.
Use these bounded native calculations instead of repeatedly interpreting matrix
and corner arithmetic. They neither query a scene nor choose a target, contact,
clearance margin or action. A transformed mesh envelope still contains empty
space; surface bounds still enclose only selected visible pixels.
box_relation(pose_a, bounds_a, pose_b, bounds_b) compares two oriented boxes in
the environment frame. axis_gap_m is the largest signed separating-axis gap:
positive certifies box separation and is a lower bound, not the exact Euclidean
distance; negative indicates envelope overlap, not mesh contact or penetration.
relation includes a floating-point-only touching tolerance. It chooses no margin,
action or permission, and checks neither the swept path nor unrepresented geometry.
'''
