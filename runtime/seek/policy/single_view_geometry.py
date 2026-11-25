import itertools

import cv2
import numpy as np
from scipy.optimize import minimize

from .calibrated_geometry import HULL_RESOLUTION, support_plane_points


def _unit_shapes():
    box = []
    for fixed in range(3):
        for side in (-1., 1.):
            for a, b in itertools.product(np.linspace(-1., 1., 7), repeat=2):
                point = [a, b]; point.insert(fixed, side); box.append(point)
    theta = np.linspace(0., 2*np.pi, 40, endpoint=False)
    ellipse = []
    for latitude in np.linspace(-np.pi/2, np.pi/2, 13):
        ellipse.extend(np.column_stack((np.cos(latitude)*np.cos(theta),
                                        np.cos(latitude)*np.sin(theta),
                                        np.full(theta.size, np.sin(latitude)))))
    flared = []
    for level in np.linspace(-1., 1., 9):
        radius = .55 + .45*(level+1.)/2.
        flared.extend(np.column_stack((radius*np.cos(theta), radius*np.sin(theta),
                                       np.full(theta.size, level))))
    return {'box': np.unique(box, axis=0), 'ellipsoid': np.asarray(ellipse),
            'flared': np.asarray(flared)}


_SHAPES = _unit_shapes()


def measure(observation, base_z):
    foreground = observation.mask
    if int(foreground.sum()) < 6:
        return None
    projection = support_plane_points(observation, base_z)
    if projection is None or projection['rejected_pixels']:
        return None
    camera = observation.view
    clearance = float(camera.pos[2] - base_z)
    height_limit = min(HULL_RESOLUTION.max_height_m, clearance*.8)
    minimum_height = HULL_RESOLUTION.cell_m
    if height_limit <= minimum_height:
        return None
    xy = projection['points'][:, :2]
    rectangle = cv2.minAreaRect(xy.astype(np.float32))
    origin = np.asarray(rectangle[0], dtype=float)
    radii = np.maximum(np.asarray(rectangle[1], dtype=float)/2., minimum_height)
    orientation = np.radians(rectangle[2])

    known = ~observation.unknown | foreground
    visible_count = int(foreground.sum())
    h, w = foreground.shape
    evaluations = 0

    def geometry(parameters, shape):
        dx, dy, sx, sy, yaw, height = parameters
        fraction = height/(2.*clearance)
        center = origin + fraction*(camera.pos[:2]-origin) + [dx*radii[0], dy*radii[1]]
        axes = radii*np.asarray([sx, sy])*(1.-fraction)
        angle = orientation + yaw
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        points = np.empty_like(shape)
        points[:, :2] = (shape[:, :2]*axes) @ rotation.T + center
        points[:, 2] = float(base_z) + (shape[:, 2]+1.)*height/2.
        return points, axes

    def evaluate(parameters, shape):
        nonlocal evaluations
        evaluations += 1
        points, axes = geometry(parameters, shape)
        rows, cols, depth = camera.project(points)
        if not np.isfinite(rows).all() or not np.isfinite(cols).all() or (depth <= 1e-6).any():
            return 1e6, 0.

        if np.max(np.abs(np.r_[rows, cols])) > 1e6:
            return 1e6, 0.
        polygon = cv2.convexHull(np.rint(np.column_stack((cols, rows))).astype(np.int32))
        rendered = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(rendered, polygon, 1)
        predicted = rendered.astype(bool) & known
        overlap = int((predicted & foreground).sum())
        match = 2.*overlap/max(1, int(predicted.sum())+visible_count)

        compactness = np.log(parameters[-1]/max(2.*min(axes), minimum_height))**2
        return 1.-match + .008*compactness, float(match)

    preferred_height = float(np.clip(2.*min(radii), minimum_height, height_limit))
    heights = np.unique(np.clip(preferred_height*np.array([.25, .5, 1., 2.]),
                                minimum_height, height_limit))
    bounds = [(-.65, .65), (-.65, .65), (.25, 1.4), (.25, 1.4),
              (-np.pi/4, np.pi/4), (minimum_height, height_limit)]
    candidates = []
    for family, shape in _SHAPES.items():
        starts = []
        for height, scale in itertools.product(heights, (.55, .8, 1.)):
            params = np.array([0., 0., scale, scale, 0., height])
            cost, match = evaluate(params, shape)
            starts.append((cost, params, match))
        cost, params, match = min(starts, key=lambda candidate: candidate[0])
        optimized = minimize(lambda q: evaluate(q, shape)[0], params, method='Powell',
                             bounds=bounds, options={'maxiter': 18, 'maxfev': 450,
                                                     'xtol': .003, 'ftol': 1e-5})
        refined_cost, refined_match = evaluate(optimized.x, shape)
        if refined_cost < cost:
            cost, params, match = refined_cost, optimized.x, refined_match
        candidates.append((cost, family, params, match))
    cost, family, parameters, match = min(candidates, key=lambda candidate: candidate[0])
    if not np.isfinite(cost) or match < .35:
        return None
    points, _ = geometry(parameters, _SHAPES[family])
    close_fits = [float(p[-1]) for _, _, p, score in candidates if score >= match-.03]
    return dict(points=points, height_m=float(parameters[-1]),
                kind='single_view_'+family+'_hypothesis', views=1, match=match,
                cell_m=HULL_RESOLUTION.cell_m, height_ambiguous=True,
                diagnostics=dict(view_constraints=1, support_z=float(base_z),
                    shape_prior='support-contacting simple solid with weak compactness preference',
                    candidate_fits=[dict(family=name, height_m=float(p[-1]),
                                        silhouette_match=float(score)) for _, name, p, score in candidates],
                    candidate_height_range_m=[min(close_fits), max(close_fits)],
                    range_scope='Spread across fitted shape families, not a confidence interval',
                    search_height_m=float(height_limit), objective_evaluations=evaluations,
                    geometry_scope='Single-view fitted hypothesis; hidden shape, height and contact remain uncertain'))
