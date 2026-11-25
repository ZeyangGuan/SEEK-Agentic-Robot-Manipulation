from __future__ import annotations

import math
import itertools

import cv2
import numpy as np

from .calibrated_geometry import HULL_RESOLUTION, MaskObservation

CELL_MIN_M = HULL_RESOLUTION.cell_m
GRID_CELLS = HULL_RESOLUTION.xy_samples
LEVEL_M = HULL_RESOLUTION.level_m
SWEEP_MAX_M = HULL_RESOLUTION.max_height_m


def _ordered(observations, occlusion_near, camera_names):
    return sorted(zip(observations, occlusion_near, camera_names), key=lambda pair: (
        tuple(pair[0].view.pose.ravel()), pair[0].view.fx, pair[0].view.fy,
        pair[0].view.cx, pair[0].view.cy, pair[0].view.shape,
        pair[0].mask.tobytes(), pair[0].unknown.tobytes()))


def _allows(observation, points, near):
    rows, cols, distance = observation.view.project(points)
    h, w = observation.mask.shape
    inside = ((distance > 1e-6) & (rows >= 0) & (rows <= h-1)
              & (cols >= 0) & (cols <= w-1))
    ri = np.clip(np.floor(np.where(inside, rows, 0.)).astype(int), 0, h-2)
    ci = np.clip(np.floor(np.where(inside, cols, 0.)).astype(int), 0, w-2)
    fr, fc = np.clip(rows-ri, 0, 1), np.clip(cols-ci, 0, 1)

    def sample(field, outside):
        values = (field[ri,ci]*(1-fr)*(1-fc) + field[ri+1,ci]*fr*(1-fc)
                  + field[ri,ci+1]*(1-fr)*fc + field[ri+1,ci+1]*fr*fc)
        return np.where(inside, values, outside) >= .5

    foreground = sample(observation.mask, 0.)
    unknown = sample(observation.unknown, 1.)
    if near is not None:
        nearest_r = np.clip(np.rint(np.where(inside, rows, 0.)).astype(int), 0, h-1)
        nearest_c = np.clip(np.rint(np.where(inside, cols, 0.)).astype(int), 0, w-1)
        front = near[nearest_r, nearest_c]

        unknown &= ~inside | ~np.isfinite(front) | (distance >= front)
    return foreground | unknown, foreground, inside


def _domain(observations, base_z, height):
    ends = []
    for observation in observations:
        rows, cols = np.nonzero(observation.mask)
        if not len(rows):
            continue
        directions = observation.view.rays(rows, cols)
        for z in (base_z, base_z + height):
            dz = directions[:, 2]
            usable = np.abs(dz) > 1e-8
            travel = np.divide(z-observation.view.pos[2], dz,
                               out=np.zeros(len(dz)), where=usable)
            usable &= travel > 0
            if usable.any():
                xy = observation.view.pos[:2] + directions[usable, :2]*travel[usable, None]
                ends.append(xy)
    if not ends:
        return None
    all_ends = np.concatenate(ends)
    low = all_ends.min(axis=0)-CELL_MIN_M
    high = all_ends.max(axis=0)+CELL_MIN_M
    cell = max(CELL_MIN_M, math.sqrt(float(np.prod(high-low))/GRID_CELLS))

    low = np.floor(low/cell)*cell
    counts = np.ceil((high-low)/cell).astype(int)+1
    if not np.isfinite(cell) or (counts < 1).any():
        return None
    return low, counts, cell


def _boundary(volume):
    interior = np.zeros_like(volume)
    interior[1:-1, 1:-1, 1:-1] = (
        volume[1:-1, 1:-1, 1:-1] & volume[:-2, 1:-1, 1:-1]
        & volume[2:, 1:-1, 1:-1] & volume[1:-1, :-2, 1:-1]
        & volume[1:-1, 2:, 1:-1] & volume[1:-1, 1:-1, :-2]
        & volume[1:-1, 1:-1, 2:])
    return volume & ~interior


def _paint_voxels(observation, points, cell, projected):
    offsets = np.array(list(itertools.product((-cell/2,cell/2),
                                              (-cell/2,cell/2),
                                              (-LEVEL_M/2,LEVEL_M/2))))
    rows, cols, depth = observation.view.project(points[:,None,:]+offsets[None,:,:])
    for r,c,d in zip(rows,cols,depth):
        if (d > 0).all():
            polygon = cv2.convexHull(np.rint(np.c_[c,r]).astype(np.int32))
            cv2.fillConvexPoly(projected, polygon, 1)


def _reprojection_score(observation, projected):
    projected = projected.astype(bool)
    known = ~observation.unknown
    predicted = projected & known
    visible = observation.mask & known
    intersection = int((predicted & visible).sum())
    total = int(predicted.sum()+visible.sum())
    return dict(visible_pixels=int(visible.sum()),
                projected_known_pixels=int(predicted.sum()),
                explained_visible_pixels=intersection,
                visible_coverage=intersection/max(1, int(visible.sum())),
                f1=2*intersection/total if total else None)


def _fit_height(volume, low, cell, zz, observations):
    projected = [np.zeros(s.mask.shape, np.uint8) for s in observations]
    best = None
    sweep = []
    for index, plane in enumerate(volume):
        iy, ix = np.nonzero(plane)
        if not len(iy):
            continue
        points = np.c_[low[0]+ix*cell, low[1]+iy*cell, np.full(len(iy), zz[index])]
        for sight, image in zip(observations, projected):
            _paint_voxels(sight, points, cell, image)
        diagnostics = [_reprojection_score(s, image) for s,image in zip(observations,projected)]
        scores = [d['f1'] for d in diagnostics if d['f1'] is not None]
        score = float(np.mean(scores)) if scores else 0.
        sweep.append(dict(index=index, top_environment_m=float(zz[index]+LEVEL_M/2), score=score))
        if best is None or score > best[0]+1e-12:
            best = score, index, diagnostics
    return best, sweep


def measure(observations: list[MaskObservation], base_z: float, occlusion_near=None, camera_names=None):
    if len(observations) < 2:
        raise ValueError('Joint silhouette reconstruction needs at least two views')
    paired = _ordered(observations, occlusion_near or [None]*len(observations),
                      camera_names or [None]*len(observations))
    observations = [s for s, _, _ in paired]
    max_above = max(float(s.view.pos[2]-base_z) for s in observations)
    height = min(SWEEP_MAX_M, max_above-0.05)
    if height <= 0:
        return None
    domain = _domain(observations, base_z, height)
    if domain is None:
        return None
    low, counts, cell = domain
    nx, ny = map(int, counts)
    zz = np.arange(LEVEL_M/2, height, LEVEL_M)+base_z
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    xy = low+np.c_[gx.ravel(), gy.ravel()]*cell
    supported = np.zeros((len(zz), ny, nx), dtype=bool)
    possible = np.zeros_like(supported)
    unseen_count = 0
    outside_count = 0
    for index, z in enumerate(zz):
        points = np.c_[xy, np.full(len(xy), z)]
        allowed = np.ones(len(points), dtype=bool)
        foreground = np.zeros(len(points), dtype=np.uint8)
        common = np.ones(len(points), dtype=bool)
        for observation, near, _ in paired:
            permits, shows, inside = _allows(observation, points, near)
            allowed &= permits
            foreground += shows
            common &= inside
        supported[index] = (allowed & (foreground >= 2) & common).reshape(ny, nx)
        possible[index] = (allowed & (foreground == 1) & common).reshape(ny, nx)
        unseen_count += int((allowed & (foreground == 0)).sum())
        outside_count += int((allowed & (foreground > 0) & ~common).sum())
    if not supported.any():
        return None

    best, sweep = _fit_height(_boundary(supported), low, cell, zz, observations)
    match, selected_height, diagnostics = best
    for diagnostic, (_, _, camera) in zip(diagnostics, paired):
        diagnostic['camera'] = camera
    maximum_height = float(zz[np.nonzero(supported)[0]].max()+LEVEL_M/2-base_z)
    fitted = supported[:selected_height+1]
    iz, iy, ix = np.nonzero(_boundary(fitted))
    points = np.c_[low[0]+ix*cell, low[1]+iy*cell, zz[iz]]
    pz, py, px = np.nonzero(possible)
    possible_bounds = None
    if len(pz):
        extension = np.c_[low[0]+px*cell, low[1]+py*cell, zz[pz]]
        possible_bounds = [extension.min(axis=0).tolist(), extension.max(axis=0).tolist()]
    clipped = bool(supported[-1].any())
    return dict(points=points, height_m=float(zz[iz].max()+LEVEL_M/2-base_z),
                kind='joint_silhouette_fit', views=len(observations),
                match=match, cell_m=float(cell),
                diagnostics=dict(
                    view_constraints=len(observations), view_reprojections=diagnostics,
                    support_z=float(base_z), search_height_m=float(height),
                    grid_cell_m=[float(cell), float(cell), LEVEL_M],
                    foreground_supported_voxels=int(supported.sum()),
                    fitted_voxels=int(fitted.sum()),
                    outer_hull_height_m=maximum_height,
                    height_score_samples=[[r['top_environment_m']-base_z,r['score']] for r in sweep],
                    minimum_foreground_views_for_metric_geometry=2,
                    one_view_supported_possible_voxels=int(possible.sum()),
                    one_view_supported_bounds_environment_m=possible_bounds,
                    allowed_without_foreground_voxels=unseen_count,
                    supported_outside_common_view_voxels=outside_count,
                    clipped_by_search_height=clipped,
                    geometry_scope='Boundary of jointly consistent voxels with foreground evidence in at least two views; possible occluded extensions are separate, and hidden concavities remain unknown'),
                height_ambiguous=clipped or bool(possible.any()) or unseen_count > 0)
