from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HullResolution:
    cell_m: float = 0.001
    xy_samples: int = 40000
    level_m: float = 0.002
    max_height_m: float = 0.36


HULL_RESOLUTION = HullResolution()


class CalibratedCamera:
    def __init__(self, intrinsic, pose, shape):
        self.intrinsic = np.array(intrinsic, dtype=float, copy=True)
        self.pose = np.array(pose, dtype=float, copy=True)
        if (self.intrinsic.shape != (3, 3) or self.pose.shape != (4, 4)
                or not np.isfinite(self.intrinsic).all() or not np.isfinite(self.pose).all()):
            raise ValueError('Camera matrices must be finite 3x3 and 4x4 arrays')
        if (len(shape) != 2 or any(int(v) != v or v < 2 for v in shape)
                or self.intrinsic[0, 0] <= 0 or self.intrinsic[1, 1] <= 0
                or not np.allclose(self.intrinsic[2], [0, 0, 1])
                or not np.allclose(self.intrinsic[1, :1], [0])):
            raise ValueError('Invalid pinhole intrinsics or image dimensions')
        rotation = self.pose[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)
                or not np.allclose(self.pose[3], [0, 0, 0, 1])):
            raise ValueError('Camera pose must be a rigid camera-to-environment transform')
        self.shape = tuple(map(int, shape))
        self.pos = self.pose[:3, 3]
        self.fx, self.fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        self.cx, self.cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        self._inverse_intrinsic = np.linalg.inv(self.intrinsic)

        axes = np.diag([1., -1., -1.])
        rotation_to_camera = axes @ rotation.T
        self.extrinsic = np.column_stack((rotation_to_camera, -rotation_to_camera @ self.pos))

    def normalized_pixels(self, rows, cols):
        rows, cols = np.broadcast_arrays(np.asarray(rows, float), np.asarray(cols, float))
        pixels = np.stack([cols, rows, np.ones_like(rows)], axis=-1)
        return pixels @ self._inverse_intrinsic.T

    def rays(self, rows, cols):
        optical = self.normalized_pixels(rows, cols)
        return (optical * np.array([1., -1., -1.])) @ self.pose[:3, :3].T

    def project(self, points):
        points = np.asarray(points, float)
        if points.shape[-1:] != (3,):
            raise ValueError('Points must end in xyz coordinates')
        optical = points @ self.extrinsic[:, :3].T + self.extrinsic[:, 3]
        homogeneous = optical @ self.intrinsic.T
        depth = optical[..., 2]
        uv = np.divide(homogeneous[..., :2], depth[..., None],
                       out=np.full_like(homogeneous[..., :2], np.nan),
                       where=np.abs(depth[..., None]) > 1e-12)
        return uv[..., 1], uv[..., 0], depth


class MaskObservation:
    def __init__(self, view, mask, unknown=None):
        self.view = view
        self.mask = np.array(mask, dtype=bool, copy=True)
        self.unknown = (np.zeros(view.shape, dtype=bool) if unknown is None
                        else np.array(unknown, dtype=bool, copy=True))
        if self.mask.shape != view.shape or self.unknown.shape != view.shape:
            raise ValueError('Foreground and unknown masks must match camera dimensions')


def support_plane_points(observation, support_z):
    if not np.isfinite(support_z):
        raise ValueError('Support plane must be finite')

    rows, cols = np.nonzero(observation.mask)
    if not rows.size:
        return None
    rays = observation.view.rays(rows, cols)
    travel = np.divide(float(support_z) - observation.view.pos[2], rays[:, 2],
                       out=np.full(rows.size, np.nan), where=np.abs(rays[:, 2]) > 1e-9)
    visible = np.isfinite(travel) & (travel > 0)
    if not visible.any():
        return None
    return dict(points=observation.view.pos + rays[visible] * travel[visible, None],
                usable_pixels=int(visible.sum()), rejected_pixels=int((~visible).sum()))


def triangulate(cameras, pixels):
    if len(cameras) < 2 or len(cameras) != len(pixels):
        raise ValueError('Triangulation needs at least two corresponding views')
    equations, rays = [], []
    for camera, pixel in zip(cameras, pixels):
        row, col = map(float, pixel)
        if not np.isfinite([row, col]).all():
            raise ValueError('Point observations must be finite')
        normalized = camera.normalized_pixels(row, col)
        projection = camera.extrinsic
        equations.extend([normalized[0] * projection[2] - projection[0],
                          normalized[1] * projection[2] - projection[1]])
        ray = camera.rays(row, col)
        rays.append(ray / np.linalg.norm(ray))
    rays = np.asarray(rays)
    cosines = np.clip(np.abs(rays @ rays.T), 0., 1.)
    angle = float(np.degrees(np.arccos(cosines)).max())
    system = np.asarray(equations)
    _, singular, right = np.linalg.svd(system, full_matrices=False)
    condition = float(singular[0] / max(singular[-2], np.finfo(float).tiny))
    homogeneous = right[-1]
    diagnostics = dict(solver='calibrated_dlt', ray_angle_deg=angle,
                       condition=condition, condition_kind='dlt_nonnull_singular_ratio')
    if (angle < 1e-5 or not np.isfinite(condition) or condition > 1e12
            or abs(homogeneous[3]) < 1e-12):
        return dict(diagnostics, valid=False, reason='degenerate_ray_geometry')
    point = homogeneous[:3] / homogeneous[3]
    offsets = point - np.array([camera.pos for camera in cameras])
    distances = np.linalg.norm(np.cross(offsets, rays), axis=1)
    errors, forward = [], True
    for camera, pixel in zip(cameras, pixels):
        row, col, depth = camera.project(point)
        errors.append(float(np.linalg.norm(np.array([row, col]) - pixel)))
        forward = forward and bool(depth > 0)
    return dict(diagnostics, valid=forward, point_environment_m=point.tolist(),
                forward=forward, residual_m=float(np.sqrt(np.mean(distances ** 2))),
                reprojection_error_px=errors)
