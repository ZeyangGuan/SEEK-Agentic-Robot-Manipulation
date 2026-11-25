import base64
import zlib

import cv2
import numpy as np
from .rigid_geometry import quat_matrix, transform_points


GUIDANCE = '''
For both measure_region and measure_patch, camera must be exactly one of:
'cam_head', 'cam_left_wrist', 'cam_right_wrist'.
When observations use a three-camera triptych, the panels from left to right are:
cam_left_wrist | cam_head | cam_right_wrist.
bbox is normalized within the selected camera's own image, not the full triptych.
Example camera argument: measure_region('cam_head', bbox, lower_bgr, upper_bgr).

measure_region(camera, bbox, lower_bgr, upper_bgr, depth_range_m=None) measures
connected RGB regions
in the CURRENT real image. bbox is normalized camera-local xyxy; the two BGR
triples are inclusive channel bounds in [0,1]. Returns pixel_count (selected pixel
count in the ROI), fraction (selected count / ROI pixel count), components,
total_components and omitted. Each component has pixels (integer pixel count),
area (pixels / FULL camera image pixel count, a fraction in [0,1]),
center (normalized camera-local [x,y]) and bbox (normalized camera-local xyxy).
Use component['pixels'] for pixel-count thresholds, never component['area'].
The center key is 'center', not 'centroid'.
This RGB experiment has no depth channel: omit depth_range_m and do not expect a
surface field. Use measure_rgb or locate_rgb_point for explicitly inferred geometry.
Each result includes unfiltered roi_color_bgr p10/p50/p90 and rgb_pixel_count.
No match has components=[]: this is a measured zero for this ROI/selector.
Interpret it using visibility, identity and selector validity; it has no automatic task label.
Components are sorted by area, not semantic identity. The code must check identity,
ambiguity, applicability and camera movement using current evidence. Thresholds,
regions, associations and transition rules are learned, not supplied by the framework.
Use observe to store these measurements, and decide to derive a current intent.
A color region alone does not prove a task event. Reuse the same measurement code
in test_mechanism during review; no motion or model call occurs inside the probe.
'''


def measure_region(archive, step, camera, bbox, lower_bgr, upper_bgr, depth_range_m=None, reference_pose=None):
    if camera not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        raise ValueError(f"Unknown observation camera {camera!r}; use 'cam_head', 'cam_left_wrist' or 'cam_right_wrist'")
    box = np.asarray(bbox, dtype=float)
    lo, hi = np.asarray(lower_bgr, dtype=float), np.asarray(upper_bgr, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        raise ValueError('Use a normalized camera-local xyxy box')
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
            or not ((0 <= lo).all() and (lo <= hi).all() and (hi <= 1).all())):
        raise ValueError('Use finite inclusive BGR bounds in [0,1]')
    depth_bounds = None
    if reference_pose is not None:
        transform_points(reference_pose, [])
    if depth_range_m is not None:
        depth_bounds = np.asarray(depth_range_m, dtype=float)
        if (depth_bounds.shape != (2,) or not np.isfinite(depth_bounds).all()
                or not 0 <= depth_bounds[0] < depth_bounds[1]):
            raise ValueError('Use finite depth_range_m=[near,far] with 0 <= near < far')
        if camera not in archive.get(step, {}).get('depth', {}):
            raise ValueError('Requested depth selection needs this camera depth channel')
    raw = archive.get(step, {}).get('images', {}).get(camera)
    if raw is None:
        raise ValueError('Requested real observation camera is unavailable')
    image = cv2.imdecode(np.frombuffer(base64.b64decode(raw.split(',')[-1]), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Observation camera cannot be decoded')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box * [w, h, w, h]).astype(int)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('ROI has no pixels at this camera resolution')
    patch = image[y1:y2, x1:x2].astype(float) / 255.
    mask = np.all((patch >= lo) & (patch <= hi), axis=2).astype(np.uint8)
    rgb_count = int(mask.sum())
    color_quantiles = np.percentile(patch, [10, 50, 90], axis=(0, 1)).tolist()
    if depth_bounds is not None:
        channel = archive[step]['depth'][camera]
        depth = np.frombuffer(zlib.decompress(base64.b64decode(channel['data'])),
            dtype='<f4').reshape(channel['depth_shape'])
        if depth.shape != (h,w):
            raise ValueError('Depth selection requires aligned RGB and depth dimensions')
        selected_depth = depth[y1:y2,x1:x2]
        mask &= (np.isfinite(selected_depth) & (selected_depth > 0)
                 & (selected_depth >= depth_bounds[0]) & (selected_depth <= depth_bounds[1]))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    indices = sorted(range(1, count), key=lambda i: (-int(stats[i, 4]), i))
    components = []
    for i in indices[:16]:
        x, y, width, height, area = [int(v) for v in stats[i]]
        components.append(dict(pixels=area, area=area / (w*h),
            center=[float((centers[i, 0]+x1)/w), float((centers[i, 1]+y1)/h)],
            bbox=[(x+x1)/w, (y+y1)/h, (x+x1+width)/w, (y+y1+height)/h]))
        if camera in archive[step].get('depth', {}):
            ys, xs = np.nonzero(labels == i)
            components[-1]['surface'] = surface_measurement(archive[step], camera, xs+x1, ys+y1, reference_pose)
    result = dict(camera=camera, step=step, bbox=box.tolist(), pixel_count=int(mask.sum()),
        rgb_pixel_count=rgb_count,
        roi_color_bgr=dict(p10=color_quantiles[0], p50=color_quantiles[1], p90=color_quantiles[2],
            scope='Unfiltered camera-local ROI channels in [0,1], including background'),
        fraction=float(mask.mean()), components=components, total_components=len(indices),
        omitted=max(0, len(indices)-len(components)),
        scope='Measured pixels only; selection is not identity, contact or a task-effect label')
    if depth_bounds is not None:
        result['depth_range_m'] = depth_bounds.tolist()
    return result


def surface_measurement(row, camera, xs, ys, reference_pose=None, *, principal_frame=False):
    channel = row.get('depth', {}).get(camera)
    if channel is None:
        return None
    depth = np.frombuffer(zlib.decompress(base64.b64decode(channel['data'])), dtype='<f4').reshape(channel['depth_shape'])
    xs, ys = np.asarray(xs, dtype=int), np.asarray(ys, dtype=int)
    values = depth[ys, xs]
    valid = np.isfinite(values) & (values > 0)
    result = dict(valid=bool(valid.any()), pixels=int(valid.sum()), selected_pixels=int(values.size),
        camera=camera, step=row['step'], source='camera_depth_channel', units='meters',
        scope='Visible selected pixel surfaces; not object center, identity, hidden geometry or contact.')
    if not valid.any():
        return result
    x, y, d = xs[valid], ys[valid], values[valid].astype(float)
    k = np.asarray(channel['intrinsic_matrix'], float)
    transform = np.asarray(channel['camera_to_environment'], float)
    rays = np.linalg.solve(k, np.stack([x + 0.5, y + 0.5, np.ones_like(x)]))

    camera_points = (rays * d) * np.array([[1.], [-1.], [-1.]])
    points = (transform[:3, :3] @ camera_points + transform[:3, 3:4]).T
    result.update(point_environment_m=np.median(points, axis=0).tolist(),
        z_p10_p90_m=np.percentile(points[:, 2], [10, 90]).tolist(),
        bounds_environment_m=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        optical_depth_m=float(np.median(d)))
    if principal_frame:
        mean = points.mean(axis=0)
        centered = points-mean
        variance, axes = np.linalg.eigh(centered.T @ centered / len(points))
        order = np.argsort(variance)[::-1]
        variance, axes = variance[order], axes[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        projected = centered @ axes
        lower, upper = projected.min(axis=0), projected.max(axis=0)
        result['principal_frame'] = dict(origin_environment_m=mean.tolist(),
            axes_environment=axes.T.tolist(), variance_m2=variance.tolist(),
            bounds_m=[lower.tolist(), upper.tolist()], extent_m=(upper-lower).tolist(),
            scope='Point-density-weighted visible surface statistics; axis signs and repeated-eigenvalue directions are ambiguous')
    if reference_pose is not None:
        transform_points(reference_pose, [])
        rotation = np.asarray(quat_matrix(reference_pose[3:]), float)
        local = (points-np.asarray(reference_pose[:3], float)) @ rotation
        result['reference_frame'] = dict(pose=list(reference_pose),
            point_m=np.median(local, axis=0).tolist(),
            p10_p90_m=np.percentile(local, [10, 90], axis=0).tolist(),
            bounds_m=[local.min(axis=0).tolist(), local.max(axis=0).tolist()],
            scope='Selected visible depth points in the supplied frame; not hidden extent or contact')
    return result


def center_surface(row, camera, center):
    channel = row.get('depth', {}).get(camera)
    if channel is None:
        return None
    h, w = channel['depth_shape']
    x, y = np.rint(center).astype(int)
    ys, xs = np.mgrid[max(0,y-1):min(h,y+2), max(0,x-1):min(w,x+2)]
    return surface_measurement(row, camera, xs.ravel(), ys.ravel())
