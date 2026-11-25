from copy import deepcopy
import itertools

import cv2
import numpy as np

from .head_calibration_capture import camera_arrays
from .rigid_geometry import quat_matrix
from .segmented_geometry import decode_mask
from .segmentation_client import segment as _shared_segment
from .calibrated_geometry import CalibratedCamera, MaskObservation, triangulate
from . import silhouette_hull, single_view_geometry

GUIDANCE = '''
This experiment is RGB ONLY: no depth images or depth-derived object measurements.
Current calibrated cameras and measured robot state are available; their calibration
does not reveal scene-object geometry. ctx.camera_calibration contains all three
current cameras; each belongs to ctx.step. Wrist calibration changes with motion.

measure_rgb(views, support_z, reference_pose=None): views is a list of dictionaries,
each {camera:'cam_head'|'cam_left_wrist'|'cam_right_wrist', bbox:[x1,y1,x2,y2]}
with a tight normalized camera-local box around the SAME object/part in CURRENT RGB.
Optionally add exclude_boxes (list of normalized boxes) for visible robot/occluders.
The local segmentation model segments each selected box; no model call is used by this measurement.
An optional text field in each view selects a segmentation concept within that box. Ambiguous
or missing text instances return invalid, rather than switching to another object.
With text, optional prompt_mode='joint' supplies the box as a positive visual example
alongside the concept during segmentation inference. The default 'concept' mode detects by
text and then selects by box. Joint prompts can recover an unusual appearance but
can also include adjacent material in the visual example; inspect the selected mask.
The modes remain explicit choices, with no automatic switch after a missing detection.
segment_rgb(camera, text) returns the current segmentation concept candidates, each with
normalized bbox, center, score, pixels and mask_id. Choose the SAME physical instance
across cameras before using measure_rgb; text labels alone do not establish identity.
This is local segmentation, with no model request or depth input. Scores are model
confidence, not geometric accuracy or evidence of a grasp. Existing box-only calls
use the segmentation model's interactive instance head and need no changes to learned RGB programs.
support_z is an EXPLICIT hypothesized
horizontal support height in environment meters, not a measured depth. The standard
benchmark table configuration is 0.765 m; this does not apply to an object in a basket,
held above the table, or resting on another object. Reconsider it in those cases.
Two or more selected views jointly constrain a sampled visual hull, then fit a
height-bounded shape against every mask with equal view weight. Changing view order
does not select another shape. Known tool envelopes mark possible occlusion behind
their calculated near boundary; they cannot erase foreground identified by the segmentation model.
No supplied view is dropped for disagreeing with a shape inferred from another view.
One selected view fits a support-contacting box, ellipsoid or flared solid by RGB
silhouette reprojection, with a weak compactness preference. It returns the same
estimate fields but always sets height_ambiguous=true. Its fitted height is a prior-
dependent hypothesis, not observed depth. A second CURRENT view of the same instance
constrains the ambiguity. Candidate fits are reported under joint_constraints.
Result geometry is under estimate (never surface): point_environment_m, bounds,
principal_frame, height_m, shape_model, views_used, silhouette_match and optional
reference_frame. source='rgb_silhouette_hypothesis'; view/mask/occlusion diagnostics
remain visible. A single view has ambiguous height; even multi-view fit is conditional
on segmentation, same-object association and support assumptions. Multi-view
diagnostics report each mask's explained pixels, voxel resolution, search-height
truncation and allowed material with no foreground evidence. The reported visual
hull fit contains geometry supported by foreground in at least two cameras, within
their common field of view. One-view-supported occluded extensions, height-score
alternatives and the larger outer hull remain separate from that fit. Unseen
material and concavities remain unknown. Silhouette agreement is not contact accuracy. Do not use
its median as a guaranteed material/contact point (rings can have an empty center).
Invalid/missing geometry has valid=false and reason; keep the image evidence.

locate_rgb_point(samples, plane_z=None): samples is a list of {camera, uv:[u,v]}
in normalized CURRENT images for the SAME visible physical surface feature. With
two or more cameras it triangulates rays and reports residual_m, ray_angle_deg,
condition, forward and per-view reprojection_error_px. Different silhouette centers
are not corresponding surface points. One sample requires explicit plane_z and
returns a plane-intersection HYPOTHESIS. No depth image is read by either method.
Use fresh wrist images after approach when they resolve geometry/contact ambiguity;
measurements do not establish capture, retention or task completion. Store and
compare the observed result after closing/lifting/transport. Both functions can run
in observe/decide or test_mechanism and are shared by both classification tasks.
'''

_cache = {}


def _segment(image, box=None, text=None, prompt_mode=None):
    return _shared_segment(image, box, text, prompt_mode=prompt_mode)


def segment_rgb(archive, step, camera, text, segmenter=None):
    row = archive[step]
    view = camera_view(row, camera)
    if not isinstance(text, str) or not text.strip() or len(text) > 256:
        raise ValueError('Text must be a nonempty concept of at most 256 characters')
    import hashlib
    image = row['images'][camera]
    key = (hashlib.sha256(image.encode()).hexdigest(), 'text', text)
    if key not in _cache:
        _cache[key] = (segmenter or _segment)(image, text=text)
        if len(_cache) > 24: del _cache[next(iter(_cache))]
    result = _cache[key]
    candidates = []
    h, w = view.shape
    for candidate in result['candidates']:
        mask = decode_mask(candidate['mask'])
        if mask.shape != view.shape:
            raise ValueError('RGB mask/calibration dimensions disagree')
        candidates.append(dict(bbox=(np.array(candidate['bbox'])/[w,h,w,h]).tolist(),
            center=(np.array(candidate['center'])/[w,h]).tolist(), score=candidate['score'],
            pixels=candidate['area'], mask_id=candidate['mask']['sha256']))
    return dict(step=step, camera=camera, text=text, source='sam3_rgb_concept_candidates',
        candidates=candidates, count=len(candidates),
        scope='Current visible RGB instances; no cross-view identity, depth or contact evidence')


def camera_view(row, camera):
    if camera not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        raise ValueError('Unknown RGB camera')
    calibration = row.get('camera_calibration', {}).get(camera)
    if calibration is None:
        raise ValueError('Current RGB camera calibration unavailable')
    k, pose, shape = camera_arrays(calibration)
    return CalibratedCamera(k, pose, tuple(shape[:2].astype(int)))


def _box(box, shape):
    b = np.asarray(box, float)
    if b.shape != (4,) or not np.isfinite(b).all() or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
        raise ValueError('Use finite normalized camera-local xyxy boxes')
    h, w = shape
    return b * [w, h, w, h]


def tool_pixels(view, state, geometry):
    hit = np.zeros(view.shape, np.uint8)
    for arm in ('left', 'right'):
        pose = state.get(arm+'_ee_pose')
        joints = state.get(arm+'_gripper_joint_positions')
        if pose is None or joints is None:
            continue
        outer = geometry['outer_bounds']
        boxes = [np.asarray(outer['base'], float)]
        for box, axis, joint in zip(outer['fingers_at_zero_joint'], outer['finger_joint_axes'], joints):
            boxes.append(np.asarray(box, float)+np.asarray(axis, float)*float(joint))
        rotation = np.asarray(quat_matrix(pose[3:]), float)
        for bounds in boxes:
            corners = np.array(list(itertools.product(*zip(*bounds))))
            xyz = corners @ rotation.T + np.asarray(pose[:3], float)
            rows, cols, distance = view.project(xyz)
            if np.all(distance > .005):
                polygon = cv2.convexHull(np.rint(np.c_[cols, rows]).astype(np.int32))
                cv2.fillConvexPoly(hit, polygon, 1)
    return hit.astype(bool)


def tool_occlusion_near(view, state, geometry):
    rows, cols = np.indices(view.shape)
    rays = view.rays(rows.ravel(), cols.ravel())
    nearest = np.full(len(rays), np.inf)
    for arm in ('left', 'right'):
        pose = state.get(arm+'_ee_pose')
        joints = state.get(arm+'_gripper_joint_positions')
        if pose is None or joints is None:
            continue
        outer = geometry['outer_bounds']
        boxes = [np.asarray(outer['base'], float)]
        for box, axis, joint in zip(outer['fingers_at_zero_joint'], outer['finger_joint_axes'], joints):
            boxes.append(np.asarray(box, float)+np.asarray(axis, float)*float(joint))
        rotation = np.asarray(quat_matrix(pose[3:]), float)
        origin = (view.pos-np.asarray(pose[:3])) @ rotation
        directions = rays @ rotation
        for bounds in boxes:
            enter = np.full(len(rays), -np.inf)
            leave = np.full(len(rays), np.inf)
            for axis in range(3):
                parallel = np.abs(directions[:, axis]) < 1e-12
                d = np.where(parallel, 1., directions[:, axis])
                a, b = (bounds[:, axis]-origin[axis])[:, None]/d
                enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(a,b)))
                leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(a,b)))
                if not bounds[0, axis] <= origin[axis] <= bounds[1, axis]:
                    leave[parallel] = -np.inf
            hit = leave >= np.maximum(enter, 0.)
            nearest[hit] = np.minimum(nearest[hit], np.maximum(enter[hit], 0.))
    return nearest.reshape(view.shape)


def point_statistics(points, reference_pose=None):
    points = np.asarray(points, float)
    mean = points.mean(axis=0)
    centered = points-mean
    values, axes = np.linalg.eigh(centered.T @ centered / len(points))
    order = np.argsort(values)[::-1]; values, axes = values[order], axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = centered @ axes
    low, high = local.min(axis=0), local.max(axis=0)
    result = dict(point_environment_m=np.median(points, axis=0).tolist(),
        bounds_environment_m=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        principal_frame=dict(origin_environment_m=mean.tolist(), axes_environment=axes.T.tolist(),
            variance_m2=values.tolist(), bounds_m=[low.tolist(), high.tolist()], extent_m=(high-low).tolist()))
    if reference_pose is not None:
        pose = np.asarray(reference_pose, float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > .001:
            raise ValueError('reference_pose must be a finite measured xyz+wxyz pose')
        relative = (points-pose[:3]) @ np.asarray(quat_matrix(pose[3:].tolist()), float)
        result['reference_frame'] = dict(pose=pose.tolist(), point_m=np.median(relative, axis=0).tolist(),
            p10_p90_m=np.percentile(relative, [10,90], axis=0).tolist(),
            bounds_m=[relative.min(axis=0).tolist(), relative.max(axis=0).tolist()])
    return result


def measure_rgb(archive, step, views, support_z, reference_pose=None, tool_geometry=None, segmenter=None):
    if not isinstance(views, list) or not 1 <= len(views) <= 3:
        raise ValueError('Select one to three current RGB views')
    if len({v['camera'] for v in views}) != len(views):
        raise ValueError('Each camera may appear once; repeated views do not add evidence')
    if not np.isfinite(support_z):
        raise ValueError('Support plane must be finite')
    row = archive[step]
    observations, diagnostics, occlusion_near = [], [], []
    for selection in views:
        name = selection['camera']; view = camera_view(row, name)
        image = row['images'][name]; box = _box(selection['bbox'], view.shape)

        import hashlib
        prompt = selection.get('text')
        mode = selection.get('prompt_mode')
        if mode is not None and (mode not in ('concept','joint') or not prompt):
            raise ValueError('prompt_mode requires text and must be concept or joint')
        key = (hashlib.sha256(image.encode()).hexdigest(), tuple(box), prompt, mode or 'concept')
        if key not in _cache:
            options = {'text':prompt} if prompt is not None else {}
            if mode is not None: options['prompt_mode'] = mode
            _cache[key] = (segmenter or _segment)(image, box, **options)
            if len(_cache) > 24:
                del _cache[next(iter(_cache))]
        segmentation = _cache[key]; mask = decode_mask(segmentation['mask'])
        if mask.shape != view.shape:
            raise ValueError('RGB mask/calibration dimensions disagree')
        near = (tool_occlusion_near(view, row['state'], tool_geometry) if tool_geometry
                else np.full(mask.shape, np.inf))
        unknown = np.isfinite(near)
        excluded_pixels = np.zeros(mask.shape, bool)
        for excluded in selection.get('exclude_boxes', []):
            x1,y1,x2,y2 = np.rint(_box(excluded, view.shape)).astype(int)
            unknown[y1:y2, x1:x2] = True
            excluded_pixels[y1:y2, x1:x2] = True
            near[y1:y2, x1:x2] = np.nan

        clean = mask & ~excluded_pixels
        diagnostics.append(dict(camera=name, bbox=selection['bbox'], pixels=int(mask.sum()),
            usable_pixels=int(clean.sum()), robot_or_excluded_pixels=int((mask & unknown).sum()),
            mask_score=segmentation.get('score'), mask_id=segmentation['mask']['sha256'],
            segmentation_valid=segmentation.get('valid'), backend=segmentation.get('backend'),
            text=prompt, prompt_mode=mode or ('concept' if prompt else 'box'),
            reason=segmentation.get('reason')))
        observations.append(MaskObservation(view, clean, unknown))
        occlusion_near.append(near)
    result = dict(step=step, source='rgb_silhouette_hypothesis', valid=False, views=diagnostics,
        support_z=float(support_z), assumptions=['same object in selected masks', 'horizontal support at supplied height',
            ('joint silhouette shape within the common field of view' if len(views) > 1 else 'support-contacting simple solid with ambiguous height'),
            'known tool envelopes order possible occlusion along calibrated rays'],
        scope='RGB-inferred visible geometry; no measured depth, hidden geometry or contact evidence')
    if any(not d['segmentation_valid'] for d in diagnostics):
        return dict(result, reason='segmentation_invalid')
    estimate = (single_view_geometry.measure(observations[0], float(support_z))
                if len(observations) == 1 else silhouette_hull.measure(observations, float(support_z),
                    occlusion_near, [v['camera'] for v in views]))
    if estimate is None:
        return dict(result, reason='no_shape_explains_selected_visible_masks')
    result.update(valid=True, estimate=dict(point_statistics(estimate['points'], reference_pose),
        height_m=estimate['height_m'], shape_model=estimate['kind'], views_used=estimate['views'],
        silhouette_match=estimate['match'], spatial_cell_m=estimate['cell_m']),
        height_ambiguous=estimate.get('height_ambiguous', estimate['views'] < 2))
    if 'diagnostics' in estimate:
        result['estimate']['joint_constraints'] = estimate['diagnostics']
    return result


def locate_rgb_point(archive, step, samples, plane_z=None):
    if not isinstance(samples, list) or not 1 <= len(samples) <= 3:
        raise ValueError('Provide one to three current point observations')
    if len({v['camera'] for v in samples}) != len(samples):
        raise ValueError('Use distinct cameras')
    origins, directions, views, pixels = [], [], [], []
    for sample in samples:
        view = camera_view(archive[step], sample['camera'])
        uv = np.asarray(sample['uv'], float)
        if uv.shape != (2,) or not np.isfinite(uv).all() or not ((uv >= 0) & (uv < 1)).all():
            raise ValueError('uv must be normalized current camera coordinates in [0,1)')
        h,w = view.shape
        col,row = uv * [w,h] - .5
        ray = view.rays(np.array([row]), np.array([col]))[0]; ray /= np.linalg.norm(ray)
        origins.append(view.pos); directions.append(ray); views.append(view); pixels.append([row,col])
    origins, directions = np.asarray(origins), np.asarray(directions)
    result = dict(step=step, source='rgb_calibrated_rays', samples=deepcopy(samples), valid=False)
    if len(samples) == 1:
        if plane_z is None or not np.isfinite(plane_z):
            raise ValueError('A single RGB ray requires an explicit plane_z hypothesis')
        if abs(directions[0,2]) < 1e-8:
            return dict(result, reason='ray_parallel_to_plane')
        t = (plane_z-origins[0,2])/directions[0,2]
        return dict(result, valid=bool(t > 0), source='rgb_plane_hypothesis', plane_z=float(plane_z),
            point_environment_m=(origins[0]+t*directions[0]).tolist(), forward=bool(t > 0))
    result.update(triangulate(views, pixels))
    result['scope'] = 'Correspondence-dependent triangulation; quality values do not prove semantic identity or contact'
    return result
