from copy import deepcopy
import base64
import cv2
import numpy as np


def measure_patch(archive, current_step, camera, bbox, reference_step):
    if camera not in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        raise ValueError(f"Unknown observation camera {camera!r}; use 'cam_head', 'cam_left_wrist' or 'cam_right_wrist'")
    box = np.asarray(bbox, float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        raise ValueError('Use a normalized camera-local xyxy box')
    if type(reference_step) is not int or reference_step not in archive or reference_step > current_step:
        raise ValueError('Reference must be a real retained observation in this episode')
    patches = []
    for step in (reference_step, current_step):
        raw = base64.b64decode(archive[step]['images'][camera].split(',')[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = np.rint(box * [w, h, w, h]).astype(int)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('ROI has no pixels at this camera resolution')
        patches.append(frame[y1:y2, x1:x2].astype(float) / 255.)
    if patches[0].shape != patches[1].shape:
        raise ValueError('Camera resolution changed between observations')
    return dict(camera=camera, bbox=box.tolist(), reference_step=reference_step,
        current_step=current_step, reference_mean_bgr=patches[0].mean(axis=(0, 1)).tolist(),
        current_mean_bgr=patches[1].mean(axis=(0, 1)).tolist(),
        mean_absolute_change=float(np.abs(patches[1]-patches[0]).mean()),
        signed_mean_change_bgr=(patches[1]-patches[0]).mean(axis=(0, 1)).tolist(),
        scope='Pixel measurement only; camera motion, occlusion and lighting may also change this ROI')


def update_effects(existing, updates, episode, observations, source):
    result = deepcopy(existing)
    for update in updates:
        if set(update) != {'id', 'name', 'status', 'evidence_ids', 'reason'}:
            raise ValueError('Effect needs id, name, status, evidence_ids and reason')
        if (not all(isinstance(update[k], str) and update[k].strip() for k in ('id', 'name', 'reason'))
                or update['status'] not in ('pending', 'confirmed', 'refuted', 'unknown')):
            raise ValueError('Invalid effect identity or assessment')
        ids = update['evidence_ids']
        if not isinstance(ids, list) or not ids or not set(ids) <= set(observations):
            raise ValueError('Task effects cite observed images/features, not action or rejection IDs')
        previous = next((e for e in reversed(result) if e['episode'] == episode and e['id'] == update['id']), None)
        if previous and previous['name'] != update['name']:
            raise ValueError('An effect event keeps its semantic name')
        record = dict(deepcopy(update), episode=episode, source=source, native_truth=False)
        if previous != record:
            result.append(record)
    return result


def current_effects(records, episode):
    result = {}
    for record in records:
        if record['episode'] == episode:
            result[record['id']] = deepcopy(record)
    return list(result.values())


def effect_counts(records):
    counts = {}
    for record in records:
        if record['status'] == 'confirmed':
            counts[record['name']] = counts.get(record['name'], 0) + 1
    return counts
