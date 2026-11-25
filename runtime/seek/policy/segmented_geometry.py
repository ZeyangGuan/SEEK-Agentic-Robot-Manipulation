import base64
import hashlib
import zlib

import numpy as np

from .visual_measurements import surface_measurement


GUIDANCE = """
Tracked camera roles preserve full-mask pixel geometry under segment_geometry. In this RGB experiment they
have no surface depth; use measure_rgb on current views for conditional 3D geometry.
"""


def encode_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('A segmentation mask must have two image dimensions')
    raw = np.packbits(mask.reshape(-1), bitorder='little').tobytes()
    return dict(encoding='zlib-packbits-little', shape=list(mask.shape),
        data=base64.b64encode(zlib.compress(raw)).decode('ascii'),
        sha256=hashlib.sha256(raw).hexdigest())


def decode_mask(value):
    if value['encoding'] != 'zlib-packbits-little':
        raise ValueError('Unsupported image mask encoding')
    shape = value['shape']
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Mask shape must contain two positive image dimensions')
    raw = zlib.decompress(base64.b64decode(value['data']))
    count = shape[0]*shape[1]
    if len(raw) != (count+7)//8 or hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('Mask payload does not match its shape and identity')
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little', count=count).reshape(shape).astype(bool)


def segment_geometry(row, camera, segmentation, reference_pose=None):
    mask = decode_mask(segmentation['mask'])
    channel = row.get('depth', {}).get(camera)
    if channel is not None and tuple(channel['depth_shape']) != mask.shape:
        raise ValueError('Segmentation and depth must describe the same image dimensions')
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    result = dict(camera=camera, step=row['step'], mask_id=segmentation['mask']['sha256'],
        mask_shape=[h,w], pixels=int(xs.size), score=segmentation.get('score'),
        selection_valid=segmentation.get('valid'),
        scope='Current selected image mask and its visible surfaces; not object identity, hidden geometry or contact')
    if xs.size:
        result['bbox'] = [float(xs.min()/w), float(ys.min()/h),
                          float((xs.max()+1)/w), float((ys.max()+1)/h)]
        result['center'] = [float(xs.mean()/w), float(ys.mean()/h)]
    else:
        result.update(bbox=None, center=None)
    if channel is not None:
        result['surface'] = surface_measurement(row, camera, xs, ys, reference_pose,
                                               principal_frame=True)
    return result
