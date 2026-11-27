import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


FRAMES = 10
SAVED_FRAMES = (0, FRAMES - 1)
DEVICES = ('cuda', 'cpu')


def _statistics(pixels, may_be_empty=False):
    measured = pixels.size or not may_be_empty
    return dict(shape=list(pixels.shape), mean=float(pixels.mean()) if measured else None,
                max=int(pixels.max()) if measured else None)


def _observed(env, frame, folder):
    for camera, data in env.get_obs()['vision'].items():
        pixels = np.asarray(data['color'])
        yield dict(frame=frame, camera=camera, route='observation', **_statistics(pixels))
        if frame in SAVED_FRAMES:
            Image.fromarray(pixels).save(folder / f'{camera}_{frame}.png')


def _rendered(env, frame):
    for index, view in enumerate(env.capture_manager.tiled_cameras):
        for device in DEVICES:
            raw = view._annotators['rgba'].get_data(device=device)
            pixels = raw.numpy() if hasattr(raw, 'numpy') else np.asarray(raw)
            yield dict(frame=frame, camera=index, route=device, product=str(view._render_product.path),
                       prim_paths=view.prim_paths, **_statistics(pixels, may_be_empty=True))


def run(env):
    folder = Path(os.environ['ASTRA_CAMERA_PREFLIGHT'])
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(FRAMES):
        rows.extend(_observed(env, frame, folder))
        rows.extend(_rendered(env, frame))
    (folder / 'capture_probe.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
