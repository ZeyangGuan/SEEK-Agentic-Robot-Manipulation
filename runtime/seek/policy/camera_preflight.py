import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


FRAMES = 10
SAVED_FRAMES = (0, FRAMES - 1)
DEVICES = ('cuda', 'cpu')


def _observation_rows(env, frame, folder):
    rows = []
    for camera, data in env.get_obs()['vision'].items():
        pixels = np.asarray(data['color'])
        rows.append({'frame': frame, 'camera': camera, 'route': 'observation',
                     'shape': list(pixels.shape), 'mean': float(pixels.mean()),
                     'max': int(pixels.max())})
        if frame in SAVED_FRAMES:
            Image.fromarray(pixels).save(folder / f'{camera}_{frame}.png')
    return rows


def _render_rows(env, frame):
    rows = []
    for index, view in enumerate(env.capture_manager.tiled_cameras):
        for device in DEVICES:
            raw = view._annotators['rgba'].get_data(device=device)
            pixels = raw.numpy() if hasattr(raw, 'numpy') else np.asarray(raw)
            filled = bool(pixels.size)
            rows.append({'frame': frame, 'camera': index, 'route': device,
                         'product': str(view._render_product.path),
                         'prim_paths': view.prim_paths, 'shape': list(pixels.shape),
                         'mean': float(pixels.mean()) if filled else None,
                         'max': int(pixels.max()) if filled else None})
    return rows


def run(env):
    folder = Path(os.environ['ASTRA_CAMERA_PREFLIGHT'])
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(FRAMES):
        rows += _observation_rows(env, frame, folder)
        rows += _render_rows(env, frame)
    (folder / 'capture_probe.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
