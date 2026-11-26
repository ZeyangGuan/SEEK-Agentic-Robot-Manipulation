import json
import os
from pathlib import Path

import numpy as np


CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
RENDER_RETRIES = 30


def _blank(pixels):
    return (pixels.ndim != 3 or pixels.shape[-1] < 3 or not pixels.size
            or not np.isfinite(pixels).all() or not np.any(pixels[..., :3]))


def blank_cameras(observations):
    return [[env_id, name] for env_id, obs in observations.items() for name in CAMERAS
            if _blank(np.asarray(obs.get('vision', {}).get(name, {}).get('color', [])))]


def retry_until_visible(render):
    attempts = []
    for attempt in range(RENDER_RETRIES + 1):
        frames, blank = render()
        if not blank:
            return frames, attempts
        attempts.append({'attempt': attempt, 'blank_cameras': blank})
    return None, attempts


def capture(env, env_ids):
    def render():
        env.obs_manager.render_for_capture()
        observations = env.obs_manager.get_obs(env_idx_list=env_ids)
        by_env = observations if isinstance(observations, dict) else dict(zip(env_ids, observations))
        return observations, blank_cameras(by_env)

    observations, attempts = retry_until_visible(render)
    if observations is not None:
        if attempts:
            _log({'event': 'recovered', 'render_only_retries': len(attempts), 'attempts': attempts})
        return observations
    _log({'event': 'failed', 'attempts': attempts})
    env._policy_failure = {
        'method': 'camera_capture', 'exception_type': 'RuntimeError',
        'classification': 'camera_capture_failure', 'official_score': None,
        'capture_attempts': len(attempts), 'render_only_retries': len(attempts) - 1,
        'blank_cameras': attempts[-1]['blank_cameras'], 'attempts': attempts,
    }
    raise RuntimeError('Camera capture remained blank after 30 render-only retries; no API request sent')


def _log(record):
    path = Path(os.environ['ASTRA_LOG_DIR']) / 'camera_readiness.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
