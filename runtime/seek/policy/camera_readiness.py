import json
import os
from pathlib import Path
import numpy as np


def blank_cameras(observations):
    bad = []
    for env_id, obs in observations.items():
        for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            pixels = np.asarray(obs.get('vision', {}).get(name, {}).get('color', []))
            if (pixels.ndim != 3 or pixels.shape[-1] < 3 or not pixels.size
                    or not np.isfinite(pixels).all() or not np.any(pixels[..., :3])):
                bad.append([env_id, name])
    return bad


def capture(env, env_ids):
    rows = []
    for attempt in range(31):
        env.obs_manager.render_for_capture()
        observations = env.obs_manager.get_obs(env_idx_list=env_ids)

        if isinstance(observations, dict):
            observation_map = observations
        else:
            observation_map = {
                env_id: observation
                for env_id, observation in zip(env_ids, observations)
            }
        bad = blank_cameras(observation_map)
        if not bad:
            if rows:
                _log({'event': 'recovered', 'render_only_retries': attempt, 'attempts': rows})
            return observations
        rows.append({'attempt': attempt, 'blank_cameras': bad})
    _log({'event': 'failed', 'attempts': rows})

    env._policy_failure = {
        'method': 'camera_capture', 'exception_type': 'RuntimeError',
        'classification': 'camera_capture_failure', 'official_score': None,
        'capture_attempts': len(rows), 'render_only_retries': len(rows) - 1,
        'blank_cameras': bad, 'attempts': rows,
    }
    raise RuntimeError('Camera capture remained blank after 30 render-only retries; no API request sent')


def _log(record):
    path = Path(os.environ['ASTRA_LOG_DIR'])/'camera_readiness.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
