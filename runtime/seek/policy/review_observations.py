import json
from pathlib import Path
import re
from .boundary_learning import read_observations
from .observation_images import camera_input_parts, prompt_image

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
MAX_OBSERVATIONS = 6
GUIDANCE = '''
inspect_observations reads archived RGB images by their episode-qualified observation ID
(for example e1:s808). Use it to inspect an omitted failure, successful recovery,
or before/after contrast when text and numerical summaries do not settle the issue.
Both the actor and reviewer can request it directly. Compare the actual contact,
object response and changed attempt before revising a method; a stored interpretation
can be reassessed against its original views. Historical images are not the current scene.
visual_observation_episodes lists the episodes with known archives. Prior-episode
views can test a remembered contact or failure interpretation; bind any new action
to the current scene. An unavailable archive is reported, never replaced by a
same-numbered frame from another episode.
Choose up to six observations and the relevant cameras; omitted cameras default to
all three. With a configured three-camera composite, the full original camera layout
is returned and cameras names the views to focus on. Missing frames/cameras are
reported explicitly. This shares the existing
experience/replay read allowance. It does not execute motion or confirm an outcome.
'''


def observation_tool():
    return dict(type='function', name='inspect_observations', strict=False,
        description='Read actual archived RGB observations from the current or remembered episodes by exact observation ID. No motion or installation.',
        parameters=dict(type='object', additionalProperties=False,
            properties=dict(
                observation_ids=dict(type='array', minItems=1, maxItems=MAX_OBSERVATIONS,
                    items=dict(type='string')),
                cameras=dict(type='array', minItems=1, maxItems=len(CAMERAS),
                    items=dict(type='string', enum=list(CAMERAS)))),
            required=['observation_ids']))


def lineage_archives(memory_path):
    root = Path(memory_path).parent
    visited, archives = set(), {}
    while root not in visited and (root/'protocol.json').is_file():
        visited.add(root)
        initial = json.loads((root/'protocol.json').read_text()).get('initial_memory')
        if not isinstance(initial, dict) or not initial.get('source'):
            break
        root = Path(initial['source']).parent
        results = root/'results.json'
        if results.is_file():
            for report in json.loads(results.read_text()).get('episodes', []):
                if report.get('trace'):
                    archives[str(report['episode'])] = str(Path(report['trace'])/'observations.jsonl')
    return archives


def inspect_observations(query, episode, observation_path, *, image_config=None, archives=None):
    if not isinstance(query, dict) or set(query)-{'cameras'} != {'observation_ids'}:
        raise ValueError('inspect_observations requires observation_ids and optional cameras')
    identities = query['observation_ids']
    if not isinstance(identities, list) or not 1 <= len(identities) <= MAX_OBSERVATIONS:
        raise ValueError('Request 1..6 episode-qualified observation IDs')
    cameras = query.get('cameras', list(CAMERAS))
    if (not isinstance(cameras, list) or not 1 <= len(cameras) <= len(CAMERAS)
            or any(camera not in CAMERAS for camera in cameras)):
        raise ValueError('Use the archived head, left-wrist or right-wrist camera names')
    requested = []
    for identity in identities:
        match = re.fullmatch(r'e([0-9]+):s([0-9]+)', identity) if isinstance(identity, str) else None
        if match is None or int(match[1]) > episode:
            raise ValueError('Use an actual current or prior episode observation ID')
        key = (int(match[1]), int(match[2]))
        if key not in requested:
            requested.append(key)
    cameras = list(dict.fromkeys(cameras))
    composite = image_config is not None and image_config.image_layout == 'triptych'
    rendered_cameras = CAMERAS if composite else cameras
    paths = dict(archives or {})
    paths[str(episode)] = str(observation_path)
    loaded = {e: read_observations(Path(paths[str(e)]), [s for n,s in requested if n==e])
              for e in dict.fromkeys(e for e,_ in requested) if str(e) in paths}
    records, parts = [], []
    for source_episode, step in requested:
        identity = f'e{source_episode}:s{step}'
        if source_episode not in loaded:
            records.append(dict(observation_id=identity, status='unavailable_episode_archive', cameras=[]))
            continue
        row = loaded[source_episode].get(step)
        if row is None:
            records.append(dict(observation_id=identity, status='missing', cameras=[]))
            continue
        present = [camera for camera in rendered_cameras if camera in row.get('images', {})]
        missing = [camera for camera in rendered_cameras if camera not in row.get('images', {})]
        records.append(dict(observation_id=identity, status='available' if present else 'missing_cameras',
            requested_cameras=cameras, cameras=present, missing_cameras=missing))
        if composite and missing:
            records[-1]['status'] = 'missing_cameras'
            continue
        if image_config is not None and present:
            images = {camera: prompt_image(row['images'][camera], image_config.image_profile)
                      for camera in present}
            parts.append(dict(type='input_text', text=f'Actual archived observation {identity}; historical camera-local views. Focus: {", ".join(cameras)}.'))
            parts.extend(camera_input_parts(step, images, present,
                profile=image_config.image_profile, detail=image_config.image_detail,
                image_layout=image_config.image_layout))
            continue
        for camera in present:
            parts.extend([
                dict(type='input_text', text=f'Actual archived observation {identity}, {camera}; historical camera-local view.'),
                dict(type='input_image', image_url=row['images'][camera], detail='high')])
    return dict(observations=records, executed=False,
        scope='Archived RGB observations; interpretation and outcome remain to be assessed',
        presentation='Configured live image encoding' if image_config is not None else 'Unmodified archived image data'), parts
