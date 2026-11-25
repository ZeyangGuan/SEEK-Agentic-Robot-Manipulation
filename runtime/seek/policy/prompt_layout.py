from copy import deepcopy
import hashlib
import json


ACTION_GUIDANCE = '''
The two user input messages are parts of ONE current decision packet, not successive
turns or new observations. Merge their top-level fields and their context fields.
The first part contains context.experience (prior-episode receipts); the final
part contains the CURRENT context including history, last_execution and errors.
All parts are supplied in full on every request. Stable placement does not make
old evidence current or turn a model hypothesis into a verified mechanism.

If only the stable prefix parts are present and the final CURRENT context part
is absent, this request only prepares a cache. Return act with empty actions and
bindings, null program and state_update, and execution_note="cache_prefix_ready".
That preparation reply will never be executed. Complete decision packets still
require a normal act decision using the final current context and observations.
'''

def _message(value):
    part = dict(type='input_text', text=json.dumps(value, ensure_ascii=False, allow_nan=False))
    return dict(role='user', content=[part])


def action_inputs(packet, observations):
    dynamic = deepcopy(packet)
    shared = {key: dynamic.pop(key) for key in (
        'official_task_prompt', 'camera_calibration', 'task', 'prior_episode_results')}
    if 'input_view' in dynamic:
        shared['input_view'] = dynamic.pop('input_view')
    shared['context'] = dict(experience=dynamic['context'].pop('experience'))
    messages = [_message(shared), _message(dynamic)]
    messages[-1]['content'].extend(deepcopy(observations))
    return messages


def restore_action_packet(inputs):
    from .context_window import decode
    shared, dynamic = [decode(json.loads(m['content'][0]['text'])) for m in inputs if m.get('role')=='user']
    shared['context'].update(dynamic.pop('context'))
    return dict(shared, **dynamic)


def with_cache_key(payload):
    key = json.dumps([payload['instructions'], payload['tools']], ensure_ascii=False)
    payload['prompt_cache_key'] = 'seek-prompt-' + hashlib.sha256(key.encode()).hexdigest()[:32]
    return payload
