import json
import math
import time


MAX_RETRY_AFTER_S = 60.0
TRUNCATED = 'provider_stream_truncated'
TERMINAL_STATUS = ('failed', 'incomplete')
TERMINAL_EVENTS = tuple(f'response.{status}' for status in TERMINAL_STATUS)
TRACED_EVENTS = ('response.created', 'response.completed', *TERMINAL_EVENTS, 'error')
REPORTED_FIELDS = ('type', 'code', 'reason', 'message', 'param')
CLASSIFIED_FIELDS = ('type', 'code', 'reason')
COMPARISON_PARAMETER = 'prompt_cache_options.comparison_response_id'
TRANSIENT_MARKERS = (
    'rate_limit', 'too_many_requests', 'server_error', 'overloaded',
    'temporarily_unavailable', 'service_unavailable', 'provider_capacity',
    'timeout', 'timed_out',
)
DETERMINISTIC_MARKERS = (
    'content_filter', 'content_policy', 'invalid_request',
    'invalid_encrypted_content', 'max_output_tokens', 'max_tokens', 'unsupported_value',
    'insufficient_quota', 'billing_hard_limit',
)


class ProviderResponseError(ValueError):
    retryable = False

    def __init__(self, message, *, retryable=None, error_code=None,
                 recovery=None, max_attempts=None):
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable
        self.error_code = error_code
        self.recovery = recovery
        self.max_attempts = max_attempts


class IncompleteResponseError(ProviderResponseError):
    retryable = True


class FailedResponseError(ProviderResponseError):
    retryable = True


class TransientResponseError(ProviderResponseError):
    retryable = True


class TruncatedResponseError(ProviderResponseError):
    retryable = True


FAILURE_TYPES = {'response.failed': FailedResponseError, 'response.incomplete': IncompleteResponseError}


def _mentions(details, markers):
    if not isinstance(details, dict):
        return False
    text = ' '.join(str(details.get(key, '')).lower() for key in CLASSIFIED_FIELDS)
    return any(marker in text for marker in markers)


def is_transient(details):
    return _mentions(details, TRANSIENT_MARKERS)


def is_deterministic(details):
    return _mentions(details, DETERMINISTIC_MARKERS)


def _classify(details, fallback, retryable, max_attempts):
    named = isinstance(details, dict) and (details.get('code') or details.get('type') or details.get('reason'))
    error_code = str(named)[:128] if named else fallback
    recovery = None
    if error_code == 'invalid_encrypted_content':
        recovery = 'drop_reasoning_items'
    elif isinstance(details, dict) and (error_code == 'invalid_comparison_response_id'
                                        or details.get('param') == COMPARISON_PARAMETER):
        recovery = 'drop_comparison_response_id'
    if recovery is not None:
        retryable, max_attempts = True, 2
    return dict(retryable=retryable, error_code=error_code, recovery=recovery,
                max_attempts=max_attempts)


def _describe(source, kind, details):
    if isinstance(details, dict):
        details = {key: details[key] for key in REPORTED_FIELDS if key in details}
    return f'{source} terminal failure: {kind}; ' + json.dumps(details, ensure_ascii=False)[:2000]


def _retry_policy(kind, details):
    filtered = (kind == 'response.incomplete' and isinstance(details, dict)
                and details.get('reason') == 'content_filter')
    if filtered:
        return True, 3
    if is_transient(details):
        return True, None
    if is_deterministic(details):
        return False, None
    return True, 2


def _fail(kind, event, response, source):
    details = (event.get('error') or response.get('error') or response.get('incomplete_details')
               or event.get('incomplete_details') or event)
    outcome = _classify(details, kind.replace('.', '_'), *_retry_policy(kind, details))
    raise FAILURE_TYPES[kind](_describe(source, kind, details), **outcome)


def _fail_error_event(event, response):
    details = event.get('error') or response.get('error') or response.get('incomplete_details') or event
    outcome = _classify(details, 'error', is_transient(details), None)
    error_type = TransientResponseError if outcome['retryable'] else ProviderResponseError
    raise error_type(_describe('SSE', 'error', details), **outcome)


def validate_json_response(body):
    if not isinstance(body, dict):
        raise TypeError('Responses JSON body must be an object')
    nested = body.get('response')
    inner = nested if isinstance(nested, dict) else {}
    if body.get('status') in TERMINAL_STATUS:
        _fail(f"response.{body['status']}", body, body, 'JSON')
    if body.get('type') in TERMINAL_EVENTS:
        _fail(body['type'], body, inner, 'JSON')
    if inner.get('status') in TERMINAL_STATUS:
        _fail(f"response.{inner['status']}", body, inner, 'JSON')
    return body


def _frames(reply, deadline):
    pending = []
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('SSE response deadline exceeded')
        raw = reply.readline()
        if not raw:
            raise TruncatedResponseError(
                'SSE ended without response.completed', error_code=TRUNCATED, max_attempts=2)
        try:
            line = raw.decode('utf-8').rstrip('\r\n')
        except UnicodeDecodeError:
            raise TruncatedResponseError('Invalid UTF-8 SSE frame', error_code=TRUNCATED) from None
        if line.startswith('data:'):
            pending.append(line[len('data:'):].lstrip(' '))
        elif not line and pending:
            frame, pending = '\n'.join(pending), []
            if frame.strip():
                yield frame


def _event(frame):
    if frame == '[DONE]':
        raise TruncatedResponseError(
            'SSE DONE without response.completed', error_code=TRUNCATED, max_attempts=2)
    try:
        event = json.loads(frame)
    except json.JSONDecodeError:
        raise TruncatedResponseError('Invalid JSON SSE frame', error_code=TRUNCATED) from None
    if not isinstance(event, dict):
        raise ValueError('SSE event must be an object')
    return event


def read_sse_response(reply, record, deadline, on_event=None):
    finished_items = {}
    for frame in _frames(reply, deadline):
        event = _event(frame)
        if on_event is not None:
            on_event(event)
        kind = event.get('type')
        if kind == 'response.output_item.done':
            finished_items[event['output_index']] = event['item']
        if kind in TRACED_EVENTS:
            record('sse_event', event_type=kind, response_id=event.get('response', {}).get('id'))
        if kind == 'response.completed':
            result = event['response']
            if result.get('status') != 'completed':
                raise ValueError('Contradictory SSE completion status')
            if not result.get('output') and finished_items:
                result['output'] = [finished_items[index] for index in sorted(finished_items)]
            return result
        if kind in TERMINAL_EVENTS:
            _fail(kind, event, event.get('response', {}), 'SSE')
        if kind == 'error':
            _fail_error_event(event, event.get('response', {}))


def http_error_details(text):
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    details = parsed.get('error', parsed) if isinstance(parsed, dict) else None
    if not isinstance(details, dict):
        return None, None
    code, parameter = details.get('code') or details.get('type'), details.get('param')
    return (None if code is None else str(code)[:128],
            None if parameter is None else str(parameter)[:256])


def retry_after_seconds(headers):
    value = (headers.get('retry-after') or headers.get('Retry-After')) if headers else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S) if math.isfinite(seconds) else None
