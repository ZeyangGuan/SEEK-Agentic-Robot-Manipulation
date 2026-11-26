import json
import math
import time


MAX_RETRY_AFTER_S = 60.0
TRUNCATED = 'provider_stream_truncated'
TERMINAL_EVENTS = ('response.failed', 'response.incomplete')
TRACED_EVENTS = ('response.created', 'response.completed', *TERMINAL_EVENTS, 'error')
REPORTED_FIELDS = ('type', 'code', 'reason', 'message', 'param')
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


def _mentions(details, markers):
    if not isinstance(details, dict):
        return False
    text = ' '.join(str(details.get(key, '')).lower() for key in ('type', 'code', 'reason'))
    return any(marker in text for marker in markers)


def is_transient(details):
    return _mentions(details, TRANSIENT_MARKERS)


def is_deterministic(details):
    return _mentions(details, DETERMINISTIC_MARKERS)


def _error_code(details, fallback):
    if not isinstance(details, dict):
        return fallback
    value = details.get('code') or details.get('type') or details.get('reason')
    return str(value)[:128] if value else fallback


def _recovery(error_code, details):
    if error_code == 'invalid_encrypted_content':
        return 'drop_reasoning_items'
    if isinstance(details, dict) and (
            error_code == 'invalid_comparison_response_id'
            or details.get('param') == COMPARISON_PARAMETER):
        return 'drop_comparison_response_id'
    return None


def _classify(details, fallback, retryable, max_attempts):
    error_code = _error_code(details, fallback)
    recovery = _recovery(error_code, details)
    if recovery is not None:
        retryable, max_attempts = True, 2
    return dict(retryable=retryable, error_code=error_code, recovery=recovery,
                max_attempts=max_attempts)


def _describe(source, kind, details):
    if isinstance(details, dict):
        details = {key: details[key] for key in REPORTED_FIELDS if key in details}
    return f'{source} terminal failure: {kind}; ' + json.dumps(details, ensure_ascii=False)[:2000]


def raise_terminal_failure(kind, event, response=None, source='SSE'):
    if response is None:
        nested = event.get('response') if isinstance(event, dict) else None
        response = nested if isinstance(nested, dict) else {}
    details = (event.get('error') or response.get('error')
               or response.get('incomplete_details')
               or event.get('incomplete_details') or event)
    max_attempts = None
    if (kind == 'response.incomplete' and isinstance(details, dict)
            and details.get('reason') == 'content_filter'):
        retryable, max_attempts = True, 3
    elif is_transient(details):
        retryable = True
    elif is_deterministic(details):
        retryable = False
    else:
        retryable, max_attempts = True, 2
    outcome = _classify(details, kind.replace('.', '_'), retryable, max_attempts)
    error_type = {
        'response.incomplete': IncompleteResponseError,
        'response.failed': FailedResponseError,
    }[kind]
    raise error_type(_describe(source, kind, details), **outcome)


def _raise_error_event(event, response):
    details = (event.get('error') or response.get('error')
               or response.get('incomplete_details') or event)
    outcome = _classify(details, 'error', is_transient(details), None)
    error_type = TransientResponseError if outcome['retryable'] else ProviderResponseError
    raise error_type(_describe('SSE', 'error', details), **outcome)


def validate_json_response(response):
    if not isinstance(response, dict):
        raise TypeError('Responses JSON body must be an object')
    status = response.get('status')
    if status in ('failed', 'incomplete'):
        raise_terminal_failure(f'response.{status}', response, response=response, source='JSON')
    nested = response.get('response')
    kind = response.get('type')
    if kind in TERMINAL_EVENTS:
        raise_terminal_failure(
            kind, response, response=nested if isinstance(nested, dict) else {}, source='JSON')
    if isinstance(nested, dict) and nested.get('status') in ('failed', 'incomplete'):
        raise_terminal_failure(
            f"response.{nested['status']}", response, response=nested, source='JSON')
    return response


def _frames(reply, deadline):
    data = []
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('SSE response deadline exceeded')
        line = reply.readline()
        if not line:
            raise TruncatedResponseError(
                'SSE ended without response.completed', error_code=TRUNCATED, max_attempts=2)
        try:
            line = line.decode('utf-8').rstrip('\r\n')
        except UnicodeDecodeError:
            raise TruncatedResponseError('Invalid UTF-8 SSE frame', error_code=TRUNCATED) from None
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            frame, data = '\n'.join(data), []
            if frame.strip():
                yield frame


def read_sse_response(reply, record, deadline, on_event=None):
    output_items = {}
    for frame in _frames(reply, deadline):
        if frame == '[DONE]':
            raise TruncatedResponseError(
                'SSE DONE without response.completed', error_code=TRUNCATED, max_attempts=2)
        try:
            event = json.loads(frame)
        except json.JSONDecodeError:
            raise TruncatedResponseError('Invalid JSON SSE frame', error_code=TRUNCATED) from None
        if not isinstance(event, dict):
            raise ValueError('SSE event must be an object')
        if on_event is not None:
            on_event(event)
        kind = event.get('type')
        if kind == 'response.output_item.done':
            output_items[event['output_index']] = event['item']
        if kind in TRACED_EVENTS:
            record('sse_event', event_type=kind, response_id=event.get('response', {}).get('id'))
        if kind == 'response.completed':
            result = event['response']
            if result.get('status') != 'completed':
                raise ValueError('Contradictory SSE completion status')
            if not result.get('output') and output_items:
                result['output'] = [output_items[index] for index in sorted(output_items)]
            return result
        if kind in TERMINAL_EVENTS:
            raise_terminal_failure(kind, event, response=event.get('response', {}))
        if kind == 'error':
            _raise_error_event(event, event.get('response', {}))


def http_error_details(text):
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    details = parsed.get('error', parsed)
    if not isinstance(details, dict):
        return None, None
    code = details.get('code') or details.get('type')
    param = details.get('param')
    return (str(code)[:128] if code is not None else None,
            str(param)[:256] if param is not None else None)


def retry_after_seconds(headers):
    if not headers:
        return None
    value = headers.get('retry-after') or headers.get('Retry-After')
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)
