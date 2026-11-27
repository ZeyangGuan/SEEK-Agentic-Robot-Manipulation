import hashlib
import http.client
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from . import transport_retry
from .eval_failure import safe_error_code
from .provider_compat import (RESPONSES, from_wire_response, provider_cache_usage, to_wire_payload,
                              validate_api_mode, validate_endpoint)
from .provider_response import (
    COMPARISON_PARAMETER, http_error_details, is_deterministic, read_sse_response,
    retry_after_seconds, validate_json_response,
)


DEFAULT_ENDPOINT = "https://api.openai.com/v1/responses"
REQUEST_KIND = 'action'
NETWORK_MODES = ('system', 'direct')
TRACED_HEADERS = ('x-request-id', 'x-oneapi-request-id', 'cf-ray', 'server', 'date')
RETRYABLE_STATUS = (408, 409, 429)
DEADLINE_REACHED = 'run_budget_exhausted: active run deadline reached'


class BudgetExceeded(RuntimeError):
    pass


class RunBudgetExceeded(BudgetExceeded):
    error_code = 'run_budget_exhausted'


class RequestTransportError(RuntimeError):
    def __init__(self, message, *, retryable=False, error_code=None, recovery=None,
                 retry_after_s=None, max_attempts=None):
        super().__init__(message)
        self.retryable = retryable
        self.error_code = error_code
        self.recovery = recovery
        self.retry_after_s = retry_after_s
        self.max_attempts = max_attempts


def streaming():
    return os.environ.get('ASTRA_STREAM', '0') == '1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reasoning_items(payload):
    return [item for item in payload['input'] if item.get('type') == 'reasoning']


def without_reasoning(payload):
    trimmed = deepcopy(payload)
    items = trimmed.get('input')
    if not isinstance(items, list):
        return trimmed, 0
    trimmed['input'] = [item for item in items
                        if not (isinstance(item, dict) and item.get('type') == 'reasoning')]
    return trimmed, len(items) - len(trimmed['input'])


def timed_out(error):
    reason = getattr(error, 'reason', None) if isinstance(error, urllib.error.URLError) else None
    return isinstance(error, TimeoutError) or isinstance(reason, TimeoutError)


def rejected_field(payload, code, parameter):
    recovery = None
    if code == 'invalid_encrypted_content' and any(
            isinstance(item, dict) and item.get('type') == 'reasoning' for item in payload.get('input', [])):
        recovery = 'drop_reasoning_items'
    options = payload.get('prompt_cache_options')
    names_comparison = code == 'invalid_comparison_response_id' or parameter == COMPARISON_PARAMETER
    if names_comparison and isinstance(options, dict) and options.get('comparison_response_id'):
        recovery = 'drop_comparison_response_id'
    return recovery


class ModelGateway:
    def __init__(self, cfg, settings, *, trace, budget, measure):
        self.api_mode = validate_api_mode(os.environ.get("ASTRA_API_MODE") or cfg.get("api_mode", RESPONSES))
        self.endpoint = validate_endpoint(
            self.api_mode, os.environ.get("ASTRA_ENDPOINT") or cfg.get("endpoint") or DEFAULT_ENDPOINT)
        self.api_key = os.environ.get("ASTRA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set SEEK_MODEL_KEY before starting the policy")
        self.trace, self.budget, self.measure = trace, budget, measure
        self.max_attempts = settings.max_transport_attempts
        self.max_requests = (None if settings.max_turns is None or self.max_attempts is None
                             else settings.max_turns * self.max_attempts)
        self.max_total_tokens = settings.max_total_tokens
        self.max_seconds = settings.max_seconds
        self.request_timeout_s = settings.request_timeout_s
        self.request_count = self.reported_tokens = self.reserved_tokens = self.pending_tokens = 0
        self.poisoned = False
        self.started = time.monotonic()
        self.new_episode()

    def new_episode(self):
        self.cache = dict(input=0, read=0, written=0, reported=0, hits=0)
        self.reasoning_dropped = False
        self.returned_reasoning = []

    def send(self, payload, *, turn, prefix_hash=None):
        self._trace_reasoning(payload, turn)
        logical_id = str(uuid4())
        repeats = {}
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._attempt(payload, logical_id, attempt)
                break
            except RequestTransportError as error:
                self.budget().check_available()
                if self._final(error, attempt, repeats):
                    raise
                removed = 0
                if error.recovery == 'drop_reasoning_items':
                    payload, removed = without_reasoning(payload)
                    if not removed:
                        raise
                    self.reasoning_dropped = True
                delay = self._backoff(error, attempt)
                self._require_time(error, attempt, delay, turn)
                self._trace_retry(error, attempt, payload, removed, delay, turn)
                if error.recovery is None:
                    time.sleep(delay)
                self.poisoned = False
        self._trace_cache(response, turn, prefix_hash)
        return response

    def elapsed(self):
        return transport_retry.active_elapsed(self.started)

    def _expired(self):
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds

    def _final(self, error, attempt, repeats):
        if not error.retryable or (self.max_attempts is not None and attempt >= self.max_attempts):
            return True
        if error.max_attempts is None:
            return False
        key = (error.error_code, error.recovery)
        repeats[key] = repeats.get(key, 0) + 1
        return repeats[key] >= error.max_attempts

    def _backoff(self, error, attempt):
        if error.recovery is not None:
            return 0.0
        if error.retry_after_s is not None:
            return error.retry_after_s
        return min(2 ** min(attempt - 1, 6), 60 if self.max_attempts is None else 8)

    def _require_time(self, error, attempt, delay, turn):
        if self.max_seconds is None:
            return
        remaining = self.max_seconds - self.elapsed()
        if delay < remaining:
            return
        self.trace('network_recovery.jsonl', {
            'turn': turn, 'failed_attempt': attempt, 'retry_skipped': 'run_deadline',
            'retry_delay_s': delay, 'run_remaining_s': max(remaining, 0.0),
            'error_code': error.error_code, 'robot_action_replayed': False})
        raise BudgetExceeded('harness_timeout: insufficient time for transport retry') from error

    def _trace_retry(self, error, attempt, payload, removed, delay, turn):
        wire = to_wire_payload(payload, self.api_mode)
        if streaming():
            wire['stream'] = True
        self.trace('network_recovery.jsonl', {
            'turn': turn, 'failed_attempt': attempt, 'next_attempt': attempt + 1,
            'retained_failed_reservation_tokens': self.pending_tokens,
            'error_code': error.error_code, 'recovery': error.recovery or 'same_payload',
            'removed_reasoning_items': removed, 'retry_delay_s': delay,
            'next_payload_sha256': hashlib.sha256(json.dumps(wire).encode()).hexdigest(),
            'possible_duplicate_inference_cost': error.recovery is None,
            'robot_action_replayed': False})

    def _trace_reasoning(self, payload, turn):
        outgoing, expected = reasoning_items(payload), self.returned_reasoning
        if self.reasoning_dropped and not outgoing:
            status = 'disabled_after_provider_rejection'
        elif not expected and not outgoing:
            status = 'no_reasoning_items_provided'
        else:
            status = 'exact_replay' if outgoing == expected else 'replay_mismatch'
        self.trace('reasoning_replay.jsonl', {
            'turn': turn, 'expected_items': len(expected), 'outgoing_items': len(outgoing), 'status': status,
            'expected_item_sha256': [digest(item) for item in expected],
            'outgoing_item_sha256': [digest(item) for item in outgoing]})

    def _trace_cache(self, response, turn, prefix_hash):
        usage = provider_cache_usage(response.get('usage', {}))
        total, read, written = usage['input_tokens'], usage['cached_tokens'], usage['cache_write_tokens']
        valid, cache = usage['cache_reporting_valid'], self.cache
        if valid:
            cache['reported'] += 1
            cache['hits'] += read > 0
            cache['input'] += total
            cache['read'] += read
            if type(written) is int and written >= 0:
                cache['written'] += written
        self.trace('cache.jsonl', {
            'turn': turn, 'stable_prefix_sha256': prefix_hash, **usage,
            'server_diagnostics': response.get('prompt_cache_diagnostics'),
            'request_has_cache_hit': read > 0 if valid else None,
            'cached_input_fraction': read / total if valid and total else None,
            'cumulative_request_hit_rate': cache['hits'] / cache['reported'] if cache['reported'] else None,
            'cumulative_cached_input_fraction': cache['read'] / cache['input'] if cache['input'] else None})

    def _spent(self, reserve):
        if self.poisoned:
            return True
        if self.max_requests is not None and self.request_count >= self.max_requests:
            return True
        committed = self.reported_tokens + self.pending_tokens + reserve
        return self.max_total_tokens is not None and committed > self.max_total_tokens

    def _timeout(self):
        timeout = self.request_timeout_s
        source = None if timeout is None else 'request'
        if self.max_seconds is None:
            return timeout, source, None
        remaining = self.max_seconds - self.elapsed()
        if remaining <= 0:
            raise RunBudgetExceeded(DEADLINE_REACHED + '; no request sent')
        if timeout is None or remaining <= timeout:
            return remaining, 'run', remaining
        return timeout, source, remaining

    def _redact(self, text):
        return text.replace(self.api_key, '[REDACTED]')

    def _attempt(self, payload, logical_id, attempt):
        self.budget().check_available()
        payload = dict(payload)
        if streaming():
            payload['stream'] = True
        metrics = self.measure(payload)
        reserve = metrics['reserved_tokens']
        if self._expired():
            raise RunBudgetExceeded(DEADLINE_REACHED + '; no request sent')
        if self._spent(reserve):
            raise BudgetExceeded('Run-scoped smoke budget exhausted; no request sent')
        mode = os.environ.get('ASTRA_NETWORK_MODE', 'system')
        if mode not in NETWORK_MODES:
            self.poisoned = True
            raise ValueError('ASTRA_NETWORK_MODE must be system or direct')
        wire = to_wire_payload(payload, self.api_mode)
        body = json.dumps(wire).encode()
        client_id = str(uuid4())
        request = urllib.request.Request(self.endpoint, data=body, headers={
            'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json',
            'X-Client-Request-Id': client_id})
        timeout, deadline_source, remaining = self._timeout()
        model_attempt = self.budget().reserve(REQUEST_KIND)
        self.request_count += 1
        self.reserved_tokens += reserve
        self.pending_tokens += reserve
        self.trace('budget.jsonl', {'event': 'reserved', 'request': self.request_count, **metrics,
                                    'cumulative_reserved_tokens': self.reserved_tokens})
        began = time.monotonic()
        context = {
            'request': self.request_count, 'request_kind': REQUEST_KIND,
            'model_attempt': model_attempt, 'client_request_id': client_id,
            'network_mode': mode, 'http_backend': 'python',
            'api_mode': self.api_mode, 'stream': wire.get('stream', False),
            'started_at': datetime.now(timezone.utc).isoformat(),
            'timeout_s': timeout, 'request_bytes': len(body),
            'deadline_source': deadline_source, 'run_remaining_s': remaining,
            'payload_sha256': hashlib.sha256(body).hexdigest(),
            'logical_request_id': logical_id,
            'transport_attempt': attempt, 'retry': attempt > 1,
        }
        self.trace('requests.jsonl', {**context, 'payload': payload})

        def record(event, **fields):
            self.trace('http_transport.jsonl', {**context, 'event': event,
                                                'elapsed_s': time.monotonic() - began, **fields})

        try:
            deadline = None if timeout is None else began + timeout
            response = self._exchange(request, mode, timeout, deadline, wire.get('stream') is True, record)
        except urllib.error.HTTPError as error:
            raise self._http_failure(error, payload, client_id, record) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            raise self._connection_failure(error, client_id, record) from None
        except (ValueError, TypeError, AttributeError) as error:
            raise self._response_failure(error, client_id, record) from None
        return self._settle(response, reserve)

    def _exchange(self, request, mode, timeout, deadline, expects_stream, record):
        if mode == 'direct':
            open_url = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        else:
            open_url = urllib.request.urlopen
        with open_url(request, timeout=timeout) as reply:
            headers = getattr(reply, 'headers', {})
            record('headers', http_status=getattr(reply, 'status', None),
                   content_type=headers.get('Content-Type'),
                   content_encoding=headers.get('Content-Encoding'),
                   response_headers={name: headers.get(name) for name in TRACED_HEADERS
                                     if headers.get(name)})
            content_type = headers.get('Content-Type', '')
            if 'text/event-stream' in content_type or not content_type and expects_stream:
                response = read_sse_response(reply, record, deadline)
            else:
                response = validate_json_response(from_wire_response(json.load(reply), self.api_mode))
        record('completed', response_id=response.get('id'), response_status=response.get('status'))
        return response

    def _stop_if_expired(self, record):
        if not self._expired():
            return
        fields = dict(error_code='run_budget_exhausted', retryable=False,
                      max_seconds=self.max_seconds, active_elapsed_s=self.elapsed())
        record('run_budget_exhausted', **fields)
        self.trace('budget.jsonl', dict(event='run_budget_exhausted', request=self.request_count, **fields))
        raise RunBudgetExceeded(DEADLINE_REACHED + ' during request; no action dispatched') from None

    def _http_failure(self, error, payload, client_id, record):
        self.poisoned = True
        try:
            body = self._redact(error.read(2048).decode('utf-8', errors='replace'))
        except Exception:
            body = '[error body unavailable]'
        provider_code, provider_param = http_error_details(body)
        retryable = error.code in RETRYABLE_STATUS or 500 <= error.code <= 599
        max_attempts = 2 if error.code == 409 else None
        if is_deterministic({'code': provider_code}):
            retryable = False
        recovery = rejected_field(payload, provider_code, provider_param) if error.code == 400 else None
        if recovery is not None:
            retryable, max_attempts = True, 2
        retry_after_s = retry_after_seconds(error.headers)
        record('http_error', exception_type=type(error).__name__, http_status=error.code,
               error_body=body,
               request_id=error.headers.get('x-request-id') if error.headers else None,
               provider_error_code=provider_code, provider_error_param=provider_param,
               retryable=retryable, recovery=recovery, retry_after_s=retry_after_s)
        self.trace('budget.jsonl', {'event': 'attempt_failed', 'type': type(error).__name__,
                                    'retryable': retryable, 'recovery': recovery,
                                    'max_attempts': max_attempts})
        code = safe_error_code(f'HTTP {error.code} {body}')
        disposition = recovery or ('retryable' if retryable else 'no retry')
        return RequestTransportError(
            f'HTTP {error.code}; error_code={code}; client_request_id={client_id}; {disposition}',
            retryable=retryable, error_code=code, recovery=recovery,
            retry_after_s=retry_after_s, max_attempts=max_attempts)

    def _connection_failure(self, error, client_id, record):
        self.poisoned = True
        self._stop_if_expired(record)
        code = 'timeout' if timed_out(error) else 'connection_error'
        record('request_timeout' if code == 'timeout' else 'connection_error',
               exception_type=type(error).__name__,
               exception_message=self._redact(str(error))[:1000],
               traceback=self._redact(traceback.format_exc())[-6000:],
               outcome='unknown; request may have reached provider')
        self.trace('budget.jsonl', {'event': 'attempt_failed', 'type': type(error).__name__,
                                    'retryable': True, 'error_code': code})
        return RequestTransportError(
            f'{type(error).__name__}; error_code={code}; '
            f'client_request_id={client_id}; outcome unknown',
            retryable=True, error_code=code)

    def _response_failure(self, error, client_id, record):
        self.poisoned = True
        self._stop_if_expired(record)
        retryable = bool(getattr(error, 'retryable', False))
        record('invalid_response', exception_type=type(error).__name__,
               exception_message=self._redact(str(error))[:2500], retryable=retryable)
        code = safe_error_code(error)
        recovery = getattr(error, 'recovery', None)
        max_attempts = getattr(error, 'max_attempts', None)
        self.trace('budget.jsonl', {'event': 'attempt_failed', 'type': type(error).__name__,
                                    'retryable': retryable, 'error_code': code,
                                    'recovery': recovery, 'max_attempts': max_attempts})
        disposition = 'retryable' if retryable else 'no retry'
        return RequestTransportError(
            f'Invalid API response; error_code={code}; client_request_id={client_id}; {disposition}',
            retryable=retryable, error_code=code, recovery=recovery, max_attempts=max_attempts)

    def _settle(self, response, reserve):
        try:
            usage = response.get('usage') or {}
            counts = [usage.get('input_tokens'), usage.get('output_tokens')]
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError('Missing/invalid provider usage')
            self.reported_tokens += sum(counts)
            self.pending_tokens -= reserve
            self.trace('budget.jsonl', {
                'event': 'response', 'request': self.request_count, 'usage': usage,
                'cumulative_reported_tokens': self.reported_tokens,
                'pending_reserved_tokens': self.pending_tokens,
                'released_reservation_tokens': reserve})
            over = self.max_total_tokens is not None and self.reported_tokens > self.max_total_tokens
            if response.get('error') or over:
                raise BudgetExceeded('Provider error or reported usage exceeds smoke budget')
            return response
        except Exception as error:
            self.poisoned = True
            self.trace('budget.jsonl', {'event': 'stopped', 'type': type(error).__name__,
                                        'classification': 'usage_validation_or_budget',
                                        'retry': False})
            raise BudgetExceeded(
                'Usage validation/budget check failed; no action dispatched') from None
