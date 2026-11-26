import hashlib
import http.client
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4

from . import transport_retry
from .eval_failure import safe_error_code
from .model_policy import Model as BaseModel
from .model_request_budget import ModelRequestBudget, request_budget
from .provider_compat import RESPONSES, from_wire_response, to_wire_payload
from .provider_response import (
    COMPARISON_PARAMETER, http_error_details, is_deterministic, read_sse_response,
    retry_after_seconds, validate_json_response,
)


TRACED_HEADERS = ('x-request-id', 'x-oneapi-request-id', 'cf-ray', 'server', 'date')
RETRYABLE_STATUS = (408, 409, 429)


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


def _timed_out(error):
    if isinstance(error, TimeoutError):
        return True
    return (isinstance(error, urllib.error.URLError)
            and isinstance(getattr(error, 'reason', None), TimeoutError))


def _http_recovery(payload, code, param):
    recovery = None
    if code == 'invalid_encrypted_content' and any(
            isinstance(item, dict) and item.get('type') == 'reasoning'
            for item in payload.get('input', [])):
        recovery = 'drop_reasoning_items'
    options = payload.get('prompt_cache_options')
    if ((code == 'invalid_comparison_response_id' or param == COMPARISON_PARAMETER)
            and isinstance(options, dict) and options.get('comparison_response_id')):
        recovery = 'drop_comparison_response_id'
    return recovery


class Model(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.request_count = 0
        self.reported_tokens = 0
        self.reserved_tokens = 0
        self.pending_reserved_tokens = 0
        self.started = time.monotonic()
        self.budget_poisoned = False

    def reset(self):
        super().reset()
        limit = getattr(getattr(self, 'dialogue_config', None), 'max_model_requests', 150)
        self.model_request_budget = ModelRequestBudget(limit, self.episode_dir)

    def _active_elapsed(self):
        return transport_retry.active_elapsed(self.started)

    def _time_limit_reached(self):
        return self.max_seconds is not None and self._active_elapsed() >= self.max_seconds

    def _run_budget_spent(self, reserve):
        if self.budget_poisoned:
            return True
        if self.max_requests is not None and self.request_count >= self.max_requests:
            return True
        committed = self.reported_tokens + self.pending_reserved_tokens + reserve
        return self.max_total_tokens is not None and committed > self.max_total_tokens

    def _attempt_timeout(self):
        timeout = getattr(self, 'request_timeout_s', 120)
        source = 'request' if timeout is not None else None
        run_remaining = None
        if self.max_seconds is not None:
            run_remaining = self.max_seconds - self._active_elapsed()
            if run_remaining <= 0:
                raise RunBudgetExceeded(
                    'run_budget_exhausted: active run deadline reached; no request sent')
            if timeout is None or run_remaining <= timeout:
                timeout, source = run_remaining, 'run'
        return timeout, source, run_remaining

    def _redact(self, text):
        return text.replace(self.api_key, '[REDACTED]')

    def _request_once(self, payload):
        request_budget(self).check_available()
        payload = dict(payload)
        if os.environ.get('ASTRA_STREAM', '0') == '1':
            payload['stream'] = True
        metrics = self.payload_metrics(payload)
        reserve = metrics['reserved_tokens']
        if self._time_limit_reached():
            raise RunBudgetExceeded(
                'run_budget_exhausted: active run deadline reached; no request sent')
        if self._run_budget_spent(reserve):
            raise BudgetExceeded('Run-scoped smoke budget exhausted; no request sent')
        mode = os.environ.get('ASTRA_NETWORK_MODE', 'system')
        if mode not in ('system', 'direct'):
            self.budget_poisoned = True
            raise ValueError('ASTRA_NETWORK_MODE must be system or direct')
        api_mode = getattr(self, 'api_mode', RESPONSES)
        wire_payload = to_wire_payload(payload, api_mode)
        body = json.dumps(wire_payload).encode()
        client_id = str(uuid4())
        request = urllib.request.Request(self.endpoint, data=body, headers={
            'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json',
            'X-Client-Request-Id': client_id})
        timeout, deadline_source, run_remaining = self._attempt_timeout()
        request_kind = getattr(self, 'model_request_kind', 'action')
        model_attempt = request_budget(self).reserve(request_kind)
        self.request_count += 1
        self.reserved_tokens += reserve
        self.pending_reserved_tokens += reserve
        self._log('budget.jsonl', {'event': 'reserved', 'request': self.request_count, **metrics,
                                  'cumulative_reserved_tokens': self.reserved_tokens})
        started = time.monotonic()
        transport_attempt = getattr(self, 'transport_attempt', 1)
        diagnostic = {
            'request': self.request_count, 'request_kind': request_kind,
            'model_attempt': model_attempt, 'client_request_id': client_id,
            'network_mode': mode, 'http_backend': 'python',
            'api_mode': api_mode, 'stream': wire_payload.get('stream', False),
            'started_at': datetime.now(timezone.utc).isoformat(),
            'timeout_s': timeout, 'request_bytes': len(body),
            'deadline_source': deadline_source, 'run_remaining_s': run_remaining,
            'payload_sha256': hashlib.sha256(body).hexdigest(),
            'logical_request_id': getattr(self, 'transport_logical_id', None),
            'transport_attempt': transport_attempt, 'retry': transport_attempt > 1,
        }
        self._log('requests.jsonl', {**diagnostic, 'payload': payload})

        def record(event, **fields):
            self._log('http_transport.jsonl', {**diagnostic, 'event': event,
                                              'elapsed_s': time.monotonic() - started, **fields})

        try:
            expects_stream = wire_payload.get('stream') is True
            deadline = None if timeout is None else started + timeout
            response = self._exchange(request, mode, api_mode, timeout, deadline, expects_stream, record)
        except urllib.error.HTTPError as error:
            raise self._http_failure(error, payload, client_id, record) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            raise self._connection_failure(error, client_id, record) from None
        except (ValueError, TypeError, AttributeError) as error:
            raise self._response_failure(error, client_id, record) from None
        return self._settle_usage(response, reserve)

    def _exchange(self, request, mode, api_mode, timeout, deadline, expects_stream, record):
        if mode == 'direct':
            send = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        else:
            send = urllib.request.urlopen
        with send(request, timeout=timeout) as reply:
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
                response = validate_json_response(from_wire_response(json.load(reply), api_mode))
        record('completed', response_id=response.get('id'), response_status=response.get('status'))
        return response

    def _reject_expired_run(self, record):
        if not self._time_limit_reached():
            return
        fields = dict(error_code='run_budget_exhausted', retryable=False,
                      max_seconds=self.max_seconds, active_elapsed_s=self._active_elapsed())
        record('run_budget_exhausted', **fields)
        self._log('budget.jsonl', dict(event='run_budget_exhausted',
                                      request=self.request_count, **fields))
        raise RunBudgetExceeded(
            'run_budget_exhausted: active run deadline reached during request; '
            'no action dispatched') from None

    def _http_failure(self, error, payload, client_id, record):
        self.budget_poisoned = True
        try:
            body = self._redact(error.read(2048).decode('utf-8', errors='replace'))
        except Exception:
            body = '[error body unavailable]'
        provider_code, provider_param = http_error_details(body)
        retryable = error.code in RETRYABLE_STATUS or 500 <= error.code <= 599
        max_attempts = 2 if error.code == 409 else None
        if is_deterministic({'code': provider_code}):
            retryable = False
        recovery = _http_recovery(payload, provider_code, provider_param) if error.code == 400 else None
        if recovery is not None:
            retryable, max_attempts = True, 2
        retry_after_s = retry_after_seconds(error.headers)
        record('http_error', exception_type=type(error).__name__, http_status=error.code,
               error_body=body,
               request_id=error.headers.get('x-request-id') if error.headers else None,
               provider_error_code=provider_code, provider_error_param=provider_param,
               retryable=retryable, recovery=recovery, retry_after_s=retry_after_s)
        self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(error).__name__,
                                  'retryable': retryable, 'recovery': recovery,
                                  'max_attempts': max_attempts})
        code = safe_error_code(f'HTTP {error.code} {body}')
        disposition = recovery or ('retryable' if retryable else 'no retry')
        return RequestTransportError(
            f'HTTP {error.code}; error_code={code}; client_request_id={client_id}; {disposition}',
            retryable=retryable, error_code=code, recovery=recovery,
            retry_after_s=retry_after_s, max_attempts=max_attempts)

    def _connection_failure(self, error, client_id, record):
        self.budget_poisoned = True
        self._reject_expired_run(record)
        timed_out = _timed_out(error)
        code = 'timeout' if timed_out else 'connection_error'
        record('request_timeout' if timed_out else 'connection_error',
               exception_type=type(error).__name__,
               exception_message=self._redact(str(error))[:1000],
               traceback=self._redact(traceback.format_exc())[-6000:],
               outcome='unknown; request may have reached provider')
        self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(error).__name__,
                                  'retryable': True, 'error_code': code})
        return RequestTransportError(
            f'{type(error).__name__}; error_code={code}; '
            f'client_request_id={client_id}; outcome unknown',
            retryable=True, error_code=code)

    def _response_failure(self, error, client_id, record):
        self.budget_poisoned = True
        self._reject_expired_run(record)
        retryable = bool(getattr(error, 'retryable', False))
        record('invalid_response', exception_type=type(error).__name__,
               exception_message=self._redact(str(error))[:2500], retryable=retryable)
        code = safe_error_code(error)
        recovery = getattr(error, 'recovery', None)
        max_attempts = getattr(error, 'max_attempts', None)
        self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(error).__name__,
                                  'retryable': retryable, 'error_code': code,
                                  'recovery': recovery, 'max_attempts': max_attempts})
        disposition = 'retryable' if retryable else 'no retry'
        return RequestTransportError(
            f'Invalid API response; error_code={code}; client_request_id={client_id}; {disposition}',
            retryable=retryable, error_code=code, recovery=recovery, max_attempts=max_attempts)

    def _settle_usage(self, response, reserve):
        try:
            usage = response.get('usage') or {}
            counts = [usage.get('input_tokens'), usage.get('output_tokens')]
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError('Missing/invalid provider usage')
            self.reported_tokens += sum(counts)
            self.pending_reserved_tokens -= reserve
            self._log('budget.jsonl', {
                'event': 'response', 'request': self.request_count, 'usage': usage,
                'cumulative_reported_tokens': self.reported_tokens,
                'pending_reserved_tokens': self.pending_reserved_tokens,
                'released_reservation_tokens': reserve})
            if response.get('error') or (self.max_total_tokens is not None
                                         and self.reported_tokens > self.max_total_tokens):
                raise BudgetExceeded('Provider error or reported usage exceeds smoke budget')
            return response
        except Exception as error:
            self.budget_poisoned = True
            self._log('budget.jsonl', {'event': 'stopped', 'type': type(error).__name__,
                                      'classification': 'usage_validation_or_budget',
                                      'retry': False})
            raise BudgetExceeded(
                'Usage validation/budget check failed; no action dispatched') from None
