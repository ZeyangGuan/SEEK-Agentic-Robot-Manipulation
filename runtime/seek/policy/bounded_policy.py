import json
import hashlib
import http.client
import math
import os
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4

from .model_policy import Model as BaseModel
from .model_request_budget import ModelRequestBudget, request_budget
from . import transport_retry
from .provider_compat import (
    RESPONSES, from_wire_response, to_wire_payload,
)


MAX_RETRY_AFTER_S = 60.0


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


def _provider_error_is_transient(details):
    if not isinstance(details, dict):
        return False
    fields = ' '.join(str(details.get(key, '')).lower()
                      for key in ('type', 'code', 'reason'))
    return any(marker in fields for marker in (
        'rate_limit', 'too_many_requests', 'server_error', 'overloaded',
        'temporarily_unavailable', 'service_unavailable', 'provider_capacity',
        'timeout', 'timed_out',
    ))


def _provider_error_is_deterministic(details):
    if not isinstance(details, dict):
        return False
    fields = ' '.join(str(details.get(key, '')).lower()
                      for key in ('type', 'code', 'reason'))
    return any(marker in fields for marker in (
        'content_filter', 'content_policy', 'invalid_request',
        'invalid_encrypted_content', 'max_output_tokens', 'max_tokens', 'unsupported_value',
        'insufficient_quota', 'billing_hard_limit',
    ))


def _provider_error_code(details, fallback):
    if not isinstance(details, dict):
        return fallback
    value = details.get('code') or details.get('type') or details.get('reason')
    return str(value)[:128] if value else fallback


def _http_error_details(error_text):
    try:
        parsed = json.loads(error_text)
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


def _retry_after_seconds(headers):
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


def _is_request_timeout(exc):
    if isinstance(exc, TimeoutError):
        return True
    return (isinstance(exc, urllib.error.URLError)
            and isinstance(getattr(exc, 'reason', None), TimeoutError))


def _raise_provider_terminal_failure(kind, event, response=None, source='SSE'):
    if response is None:
        nested = event.get('response') if isinstance(event, dict) else None
        response = nested if isinstance(nested, dict) else {}
    details = (event.get('error') or response.get('error')
               or response.get('incomplete_details')
               or event.get('incomplete_details') or event)
    recovery = None
    max_attempts = None
    if (kind == 'response.incomplete' and isinstance(details, dict)
            and details.get('reason') == 'content_filter'):
        retryable = True
        max_attempts = 3
    elif _provider_error_is_transient(details):
        retryable = True
    elif _provider_error_is_deterministic(details):
        retryable = False
    else:
        retryable = True
        max_attempts = 2
    error_code = _provider_error_code(details, kind.replace('.', '_'))
    if error_code == 'invalid_encrypted_content':
        retryable = True
        recovery = 'drop_reasoning_items'
        max_attempts = 2
    elif (isinstance(details, dict) and (
            error_code == 'invalid_comparison_response_id'
            or details.get('param') == 'prompt_cache_options.comparison_response_id')):
        retryable = True
        recovery = 'drop_comparison_response_id'
        max_attempts = 2
    if isinstance(details, dict):
        details = {k: details[k] for k in
                   ('type', 'code', 'reason', 'message', 'param') if k in details}
    error_type = {
        'response.incomplete': IncompleteResponseError,
        'response.failed': FailedResponseError,
    }[kind]
    raise error_type(f'{source} terminal failure: {kind}; '
                     + json.dumps(details, ensure_ascii=False)[:2000],
                     retryable=retryable, error_code=error_code,
                     recovery=recovery, max_attempts=max_attempts)


def _validate_json_response(response):
    if not isinstance(response, dict):
        raise TypeError('Responses JSON body must be an object')
    status = response.get('status')
    if status in ('failed', 'incomplete'):
        _raise_provider_terminal_failure(
            f'response.{status}', response, response=response, source='JSON')
    nested = response.get('response')
    kind = response.get('type')
    if kind in ('response.failed', 'response.incomplete'):
        _raise_provider_terminal_failure(
            kind, response,
            response=nested if isinstance(nested, dict) else {}, source='JSON')
    if isinstance(nested, dict) and nested.get('status') in ('failed', 'incomplete'):
        _raise_provider_terminal_failure(
            f"response.{nested['status']}", response,
            response=nested, source='JSON')
    return response


def read_sse_response(reply, record, deadline, on_event=None):
    data, total = [], 0
    output_items = {}
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('SSE response deadline exceeded')
        line = reply.readline()
        total += len(line)
        if not line:
            raise TruncatedResponseError(
                'SSE ended without response.completed',
                error_code='provider_stream_truncated', max_attempts=2,
            )
        try:
            line = line.decode('utf-8').rstrip('\r\n')
        except UnicodeDecodeError:
            raise TruncatedResponseError('Invalid UTF-8 SSE frame', error_code='provider_stream_truncated') from None
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            raw, data = '\n'.join(data), []

            if not raw.strip():
                continue
            if raw == '[DONE]':
                raise TruncatedResponseError(
                    'SSE DONE without response.completed',
                    error_code='provider_stream_truncated', max_attempts=2,
                )
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                raise TruncatedResponseError('Invalid JSON SSE frame', error_code='provider_stream_truncated') from None
            if not isinstance(event, dict):
                raise ValueError('SSE event must be an object')
            if on_event is not None:
                on_event(event)
            kind = event.get('type')
            if kind == 'response.output_item.done':
                output_items[event['output_index']] = event['item']
            if kind in ('response.created', 'response.completed', 'response.failed', 'response.incomplete', 'error'):
                record('sse_event', event_type=kind, response_id=event.get('response', {}).get('id'))
            if kind == 'response.completed':
                result = event['response']
                if result.get('status') != 'completed':
                    raise ValueError('Contradictory SSE completion status')
                if not result.get('output') and output_items:
                    result['output'] = [output_items[i] for i in sorted(output_items)]
                return result
            if kind in ('response.failed', 'response.incomplete', 'error'):
                response = event.get('response', {})
                if kind in ('response.failed', 'response.incomplete'):
                    _raise_provider_terminal_failure(kind, event, response=response)
                details = (event.get('error') or response.get('error')
                           or response.get('incomplete_details') or event)
                retryable = _provider_error_is_transient(details)
                recovery = None
                max_attempts = None
                error_code = _provider_error_code(details, kind.replace('.', '_'))
                if error_code == 'invalid_encrypted_content':
                    retryable = True
                    recovery = 'drop_reasoning_items'
                    max_attempts = 2
                elif (isinstance(details, dict) and (
                        error_code == 'invalid_comparison_response_id'
                        or details.get('param') == 'prompt_cache_options.comparison_response_id')):
                    retryable = True
                    recovery = 'drop_comparison_response_id'
                    max_attempts = 2
                if isinstance(details, dict):
                    details = {k: details[k] for k in ('type', 'code', 'reason', 'message', 'param') if k in details}
                error_type = TransientResponseError if retryable else ProviderResponseError
                raise error_type(f'SSE terminal failure: {kind}; '
                                 + json.dumps(details, ensure_ascii=False)[:2000],
                                 retryable=retryable, error_code=error_code,
                                 recovery=recovery, max_attempts=max_attempts)


class Model(BaseModel):
    def reset(self):
        super().reset()
        self.model_request_budget = ModelRequestBudget(
            getattr(getattr(self, 'dialogue_config', None), 'max_model_requests', 150), self.episode_dir)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.request_count = 0
        self.reported_tokens = 0
        self.reserved_tokens = 0
        self.pending_reserved_tokens = 0
        self.started = time.monotonic()
        self.budget_poisoned = False

    def _time_limit_reached(self):
        return self.max_seconds is not None and self._active_elapsed() >= self.max_seconds

    def _active_elapsed(self):
        return transport_retry.active_elapsed(self.started)

    def _request(self, payload):
        def before(logical_id, attempt):
            if not hasattr(self, 'dialogue_config'):
                self.transport_logical_id = logical_id
                self.transport_attempt = attempt
            if attempt > 1:
                self.budget_poisoned = False
        def can_retry(error):
            if (isinstance(error, RequestTransportError) and error.retryable
                    and error.recovery is None):
                request_budget(self).check_available()
                return True
            return False
        return transport_retry.request(
            lambda: self._request_once(payload), can_retry,
            lambda event, **fields: self._log('transport_retries.jsonl', dict(event=event, **fields)),

            before=before, max_attempts=1 if hasattr(self, 'dialogue_config') else None)

    def _request_once(self, payload):
        request_budget(self).check_available()
        payload = dict(payload)

        api_mode = getattr(self, 'api_mode', RESPONSES)
        if os.environ.get('ASTRA_STREAM', '0') == '1':
            payload['stream'] = True
        metrics = self.payload_metrics(payload)
        reserve = metrics['reserved_tokens']
        if self._time_limit_reached():
            raise RunBudgetExceeded('run_budget_exhausted: active run deadline reached; no request sent')
        if (self.budget_poisoned or (self.max_requests is not None and self.request_count >= self.max_requests) or
            (self.max_total_tokens is not None and self.reported_tokens + self.pending_reserved_tokens + reserve > self.max_total_tokens)):
            raise BudgetExceeded('Run-scoped smoke budget exhausted; no request sent')
        mode = os.environ.get('ASTRA_NETWORK_MODE', 'system')
        if mode not in ('system', 'direct'):
            self.budget_poisoned = True
            raise ValueError('ASTRA_NETWORK_MODE must be system or direct')
        wire_payload = to_wire_payload(payload, api_mode)
        body = json.dumps(wire_payload).encode()
        client_id = str(uuid4())
        authorization = f'Bearer {self.api_key}'
        request = urllib.request.Request(self.endpoint, data=body,
                  headers={'Authorization': authorization, 'Content-Type': 'application/json',
                           'X-Client-Request-Id': client_id})
        remaining = getattr(self, 'request_timeout_s', 120)
        deadline_source = 'request' if remaining is not None else None
        run_remaining = None
        if self.max_seconds is not None:
            run_remaining = self.max_seconds - self._active_elapsed()
            if run_remaining <= 0:
                raise RunBudgetExceeded('run_budget_exhausted: active run deadline reached; no request sent')
            if remaining is None or run_remaining <= remaining:
                remaining, deadline_source = run_remaining, 'run'
        request_kind = getattr(self, 'model_request_kind', 'action')
        model_attempt = request_budget(self).reserve(request_kind)
        self.request_count += 1
        self.reserved_tokens += reserve
        self.pending_reserved_tokens += reserve
        self._log('budget.jsonl', {'event': 'reserved', 'request': self.request_count, **metrics,
                                  'cumulative_reserved_tokens': self.reserved_tokens})
        started = time.monotonic()
        diagnostic = {'request': self.request_count, 'request_kind': request_kind, 'model_attempt': model_attempt, 'client_request_id': client_id,
                      'network_mode': mode, 'http_backend': 'python',
                      'api_mode': api_mode, 'stream': wire_payload.get('stream', False),
                      'started_at': datetime.now(timezone.utc).isoformat(),
                      'timeout_s': remaining, 'request_bytes': len(body),
                      'deadline_source': deadline_source, 'run_remaining_s': run_remaining,
                      'payload_sha256': hashlib.sha256(body).hexdigest(),
                      'logical_request_id': getattr(self, 'transport_logical_id', None),
                      'transport_attempt': getattr(self, 'transport_attempt', 1),
                      'retry': getattr(self, 'transport_attempt', 1) > 1}
        self._log('requests.jsonl', {**diagnostic, 'payload': payload})
        def record(event, **fields):
            self._log('http_transport.jsonl', {**diagnostic, 'event': event,
                      'elapsed_s': time.monotonic()-started, **fields})
        def reject_expired_run():
            if self._time_limit_reached():
                fields = dict(error_code='run_budget_exhausted', retryable=False,
                              max_seconds=self.max_seconds, active_elapsed_s=self._active_elapsed())
                record('run_budget_exhausted', **fields)
                self._log('budget.jsonl', dict(event='run_budget_exhausted',
                                              request=self.request_count, **fields))
                raise RunBudgetExceeded(
                    'run_budget_exhausted: active run deadline reached during request; no action dispatched') from None
        try:
            open_request = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                            if mode == 'direct' else urllib.request.urlopen)
            with open_request(request, timeout=remaining) as reply:
                headers = getattr(reply, 'headers', {})
                record('headers', http_status=getattr(reply, 'status', None),
                       content_type=headers.get('Content-Type'),
                       content_encoding=headers.get('Content-Encoding'),
                       response_headers={k: headers.get(k) for k in
                           ('x-request-id', 'x-oneapi-request-id', 'cf-ray', 'server', 'date')
                           if headers.get(k)})
                content_type = headers.get('Content-Type', '')
                if ('text/event-stream' in content_type
                        or not content_type and wire_payload.get('stream') is True):
                    response = read_sse_response(
                        reply, record, None if remaining is None else started+remaining)
                else:
                    response = _validate_json_response(
                        from_wire_response(json.load(reply), api_mode))
            record('completed', response_id=response.get('id'), response_status=response.get('status'))
        except urllib.error.HTTPError as exc:
            self.budget_poisoned = True

            try:
                error_text = exc.read(2048).decode('utf-8', errors='replace').replace(self.api_key, '[REDACTED]')
            except Exception:
                error_text = '[error body unavailable]'
            provider_code, provider_param = _http_error_details(error_text)
            retryable = exc.code in (408, 409, 429) or 500 <= exc.code <= 599
            max_attempts = 2 if exc.code == 409 else None
            if _provider_error_is_deterministic({'code': provider_code}):
                retryable = False
            recovery = None
            if (exc.code == 400 and provider_code == 'invalid_encrypted_content'
                    and any(isinstance(item, dict) and item.get('type') == 'reasoning'
                            for item in payload.get('input', []))):
                retryable = True
                recovery = 'drop_reasoning_items'
                max_attempts = 2
            options = payload.get('prompt_cache_options')
            comparison_rejected = (
                provider_code == 'invalid_comparison_response_id'
                or provider_param == 'prompt_cache_options.comparison_response_id'
            )
            if (exc.code == 400 and comparison_rejected
                    and isinstance(options, dict)
                    and options.get('comparison_response_id')):
                retryable = True
                recovery = 'drop_comparison_response_id'
                max_attempts = 2
            retry_after_s = _retry_after_seconds(exc.headers)
            record('http_error', exception_type=type(exc).__name__, http_status=exc.code,
                   error_body=error_text, request_id=exc.headers.get('x-request-id') if exc.headers else None,
                   provider_error_code=provider_code, provider_error_param=provider_param,
                   retryable=retryable, recovery=recovery, retry_after_s=retry_after_s)
            self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(exc).__name__,
                                      'retryable': retryable, 'recovery': recovery,
                                      'max_attempts': max_attempts})
            from .eval_failure import safe_error_code
            code = safe_error_code(f'HTTP {exc.code} {error_text}')
            disposition = recovery or ('retryable' if retryable else 'no retry')
            raise RequestTransportError(
                f'HTTP {exc.code}; error_code={code}; client_request_id={client_id}; {disposition}',
                retryable=retryable, error_code=code, recovery=recovery,
                retry_after_s=retry_after_s,
                max_attempts=max_attempts,
            ) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            self.budget_poisoned = True
            reject_expired_run()
            timed_out = _is_request_timeout(exc)
            error_code = 'timeout' if timed_out else 'connection_error'
            record('request_timeout' if timed_out else 'connection_error',
                   exception_type=type(exc).__name__,
                   exception_message=str(exc).replace(self.api_key, '[REDACTED]')[:1000],
                   traceback=traceback.format_exc().replace(self.api_key, '[REDACTED]')[-6000:],
                   outcome='unknown; request may have reached provider')
            self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(exc).__name__,
                                      'retryable': True, 'error_code': error_code})
            raise RequestTransportError(f'{type(exc).__name__}; error_code={error_code}; '
                                        f'client_request_id={client_id}; outcome unknown',
                                        retryable=True, error_code=error_code) from None
        except (ValueError, TypeError, AttributeError) as exc:
            self.budget_poisoned = True
            reject_expired_run()
            retryable = bool(getattr(exc, 'retryable', False))
            record('invalid_response', exception_type=type(exc).__name__,
                   exception_message=str(exc).replace(self.api_key, '[REDACTED]')[:2500],
                   retryable=retryable)
            from .eval_failure import safe_error_code

            code = safe_error_code(exc)
            recovery = getattr(exc, 'recovery', None)
            max_attempts = getattr(exc, 'max_attempts', None)
            self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(exc).__name__,
                                      'retryable': retryable, 'error_code': code,
                                      'recovery': recovery, 'max_attempts': max_attempts})
            disposition = 'retryable' if retryable else 'no retry'
            raise RequestTransportError(
                f'Invalid API response; error_code={code}; client_request_id={client_id}; {disposition}',
                retryable=retryable, error_code=code, recovery=recovery,
                max_attempts=max_attempts,
            ) from None
        try:
            usage = response.get('usage') or {}
            values = [usage.get('input_tokens'), usage.get('output_tokens')]
            if any(type(n) is not int or n < 0 for n in values):
                raise ValueError('Missing/invalid provider usage')
            self.reported_tokens += sum(values)

            self.pending_reserved_tokens -= reserve
            self._log('budget.jsonl', {'event': 'response', 'request': self.request_count,
                      'usage': usage, 'cumulative_reported_tokens': self.reported_tokens,
                      'pending_reserved_tokens': self.pending_reserved_tokens,
                      'released_reservation_tokens': reserve})
            if response.get('error') or (self.max_total_tokens is not None and self.reported_tokens > self.max_total_tokens):
                raise BudgetExceeded('Provider error or reported usage exceeds smoke budget')
            return response
        except Exception as exc:
            self.budget_poisoned = True
            self._log('budget.jsonl', {'event': 'stopped', 'type': type(exc).__name__,
                                      'classification': 'usage_validation_or_budget', 'retry': False})
            raise BudgetExceeded('Usage validation/budget check failed; no action dispatched') from None
