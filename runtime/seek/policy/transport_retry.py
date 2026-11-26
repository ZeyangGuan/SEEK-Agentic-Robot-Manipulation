import math
import os
import time
from uuid import uuid4


MAX_BACKOFF_S = 60
_outages = []


class TransportInterruption(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.error_code = code


def enabled():
    return os.environ.get('SEEK_RESILIENT_TRANSPORT') == '1'


def active_elapsed(started):
    now = time.monotonic()
    if not enabled():
        return now - started
    waited = sum(max(0.0, min(end, now) - max(begin, started)) for begin, end in _outages)
    return max(0.0, now - started - waited)


def _backoff(attempt, error):
    delay = min(2 ** min(attempt, 6), MAX_BACKOFF_S)
    hint = getattr(error, 'retry_after_s', None)
    if isinstance(hint, (int, float)) and math.isfinite(hint):
        delay = max(delay, hint)
    return delay


def _wait(delay, started):
    try:
        time.sleep(delay)
    finally:
        _outages.append((started, time.monotonic()))


def request(send, retryable, log, before=None, max_attempts=None):
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 1):
        raise ValueError("max_attempts must be a positive integer or null")
    if not enabled():
        return send()
    logical_id = str(uuid4())
    failures = {}
    attempt = 0
    while True:
        attempt += 1
        if before is not None:
            before(logical_id, attempt)
        started = time.monotonic()
        try:
            result = send()
        except Exception as error:
            if max_attempts is not None and attempt >= max_attempts:
                raise
            if not retryable(error):
                raise
            code = getattr(error, 'error_code', None) or type(error).__name__
            failures[code] = failures.get(code, 0) + 1
            cap = getattr(error, 'max_attempts', None)
            if cap is not None and failures[code] >= cap:
                raise
            delay = _backoff(attempt, error)
            log('retry_wait', logical_request_id=logical_id, failed_attempt=attempt,
                error_code=getattr(error, 'error_code', type(error).__name__),
                failed_request_seconds=time.monotonic() - started, retry_delay_seconds=delay,
                robot_action_replayed=False, possible_duplicate_inference_cost=True)
            _wait(delay, started)
            continue
        if attempt > 1:
            log('recovered', logical_request_id=logical_id, attempts=attempt,
                robot_action_replayed=False)
        return result
