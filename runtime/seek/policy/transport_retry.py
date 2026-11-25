import os
import math
import time
from uuid import uuid4


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
    outage = sum(max(0.0, min(end, now) - max(begin, started)) for begin, end in _outages)
    return max(0.0, now - started - outage)


def request(send, retryable, log, before=None, max_attempts=None):
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts<1):
        raise ValueError("max_attempts must be a positive integer or null")
    if not enabled():
        return send()
    logical_id = str(uuid4())
    attempt = 0
    failures_by_code = {}
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
            failures_by_code[code] = failures_by_code.get(code, 0) + 1
            maximum = getattr(error, 'max_attempts', None)

            if maximum is not None and failures_by_code[code] >= maximum:
                raise
            delay = min(2 ** min(attempt, 6), 60)
            retry_after = getattr(error, 'retry_after_s', None)
            if isinstance(retry_after, (int, float)) and math.isfinite(retry_after):
                delay = max(delay, retry_after)
            log('retry_wait', logical_request_id=logical_id, failed_attempt=attempt,
                error_code=getattr(error, 'error_code', type(error).__name__),
                failed_request_seconds=time.monotonic() - started, retry_delay_seconds=delay,
                robot_action_replayed=False, possible_duplicate_inference_cost=True)
            try:
                time.sleep(delay)
            finally:
                _outages.append((started, time.monotonic()))
            continue
        if attempt > 1:
            log('recovered', logical_request_id=logical_id, attempts=attempt,
                robot_action_replayed=False)
        return result
