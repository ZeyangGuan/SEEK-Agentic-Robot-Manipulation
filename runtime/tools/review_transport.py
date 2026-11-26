import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from seek.policy import transport_retry
from seek.policy.model_request_budget import ModelRequestBudget
from seek.policy.bounded_policy import RequestTransportError
from seek.policy.provider_response import (read_sse_response, TruncatedResponseError, ProviderResponseError,
                                           retry_after_seconds)


QUOTA_MARKERS = ('insufficient_quota', 'billing_hard_limit', 'quota_exceeded', 'insufficient_balance',
                 'insufficient balance', 'credit balance is too low', 'usage_limit_reached')


def is_quota_error(text):
    text = text.lower()
    return any(marker in text for marker in QUOTA_MARKERS)


class StreamLines:
    def __init__(self,response):self.lines=iter(response.iter_lines())
    def readline(self):
        line=next(self.lines,None)
        return b'' if line is None else line+b'\n'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def send_review(endpoint, token, payload, receipt, folder):
    import requests
    folder = Path(folder)
    budget_dir = os.environ.get('SEEK_REVIEW_BUDGET_DIR')
    budget = ModelRequestBudget.resume(budget_dir) if budget_dir else None
    wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    receipt.update(logical_requests=1, requests=0, retry_requests=0, failed_requests=0,
                   completed=False, completed_requests=0, transport_recovery_enabled=transport_retry.enabled(),
                   logical_request_id=str(uuid4()), payload_sha256=hashlib.sha256(wire).hexdigest())

    def persist():
        save(folder/'receipt.json', receipt)

    def log(event, **fields):
        row = dict(event=event, unix=time.time(), **fields)
        with (folder/'transport.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')

    def before(logical_id, attempt):
        receipt['logical_request_id'] = logical_id

    def send():
        if budget: budget.check_available()
        if budget: budget.reserve('review')
        receipt['requests'] += 1
        receipt['retry_requests'] = receipt['requests']-1
        fields = dict(logical_request_id=receipt['logical_request_id'], attempt=receipt['requests'],
                      payload_sha256=receipt['payload_sha256'], robot_action_replayed=False)
        log('request_started', **fields)
        persist()
        started = time.monotonic()
        try:
            with requests.post(endpoint, headers={'Authorization': 'Bearer '+token,
                               'Content-Type': 'application/json'}, data=wire,
                               stream=True, timeout=(30, 600)) as response:
                receipt['http_status'] = response.status_code
                if response.status_code != 200:
                    transient=(response.status_code in (408,409,429) or 500 <= response.status_code <= 599)
                    raise RequestTransportError('Review HTTP '+str(response.status_code),
                        retryable=transient and not is_quota_error(response.text[:2048]),
                        error_code='http_'+str(response.status_code),
                        retry_after_s=retry_after_seconds(getattr(response,'headers',None)))
                def record_event(event):
                    if event.get('type') in ('response.output_item.done','response.completed','response.failed','response.incomplete','error'):
                        with (folder/'stream_events.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(attempt=receipt['requests'],event=event),ensure_ascii=False)+'\n')
                try:
                    completed=read_sse_response(StreamLines(response),log,None,on_event=record_event)
                except TruncatedResponseError as error:
                    raise transport_retry.TransportInterruption('provider_stream_truncated') from error
                if not isinstance(completed.get('output'),list):raise ValueError('Invalid completed review output')
        except Exception as error:
            receipt['failed_requests'] += 1
            receipt['last_error_type'] = type(error).__name__
            receipt['last_error_code'] = getattr(error, 'error_code', type(error).__name__)
            log('request_failed', **fields, error_type=type(error).__name__,
                error_code=receipt['last_error_code'], wall_s=time.monotonic()-started,
                usage_unknown=True)
            persist()
            raise
        receipt.update(completed=True, completed_requests=1, usage=completed.get('usage'))
        log('request_completed', **fields, wall_s=time.monotonic()-started,
            usage=completed.get('usage'))
        persist()
        return completed

    persist()
    def retryable(error):
        if isinstance(error,(RequestTransportError,ProviderResponseError)):
            if not error.retryable or error.recovery is not None:return False
            if budget:budget.check_available()
            return True
        if isinstance(error, (transport_retry.TransportInterruption, requests.Timeout,
                requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
            if budget: budget.check_available()
            return True
        return False
    return transport_retry.request(send, retryable, log, before=before)
