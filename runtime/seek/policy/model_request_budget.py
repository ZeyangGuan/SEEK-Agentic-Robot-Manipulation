import json
from pathlib import Path
from threading import RLock


STATE_FILE = 'model_request_budget.json'
EVENT_FILE = 'model_request_budget.jsonl'
MAX_LIMIT = 150


class RequestBudgetExceeded(RuntimeError):
    error_code = 'request_budget_exhausted'
    retryable = False


def _count(value):
    return type(value) is int and value >= 0


class ModelRequestBudget:
    def __init__(self, limit=MAX_LIMIT, folder=None):
        if limit is not None and (type(limit) is not int or not 1 <= limit <= MAX_LIMIT):
            raise ValueError('max_model_requests must be null or an integer in [1,150]')
        self.limit = limit
        self.used = 0
        self.by_kind = {}
        self.folder = Path(folder) if folder is not None else None
        self.lock = RLock()
        self.blocked = False

    @classmethod
    def resume(cls, folder):
        path = Path(folder) / STATE_FILE
        if not path.exists():
            return cls(folder=folder)
        saved = json.loads(path.read_text())
        budget = cls(limit=saved['limit'], folder=folder)
        used = saved['used']
        if not _count(used) or (budget.limit is not None and used > budget.limit):
            raise ValueError('Invalid persisted episode allowance')
        by_kind = saved['by_kind']
        if not all(_count(value) for value in by_kind.values()) or sum(by_kind.values()) != used:
            raise ValueError('Persisted request accounting mismatch')
        budget.used, budget.by_kind = used, by_kind
        return budget

    def exhausted(self):
        return self.limit is not None and self.used >= self.limit

    def _state(self):
        return dict(limit=self.limit, used=self.used, by_kind=self.by_kind, exhausted=self.exhausted())

    def _record(self, event, **details):
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        state = self._state()
        path = self.folder / STATE_FILE
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(path)
        with (self.folder / EVENT_FILE).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, **state, **details)) + '\n')

    def check_available(self):
        with self.lock:
            if not self.exhausted():
                return
            if not self.blocked:
                self._record('request_budget_exhausted', retryable=False)
                self.blocked = True
            raise RequestBudgetExceeded(
                f'request_budget_exhausted: {self.used}/{self.limit} model attempts; no request sent')

    def reserve(self, kind):
        with self.lock:
            self.check_available()
            self.used += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
            self._record('attempt_reserved', kind=kind, attempt=self.used)
            return self.used


def request_budget(owner):
    budget = getattr(owner, 'model_request_budget', None)
    if budget is None:
        budget = owner.model_request_budget = ModelRequestBudget()
    return budget
