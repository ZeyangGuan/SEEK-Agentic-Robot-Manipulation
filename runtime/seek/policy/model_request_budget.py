import json
from pathlib import Path
from threading import RLock


class RequestBudgetExceeded(RuntimeError):
    error_code = 'request_budget_exhausted'
    retryable = False


class ModelRequestBudget:
    @classmethod
    def resume(cls, folder):
        path = Path(folder)/'model_request_budget.json'
        budget = cls(folder=folder)
        if path.exists():
            saved = json.loads(path.read_text())
            budget = cls(limit=saved['limit'], folder=folder)
            if (type(saved['used']) is not int or saved['used'] < 0 or
                    (budget.limit is not None and saved['used'] > budget.limit)):
                raise ValueError('Invalid persisted episode allowance')
            if (any(type(v) is not int or v < 0 for v in saved['by_kind'].values()) or
                    sum(saved['by_kind'].values()) != saved['used']):
                raise ValueError('Persisted request accounting mismatch')
            budget.used, budget.by_kind = saved['used'], saved['by_kind']
        return budget

    def __init__(self, limit=150, folder=None):
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 150):
            raise ValueError('max_model_requests must be null or an integer in [1,150]')
        self.limit, self.used, self.by_kind = limit, 0, {}
        self.folder = Path(folder) if folder is not None else None
        self.lock = RLock()
        self.blocked = False

    def _record(self, event, **details):
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        state = dict(limit=self.limit, used=self.used, by_kind=self.by_kind,
                     exhausted=self.limit is not None and self.used >= self.limit)
        path = self.folder / 'model_request_budget.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(path)
        with (self.folder / 'model_request_budget.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, **state, **details)) + '\n')

    def check_available(self):
        with self.lock:
            if self.limit is not None and self.used >= self.limit:
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
