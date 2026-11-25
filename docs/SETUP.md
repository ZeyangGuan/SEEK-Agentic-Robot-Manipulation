# Running SEEK

All commands below are run from `runtime/`.

- [Install](#install)
- [Check the installation](#check-the-installation)
- [Run one attempt](#run-one-attempt)
- [Review the attempt](#review-the-attempt)
- [Chain attempts](#chain-attempts)
- [Evaluate a snapshot](#evaluate-a-snapshot)

## Install

SEEK uses three separate Python environments: one for the simulator, one for the
policy, and one for SAM 3 segmentation. The simulator needs Linux, an NVIDIA GPU,
and the matching Isaac Sim installation.

```bash
python setup/prepare.py
/path/to/policy/bin/pip install -r setup/requirements-policy.txt
/path/to/sim/bin/pip install -r setup/requirements-sim.txt -c setup/constraints-sim.txt
/path/to/policy/bin/python setup/fetch_assets.py --data-root /path/to/data --cache /path/to/asset-cache
cp setup/seek.env.example setup/seek.env
```

`setup/prepare.py` clones the third-party packages pinned in
`configs/upstream.lock.json` (RoboDojo, XPolicyLab, Isaac Lab, cuRobo) into
`third_party/`. RoboDojo is taken from its public repository at the pinned commit
and patched with `setup/robodojo_integration.patch`, which adds the execution-side
IK continuity guard, camera validation, and external asset paths, and carries the
upstream fix that pushes the restored drive targets of articulated objects to PhysX
on reset (RoboDojo pull request #48); the script checks that the result is the
commit recorded in the lock file. Use
`--only NAME ...` to install a subset. The task specifications are in
`configs/tasks/`, and `configs/perception-runtime.json` names the SAM 3 revision
and the checksum of its checkpoint, which is obtained separately.

Set the paths in `setup/seek.env`, then source that file in the shell that runs
SEEK. Set `SEEK_MODEL_KEY` and `SEEK_MODEL_ENDPOINT` in the same shell. Isaac Sim
starts only after its license has been accepted, for a headless run with
`export OMNI_KIT_ACCEPT_EULA=YES`. Additional Isaac Sim options, such as a private
cache location on a shared machine, can be passed in `SEEK_KIT_ARGS`. The
endpoint must implement the Responses API with streaming. The model and its
reasoning effort are set in the protocol profile used by the attempt and by its
review ([`configs/protocols/seek.json`](../runtime/configs/protocols/seek.json):
`gpt-6.2-astra` at `xhigh`, with no cap on model requests).

## Check the installation

Neither command makes a model request or starts the simulator.

```bash
"$SEEK_POLICY_PYTHON" -m seek.run --help
"$SEEK_POLICY_PYTHON" -m seek.run deposit_coin --seed 0 --layout 1 \
  --profile configs/protocols/seek.json --dry-run
```

The offline tests cover the procedure interpreter, the geometry interfaces, and
the evaluation modes:

```bash
cd tests
PYTHONPATH=.. "$SEEK_POLICY_PYTHON" -m unittest discover -p 'test_*.py'
```

## Run one attempt

An attempt needs an input snapshot, an output path, and an attempt number. The
example below is attempt 1 of a new task, starting from an empty snapshot
($\mathcal K_0$). `RUN_ROOT` must be a new directory.

```bash
export RUN_ROOT=/path/to/new/run
mkdir "$RUN_ROOT"
cp configs/protocols/seek.json "$RUN_ROOT/profile.json"

export SEEK_MEMORY="$RUN_ROOT/memory_0.json"
export SEEK_OUTPUT="$RUN_ROOT/memory_1.json"
export SEEK_EPISODE=1

"$SEEK_POLICY_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from seek.policy.execution_policy import empty_memory
Path(os.environ['SEEK_MEMORY']).write_text(json.dumps(empty_memory('deposit_coin')))
PY

"$SEEK_POLICY_PYTHON" -m seek.run deposit_coin --seed 0 --layout 1 \
  --profile "$RUN_ROOT/profile.json" --results-root "$RUN_ROOT/episode_1"
```

The launcher writes a timestamped directory under `--results-root` containing
`status.json`, the benchmark's `result.json`, and the trace in `model/<id>/`
(`observations.jsonl`, `requests.jsonl`, `seek_transitions.jsonl`, and others). The
output snapshot holds everything in the input snapshot plus the evidence and
online procedure revisions from this attempt.

## Review the attempt

The launcher does not run the post-attempt review. Run it once the attempt has
exited normally with a valid benchmark score:

1. Append the attempt's outcome to `episodes` in the output snapshot.
2. Call `tools.review.review(root, report)`. It reads `root/memory_<k>.json`,
   sends the review request using the model and reasoning effort in
   `root/profile.json`, and writes the reviewed snapshot back to the same file.
   Its inputs, requests, and decision are saved under `root/review_<k>/`.

```python
import json, os
from collections import Counter
from pathlib import Path
from tools.review import review

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

root, k, layout = Path(os.environ['RUN_ROOT']), 1, 1
run = next((root / f'episode_{k}').glob('20*'))
native = json.loads((run / 'result.json').read_text())
detail = next(iter(native.get('details', {}).values()), {})
trace = next(run.glob('model/*/observations.jsonl')).parent

report = dict(
    episode=k, layout=layout, trace=str(trace),
    success=detail.get('success'), score=native.get('score'),
    final_step=rows(trace / 'observations.jsonl')[-1]['step'],
    all_model_requests=len(rows(trace / 'requests.jsonl')),
    execution_sources=dict(Counter(t['source'] for t in rows(trace / 'seek_transitions.jsonl'))),
    policy_outcome=native.get('policy_outcome'))

snapshot = root / f'memory_{k}.json'
memory = json.loads(snapshot.read_text())
memory['episodes'].append({key: report[key] for key in (
    'episode', 'layout', 'success', 'score', 'final_step',
    'all_model_requests', 'execution_sources', 'policy_outcome')})
snapshot.write_text(json.dumps(memory))

print(review(root, report)['outcome'])
```

Keep a copy of the snapshot from before the review if you want to compare the
two.

## Chain attempts

Attempt `k` reads `memory_<k-1>.json` and writes `memory_<k>.json`:

```bash
export SEEK_MEMORY="$RUN_ROOT/memory_$((k-1)).json"
export SEEK_OUTPUT="$RUN_ROOT/memory_$k.json"
export SEEK_EPISODE=$k
```

Run the attempt on a new layout, review it, and repeat. Use a new
`--results-root` for every attempt. The snapshot refers to the observation
archives of earlier attempts by path, so keep the whole `RUN_ROOT` together when
moving it.

The results in the README use 5 evolution attempts per task on distinct layouts;
the reviewed snapshot after the fifth attempt is $\mathcal K_5$.

## Evaluate a snapshot

Evaluation episodes run on layouts unseen during evolution. Every episode starts
from a copy of the same snapshot, may collect evidence and revise procedures
within the episode, and writes nothing back:

```bash
export EVAL_ROOT=/path/to/new/evaluation
export SEEK_FREEZE_LEARNING=1
export SEEK_EPISODE_ADAPTATION=1
export SEEK_MEMORY="$RUN_ROOT/memory_5.json"
export SEEK_OUTPUT="$EVAL_ROOT/episode_$n/memory_out.json"
export SEEK_EPISODE=6
```

The output snapshot is restored byte-for-byte from the input; the evidence and
revisions of the episode are saved next to it in `evaluation_trace_memory.json`
and `episode_adaptation_audit.json`. No post-attempt review is run. Add
`--variant random` for the randomized layouts of the Generalization tasks.
With `SEEK_FREEZE_LEARNING=1` alone, procedures are fixed for the whole episode.

The no-carryover control evaluates the same episodes from an empty snapshot.

The scheduler used for the paper's multi-attempt experiments, which adds GPU
allocation, resumption, and bookkeeping around this loop, is not part of this
release.
