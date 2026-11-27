# SEEK: method ↔ code

This document maps the method described in the paper to the implementation in
[`runtime`](../runtime). Paths below are relative to
`runtime/seek/policy/` unless stated otherwise.

- [Notation](#notation)
- [Names used in the code](#names-used-in-the-code)
- [One control step](#one-control-step)
- [PAEB: what is stored and how it is retrieved](#paeb-what-is-stored-and-how-it-is-retrieved)
- [SEPP: how programs are revised](#sepp-how-programs-are-revised)
- [Snapshots and what resets](#snapshots-and-what-resets)
- [Module index](#module-index)

## Notation

| Paper | Meaning | Implementation |
|---|---|---|
| $\mathcal K_k=\{\mathcal E_k,\Pi_k\}$ | Knowledge after attempt $k$ | One JSON snapshot, `memory_<k>.json` |
| $\Pi_k=(P_k,\mathcal O_k)$ | SEPP: main program and named operations | `programs`, `active`, and `operation_knowledge` in the snapshot |
| $\mathcal E_k$ | PAEB: accumulated execution evidence | `transitions`, `episodes`, `improvement_history`, and the per-attempt record lists |
| $e_t=(x_t,a_t,x_{t+1},c_t,\ell_t)$ | One executed action with its measured effect | One entry of `transitions`, ID `e<attempt>:t<step>` |
| $g_t$ | Numerical objective | `intent` / control objective tracked by `experience_control.py` |
| $\mathrm{Recall}(\mathcal E,q_t)$ | Evidence retrieval | `recall_experience` → `experience_access.lookup` |
| $\mathcal E_k=\mathcal E_{k-1}\oplus\Delta\mathcal E(	au_k,y_k)$ | Evidence accumulation: append the attempt's records | Written by `execution_policy.py` during the attempt |
| $\operatorname{RevisePolicy}$ | Policy consolidation: keep, replace, or revert | [`tools/review.py`](../runtime/tools/review.py) |

$\mathcal K_0$ is created by `execution_policy.empty_memory(task)`: every record list is
empty, no program is active, and no operation is installed. Execution then starts
from the supplied delegation routine (`DEFAULT_PROGRAM`), which checks objectives
and consumes model-proposed candidates.

## Names used in the code

| In the code | In the paper |
|---|---|
| episode | attempt |
| teacher | the fixed general-purpose model |
| mechanism, mechanism program | SEPP main program |
| operation | named operation in $\mathcal O_k$ |
| experience, memory | PAEB evidence; the snapshot that stores it |
| boundary, handoff | a point where control passes to or from the model |
| candidate plan | a batch of actions proposed by the model |
| intent, control objective | numerical objective $g_t$ |
| progress guard | fixed progress monitor |


## One control step

**Observation.** The policy receives three 640×480 RGB images (head and two
wrist cameras), camera intrinsics and current extrinsics, proprioception, and the
language instruction. Depth and simulator object state are excluded by an
explicit whitelist in `model_policy.py`. From these inputs the runtime derives

- robot features per arm: end-effector position and rotation vector, gripper
  target, and measured gripper aperture;
- role measurements: for each image region the model has bound to a role, its
  pixel centre and area, maintained by segmentation with temporal memory
  (`unified_vision.py`, `temporal_segmentation.py`);
- on-demand geometry: `segment_rgb`, `measure_rgb` (multi-view silhouette shape),
  and `locate_rgb_point` (ray triangulation, or a ray–plane intersection for a
  single view), all in `rgb_perception.py`.

**Decision.** Programs are written in a restricted Python subset and run by the
interpreter in `mechanism_program.py` (at most 60,000 source characters, 10 s of
thread CPU time per call, call depth 16; no imports, I/O, or introspection). A
program defines `decide(ctx, state)` and may define `observe(ctx, state)`.
`decide` returns one of six decision kinds:

| Kind | Effect |
|---|---|
| `act` | Dispatch an action computed by the program |
| `cached` | Dispatch the next still-valid candidate proposed by the model |
| `continue` | Advance local state or select a named operation, without a physical step |
| `delegate` | Hand the step to the delegation routine |
| `ask` | Request model assistance |
| `blocked` | Report that the program cannot proceed |

**Action.** An action has 14 numbers: for each arm a world-frame translation
increment, a rotation-vector increment, and an absolute gripper target. Per-step
limits are 0.03 m and 0.15 rad (`configs/shared_harness.json`; a task file may
override them). Programs can query feasibility before dispatch with
`realize_motion` and `realize_joint_motion` (`motion_query.py`,
`joint_motion.py`).

**Candidates.** A model response may carry up to $H$ actions. $H$ depends only on
the task's native step limit $T$ (`prediction_horizon.py`):

| $T$ | ≤ 400 | ≤ 700 | ≤ 1100 | ≤ 1600 | > 1600 |
|---|---|---|---|---|---|
| $H$ | 5 | 10 | 15 | 20 | 25 |

A fresh observation and a program decision precede every dispatched action, so a
candidate batch can be interrupted after any step.

**Receipt.** Each physical step produces exactly one execution receipt with the
command actually sent and the robot state before and after it
(`execution_receipt.py`). Only steps with a receipt become transitions.

**Progress monitor.** `execution_progress.py` inspects receipts independently of
the program. With the default limits it raises a review request after 8 steps
without motion, or after 12 steps of oscillation or of response below 10% of the
commanded motion. A request clears pending candidates and asks the model to
review; it is not a task failure.

## PAEB: what is stored and how it is retrieved

A stored transition keeps the requested action, the executed command, the
measured states before and after, the program version, the operation invocation,
and where the action came from. Other record
types are kept separately: programs, model proposals, errors, motion-query
evidence, effect assessments, objectives and their invocations, operations and
their invocations, reviews, and lessons.

`recall_experience` is available to programs, to the model during an attempt, and
to the post-attempt review. `experience_access.lookup` applies, in order:

1. attempt and record-type filters;
2. exact record IDs within the filtered set;
3. text ranking by the number of distinct tokens shared between the query and
   each serialized record.

One lookup accepts up to 8 IDs, a query of at most 500 characters, and returns 1
to 12 records. The model may issue two lookups and two deferred-context reads per
observation or review. Lookups made by a program are local and do not count as
model requests.

PAEB stores evidence only. It never produces an action; actions come from the
program, from an operation, or from a candidate proposed by the model.

## SEPP: how programs are revised

**Online.** When the model is asked for help it can return, in one response,
candidate actions, role bindings, a new main program, new or edited operations, an
objective, and effect assessments. The runtime validates the whole response before
applying any part of it: the code must compile under the interpreter, cited
evidence IDs must exist, and the program is run on the current context to check
that it returns a valid decision. An accepted program is stored as a new version
whose ID is the SHA-256 of its source, and `decide` is re-run with it
immediately.

**After the attempt.** `tools/review.py` gives the model the benchmark outcome,
the executed transitions, errors, operation use, and the version history. The
model returns one of

- `keep` — the current main program stays active;
- `replace` — complete new source is installed;
- `revert` — an earlier stored version becomes active again.

Independently of that choice it may add, edit, or restore individual operations
and record lessons (`when` / `do` / `expect` / `if_failed`). The review may inspect
archived observations and replay candidate code on recorded inputs
(`review_observations.py`, `review_replay.py`). Up to three review responses are
allowed; if none is valid the previous policy is retained. The raw execution
records are never modified by a review.

## Snapshots and what resets

Attempt $k$ reads `memory_<k-1>.json` and writes `memory_<k>.json`; the input file
is hashed on load and is never modified. The review then updates
`memory_<k>.json` in place and saves its own inputs and outputs under
`review_<k>/`.

At the start of every attempt the runtime clears all scene-dependent state: role
bindings, active objectives, candidate plans, tracking state, operation
invocations, and the program's local `state`. Programs, operations, and evidence
are carried over.

For evaluation on unseen layouts, `SEEK_FREEZE_LEARNING=1` keeps incoming programs
and operations fixed while observations and execution feedback remain available.
Adding `SEEK_EPISODE_ADAPTATION=1` allows program and operation revisions within
the episode; this is the evaluation protocol of the paper. In both modes the
reusable output is restored byte-for-byte from the input; local evidence is saved
separately in `evaluation_trace_memory.json`. The next episode starts from the same
snapshot, and no post-attempt review runs.

## Module index

**Runtime core**

| Module | Role |
|---|---|
| `execution_policy.py` | The SEEK policy: per-step loop, model requests, program installation, snapshot checkpoints |
| `mechanism_program.py` | Restricted interpreter for programs and operations |
| `operation_knowledge.py`, `builtin_operations.py` | Named operations, their versions and invocations; the empty operation bank of $\mathcal K_0$ |
| `experience_control.py` | Numerical objectives and their status |
| `candidate_plan.py`, `candidate_conditions.py`, `prediction_horizon.py` | Model-proposed candidates, their guard conditions, and the horizon $H(T)$ |
| `plan_memory.py` | Snapshot schema and program interface identifiers |
| `manipulation_contract.py` | Instructions given to the model |

**Execution**

| Module | Role |
|---|---|
| `execution_receipt.py`, `action_feedback.py` | Execution receipts and the tracking-error guard |
| `execution_progress.py` | Fixed progress monitor |
| `motion_query.py`, `joint_motion.py` | Feasibility queries and joint-space proposals |
| `observed_base_ik.py`, `ik_continuity.py`, `joint_reference.py` | Inverse kinematics in the measured base frame and joint-continuity checks |
| `feature_geometry.py` | Local coordinates for pose features and goal errors |

**PAEB**

| Module | Role |
|---|---|
| `experience_access.py` | Record types, IDs, and `recall_experience` |
| `effect_evidence.py` | Image-patch measurements and effect assessments |
| `motion_retrieval.py` | Nearest motion-query evidence for a joint state |
| `boundary_learning.py` | Records at model handoffs and the operation cases shown to the model |
| `improvement_memory.py` | Review history and lessons |

**Perception**

| Module | Role |
|---|---|
| `rgb_perception.py` | `segment_rgb`, `measure_rgb`, `locate_rgb_point` |
| `silhouette_hull.py`, `single_view_geometry.py` | Multi-view silhouette constraints and support-constrained single-view shape hypotheses |
| `calibrated_geometry.py` | Camera projection, ray geometry, mask observations and calibrated DLT |
| `segmented_geometry.py`, `visual_measurements.py`, `rigid_geometry.py`, `tool_geometry.py` | Mask geometry, region measurements, rigid transforms, gripper envelopes |
| `unified_vision.py`, `segmentation_client.py`, `temporal_segmentation.py` | Role binding and tracking with the segmentation service |
| `observation_images.py`, `head_calibration_capture.py` | Image encoding and the camera calibration captured from the simulator |
| `camera_readiness.py`, `camera_preflight.py` | Blank-frame retries on every observation, and the capture-only camera probe |

**Model interface**

| Module | Role |
|---|---|
| `model_policy.py` | Observation whitelist |
| `action_validation.py` | Chunk shape, per-step bounds, and conversion to executor commands |
| `dialogue_policy.py`, `robot_prompt.py` | Run settings, the robot-command instructions, and request sizing |
| `model_gateway.py` | Model requests, budgets, and retries |
| `provider_response.py` | Streamed responses and provider error classification |
| `provider_compat.py`, `transport_retry.py` | Endpoint validation and transport retries |
| `model_request_budget.py` | Per-attempt request accounting |
| `context_window.py`, `context_view.py`, `prompt_layout.py` | What the model sees: bounded views, deferred history, and message layout |

**Review**

| Module | Role |
|---|---|
| [`tools/review.py`](../runtime/tools/review.py), [`tools/review_transport.py`](../runtime/tools/review_transport.py) | Post-attempt review and its transport |
| `review_learning.py`, `review_observations.py`, `review_replay.py` | Review evidence, archived-frame inspection, and code replay on recorded inputs |

**Evaluation**

| Module | Role |
|---|---|
| `eval_failure.py`, `episode_outcome.py`, `task_scope.py` | Policy-failure handling, final scoring, and instruction matching |
| `frozen_evaluation.py` | Evaluation modes: fixed procedures, or within-episode revision with nothing carried over |
