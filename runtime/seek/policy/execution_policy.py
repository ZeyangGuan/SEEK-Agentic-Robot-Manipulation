from copy import deepcopy
from dataclasses import replace
import hashlib,json,os,time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .dialogue_policy import Model as Transport, canonical_tool_history
from .model_policy import plain
from .mechanism_program import (Program,ProgramError,digest,validate_candidate,source_schema,
                                DEFAULT_CPU_BUDGET_SECONDS)
from .execution_progress import ExecutionProgressGuard,GUIDANCE as PROGRESS_GUIDANCE
from .unified_vision import UnifiedVision,frame
from .improvement_memory import GUIDANCE,improvement_context,lesson_context,LESSON_GUIDANCE
from .context_view import GUIDANCE as CONTEXT_GUIDANCE,action_view
from .prompt_layout import ACTION_GUIDANCE,action_inputs,with_cache_key
from .motion_query import CapabilityRequest, selected_motion_request
from .motion_retrieval import MotionEvidenceIndex
from .effect_evidence import measure_patch,update_effects,current_effects,effect_counts
from .candidate_plan import CandidatePlan,command_delta
from .plan_memory import SCHEMA,PROGRAM_ABI,accept_program,require_compatible
from .execution_receipt import SCHEMA as RECEIPT_SCHEMA
from .experience_control import ExperienceControl, measured_features, intent_schema, GUIDANCE as EXPERIENCE_GUIDANCE
from .experience_access import (lookup,available_ids,recall_tool,validate_lookup_arguments,
    is_context_lookup,LOOKUP_LIMIT,CONTEXT_LOOKUP_LIMIT,GUIDANCE as ACCESS_GUIDANCE)
from .candidate_conditions import guard_schema,validate_guard,guard_reason,GUIDANCE as CONDITION_GUIDANCE
from .prediction_horizon import prediction_horizon,prediction_budget,prediction_contract,GUIDANCE as HORIZON_GUIDANCE
from .boundary_learning import capture_boundary, learning_agenda, online_learning_agenda
from .context_window import ContextArchive,bound_payload,operation_source_ids
from .visual_measurements import measure_region, center_surface, GUIDANCE as VISUAL_GUIDANCE
from .segmented_geometry import segment_geometry, GUIDANCE as SEGMENT_GUIDANCE
from .rgb_perception import measure_rgb, locate_rgb_point, segment_rgb, GUIDANCE as RGB_GUIDANCE
from .rigid_geometry import GUIDANCE as GEOMETRY_GUIDANCE
from .tool_geometry import TOOL_GEOMETRY, tool_base_relation
from .operation_knowledge import (OperationKnowledge, prepare_updates, updates_schema,
    invocation_schema, invocation_state, GUIDANCE as OPERATION_GUIDANCE)
from .manipulation_contract import ACTOR_ROLE, assemble_instructions
from .review_observations import (observation_tool, inspect_observations, lineage_archives,
    GUIDANCE as OBSERVATION_GUIDANCE)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)

executed_delta=command_delta


def semantic_feedback(value):
    if isinstance(value,dict):
        return {k:semantic_feedback(v) for k,v in value.items() if k not in ('id','query_id','plan_id','candidate_id','experience_invocation','created_at','timestamp')}
    if isinstance(value,list):return [semantic_feedback(v) for v in value]
    return value


def referenced_observations(state,episode,archive,current_step):
    steps=set()
    if isinstance(state,dict):
        identity=state.get('observation_id')
        prefix=f'e{episode}:s'
        if isinstance(identity,str) and identity.startswith(prefix) and identity[len(prefix):].isdigit():
            step=int(identity[len(prefix):])
            if step in archive and step<=current_step:steps.add(step)
        for value in state.values():steps.update(referenced_observations(value,episode,archive,current_step))
    elif isinstance(state,list):
        for value in state:steps.update(referenced_observations(value,episode,archive,current_step))
    return steps


ABI='''SEPP executable mechanism (seek-control).
The program may interpret observations, choose actions and recover. No predefined grasp
phases or diagnosis labels are required. One act tool returns actions (up to the configured candidate horizon), execution_note,
bindings (optional named roles with CURRENT camera-local normalized xyxy boxes), program,
and state_update. state_update is an object/null: a full replacement of CURRENT
evidence-based beliefs/parameters, applied on accepting the proposal, never a future action effect.
Teacher actions do not carry a second state queue. Only decide owns receipt-dependent state.
Never put a proposed action's completion in state_update. Physical execution is not proof of task success.
Boxes are relative to the binding's camera panel, never the entire three-camera triptych.
Binding camera is optional: cam_head (default), cam_left_wrist or cam_right_wrist.
program=null keeps the active program. Otherwise {source,description,evidence_ids} replaces it.
source is complete code defining def decide(ctx, state), plus optional helper functions.
This code runs before EACH physical step, including the FIRST fresh teacher action.
The default delegate checks the measured objective and consumes valid teacher candidates.
Learned programs may generate actions from current observations and parameters.
Use explicit cached when a learned operation needs its particular teacher path. You may teach actions
with a measured intent without inventing a program first. A null program never bypasses an existing
mechanism. After an act response, decide sees the NEW state and buffered_actions;
use cached to consume a teacher candidate, act to replace it, or ask to request missing information.
Use continue with state_update to make an evidence-supported LOCAL stage transition
and run decide again on the same observation without a model call or physical step.
Use delegate to explicitly check the current objective and then guarded
teacher candidates; delegation asks the model only when that local path cannot proceed.
An explicit ask still requires model input and never silently falls back to a stale
candidate. Missing current-scene bindings, unresolved interpretation, unsupported
recovery or physical progress supervision can require the model; a timer, completed
numeric subgoal or empty candidate buffer alone does not imply missing information.
When the next decision is supported by current measurements, perform it locally.
Do not translate an uncertain task effect into continue merely to avoid a call.
context.teacher_execution reports actual consumption since the last accepted reply.
If the mechanism generates actions itself, supply its required parameters and use
actions=[] rather than also generating an unused trajectory. Supply multiple rows
when decide will actually use cached/delegate and their applicability is supported.
Local act intentionally replaces the remaining candidate plan. There is no hidden
execute-five cap: a fresh observation and checked decision precede every step.
Teacher batches are resolved ONCE: first delta from the anchor observation, later deltas
from the previous requested target. Tracking drift does not move those intended targets.
ctx.candidate_plan identifies this fixed-target plan, anchor, next target and lifecycle.
buffered_actions is a CURRENT projection of its remaining targets: the first delta starts
at the current observed pose; later deltas start at the previous requested target.
It is not a second queue. cached dispatches the exact stored target. A direct decide act
is a NEW one-step plan relative to the current observation and replaces the old suffix.
Existing per-step bounds also apply from the current observation to the next fixed target.
If it exceeds them, the plan is invalidated explicitly and decide can replan; no target
is silently rebased or clamped. Execution advances a plan only on its matching receipt.
Tracking continuity belongs to a plan, across RPCs. Controller events and interruptions
survive in last_execution/history; executed-but-drifted is still an executed transition.
requested_action is the original increment; action is the actual command delta from the
pre-execution observation. plan_reference, intended_target and execution_receipt preserve
their identities, accepted IK targets and measured joint response separately.
Code persists across episodes; state and role bindings reset. Older ABI memories are not silently imported.
Use current features and episode state parameters; do not embed layout coordinates in permanent code.
ctx: step,instruction,robot,roles,features,history,experience,buffered_actions,last_execution,last_error,limits,native_step_limit.
robot contains measured EE poses [x,y,z,w,x,y,z], arm joints, gripper_aperture and
gripper_joint_positions (meters), plus executor gripper_target. ee_joint_state is its legacy
command alias. features: left/right.x/y/z/rx/ry/rz, gripper_target, gripper_aperture,
and role.<name>.u/v/area plus role.<name>.surface_x/surface_y/surface_z when
camera depth is available (otherwise null). Surface coordinates are visible points
at the tracked camera role center in environment meters, not object centers
or proof of retention. gripper is a legacy target alias, never a measured opening.
gripper_aperture normalizes the first measured finger joint only; it is not total jaw width.
For both fingers' achieved opening and asymmetry, use gripper_joint_positions with tool_geometry.
Finger positions can diverge during motion even after both reached the opening target.
Old receipts lack gripper_aperture; absence remains unknown, not copied from a command.
roles contain camera,valid,center,method,generation,description,identity_reason.
Role center is [x,y] in the declared camera tile PIXELS; u/v and feature role.<name>.u/v are
normalized to that tile. area is pixels squared; normalized_area is the image fraction.
Do not compare center directly to a normalized bbox or add the triptych tile offset.
Unknown features are null.
ctx.tool_geometry gives fixed X5 hardware geometry in the observed link6 frame.
Each inner pad spans inner_pad_x and inner_pad_z. With measured gripper joints [j7,j8],
the two inner-face y coordinates are inner_pad_y_at_zero_joint + [j7,-j8].
Use bounds = ctx['tool_geometry']['outer_bounds'] for the whole-tool geometry.
bounds['base'] and bounds['fingers_at_zero_joint'] are axis-aligned mesh envelopes
in link6. The joint axes are bounds['finger_joint_axes'], in the same nested object.
For finger i, translate bounds['fingers_at_zero_joint'][i] by
bounds['finger_joint_axes'][i] times measured gripper_joint_positions[i]
(or the candidate joint position when evaluating a proposed opening).
Transform all eight corners by the EE pose when comparing tool space with visible
surface bounds.
These envelopes cover the whole tool, include empty space, and are not contact measurements.
ctx.tool_base_relation compares the two fixed base envelopes at the current observed
EE poses. Motion queries return result.tool_base_relations for current and candidate
poses plus axis_gap_change_m. A positive axis gap certifies base-box separation;
overlap is only a possible interference cue. Use actual response and RGB to interpret
it, including when a useful hold by one arm obstructs the other's goal. The control
program can compare current/candidate relations when choosing coordinated passage,
separation or useful work by the other arm. Geometry alone does not reject an action
or prevent departure from overlap. Fingers, arm links and swept paths are not checked.
For a local tool point r, its environment position is ee_xyz + R(ee_quaternion) * r;
to place it at a visually estimated point g, the link6 target is g - R(q) * r.
These dimensions replace guessed EE-to-finger offsets; they do not supply object
height, contact, retention or an ideal grasp point. Choose contact from current RGB.
history is the last 24 actual receipts; experience samples up to 48 actual prior-episode receipts.
Both contain before/action/after, IDs and source episodes. Complete records remain available through recall_experience.
Return {'kind':'act','action':[14 numbers],'on_execution':{},'reason':'...', 'diagnosis':anything JSON},
or {'kind':'cached','reason':'...'} to execute the next teacher prediction,
or {'kind':'continue','state_update':{},'reason':'measured basis for local transition'},
or {'kind':'delegate','reason':'use the current objective/candidates under local checks'},
or {'kind':'ask','state_update':{},'reason':'specific missing information, error or capability needed'},
or {'kind':'blocked','reason':'why no supported action can continue'} to END this episode.
blocked is terminal: the current state is scored and this rollout cannot resume.
Use blocked for an actual missing prerequisite/capability that prevents continuation,
not merely incomplete certainty. No action candidate is made feasible by a belief.
Any decision may supply state_update for CURRENT evidence. act/cached may also supply on_execution
for a receipt-dependent full state replacement; omitted/null keeps current state for BOTH act/cached.
A non-null on_execution is invalid for ask/blocked/continue/delegate. These are the only state commit semantics.
continue/delegate may also carry intent with the measured control contract. It is
validated against this observation before any state/effect update, then made
available to control_status. Changing it invalidates candidates for the old
objective. Repeating the same objective keeps its receipt and failure evidence.
Derive targets from current observations under the learned rule; do not request
the model merely to install a target already determined by local code. When consolidating
a tested method, persist its observer and next-goal rule in program: episode state
resets, and one-off parameter updates do not teach the next episode. Exploratory
actions may precede that consolidation.
continue commits no execution and no automatic task effect. Cyclic or excessive
local transitions request model repair rather than directly terminating the episode.
State can represent arbitrary method stages and diagnostics. Rejection discards pending on_execution
and all remaining teacher predictions, but preserves valid current-evidence updates. Empty buffer +
cached hands back to the model without committing any on_execution. Legacy decision.state is invalid.
last_execution is an immutable executed/rejected event with id, step, action, executed and detail.
State objects with observation_id='eN:sK' retain a current-episode visual reference;
the referenced real images accompany later teacher requests for direct comparison.
For executed, receipt and receipt_id are opaque STRING transition IDs, never dictionaries.
detail is {type:'action_executed',receipt_id,tracking,controller_event}; controller_event
may be null. Rejected has receipt_id=null and detail is the exact native rejection.
The existing top-level plan_reference/controller_event/tracking fields remain available.
It survives teacher responses and proposal errors until replaced by a NEW execution result. Process
its id in your state if needed. Rejection returns to decide first; local recovery does not require the model.
last_error describes proposal or mechanism errors/requests, never replaces last_execution.
Language: plain positional def without annotations/defaults; if/elif/else; for/while; break/continue;
list/dict, indexing/slicing, comprehensions, assignments, arithmetic/comparisons. No imports/classes,
lambdas/try/generators, files/network/eval/global variables. Only data methods get/items/keys/values/
append/extend/pop/copy/update/count/index. Functions abs,min,max,sum,len,round,float,int,bool,str,list,
dict,sorted,range,enumerate,zip,all,any,sqrt,sin,cos,atan2,exp,log,clip(x,lo,hi),norm(vector).
control_status() assesses the current measured objective and returns ask/fallback.
It does not fit responses or generate actions. Numeric completion is not task success.
Role names must keep a consistent semantic meaning across episodes. Check context and actual effects.
Execution uses a uniform __CPU_BUDGET_SECONDS__ second thread-CPU safeguard per observe/decide
call. Syntax counts are diagnostic, not a default cutoff. This excludes network waits
and other threads and is independent of model input tokens or historical indexing.
Only current observation or saved evidence IDs may be cited. A proposal/method/error citation
is not an executed transition or confirmed task effect. Empty citations allow initial provisional code.
Syntax/interface errors return exact feedback without motion. Historical zero activation does NOT
reject a legal candidate. Future real results determine usefulness. Native task scoring is immutable.
Local template/flow/SAM tracking does NOT automatically call a model. Invalid attribution stays unknown.
After a current visible instance is bound, segmentation temporal memory tracks it from past and current RGB
without model calls. method='temporal_segmentation' is an image association that can drift, not grasp,
contact or retention proof. A missing track remains unknown; new bindings start separate memories.
Each binding may include text: a short visual concept for the segmentation model to locate within the box.
The box selects the current instance; text specifies what to segment when adjacent objects
share the box. Omit text or use null for box-only segmentation. Description remains the
role's task meaning and is not automatically used as a segmentation prompt. The selected mask
seeds temporal tracking; a semantic prompt is not evidence that the mask is correct.
With text, optional prompt_mode='joint' conditions the segmentation model on the concept and the box
as a positive visual example. The default 'concept' detects by text before selecting
by box. Joint prompts may recover unusual appearances or include adjacent material;
inspect the visible mask when choosing the mode. Missing detections do not switch modes.
Each role tracks only its declared camera. Separate roles may observe the same object
from different cameras; choosing matching instances and interpreting object-to-tool
motion are learned decisions. Pixel coordinates from different cameras are not interchangeable.
Use the images already supplied in this act request to bind objects; avoid repeating a fully occluded
reference query. New bindings have a new generation. Interpretation may use any number of named roles.
If unknown, request genuinely new information or a bounded physical information action; do not invent
measurements or repeat observation-only requests with unchanged evidence. An exactly repeated request
with unchanged observation, program, state, bindings, candidates and feedback ends as unresolved;
there is no failure-count switch or task-specific recovery. Continue until native end or explicit blocked.
'''.replace('__CPU_BUDGET_SECONDS__',str(DEFAULT_CPU_BUDGET_SECONDS))

DEFAULT_PROGRAM="""def decide(ctx,state):
 assessed=control_status()
 if assessed['kind']=='ask':
  return assessed
 if ctx['buffered_actions']:
  if ctx['capabilities']['motion_query']:
   motion=realize_motion(ctx['buffered_actions'][0],'prefix')
   if motion['result']['status']!='feasible':
    return {'kind':'ask','reason':'No tested prefix is feasible; use motion evidence to change intent'}
   if motion['result']['fraction']<1:
    return {'kind':'act','action':motion['result']['action'],'reason':'Kinematically feasible prefix; reobserve before continuing the original intent'}
  return {'kind':'cached','reason':'Use the checked teacher plan'}
 return {'kind':'ask','reason':assessed['reason'],'diagnosis':{'learning':assessed,'candidate_plan':ctx['candidate_plan']}}"""

FEEDBACK_ABI='''
Feedback capabilities under ABI 4; SEEK source programs require explicit migration/review:
ctx.capabilities.motion_query advertises the live execution-side numerical solver.
realize_motion(action, mode) is a pure local query, with mode exact or prefix.
realize_joint_motion(joint_deltas, grippers) is an optional pure query for an explicit
posture change: joint_deltas={'left':[six radians], 'right':[six radians]},
grippers={'left':target, 'right':target}. Both arms are explicit; zero deltas retain
the measured joint configuration. The robot URDF and measured base pose predict
the corresponding EE increment. Existing joint continuity and EE action bounds
apply. This exact query never clips the proposal or selects a retreat for you.
It does not run IK or its collision model: feasibility here covers those bounds,
not collision clearance. Assess the proposed robot/tool route from available geometry.
If q['result']['status']=='feasible', return kind='act', action=q['result']['action'],
query_id=q['id'] to execute that joint solution. Without query_id an EE action uses
ordinary IK; the same endpoint can have different joint solutions. The 14-vector
action and recorded action history remain EE increments, never joint radians.
Joint-selected receipts remain available in history/recall with their chosen joint
increments and measured responses.
q['result']['kinematics'] gives current/target joints and EE poses plus current FK
residuals. It predicts robot kinematics, not collision clearance, physical response,
contact or task effects. Choose this coordinate system when useful for your method;
compare the resulting measured joint/tool response before reusing it.
The returned record contains result: use q['result']['status'], q['result']['action'],
q['result']['fraction'] and q['result']['trials']; status is not a top-level field.
Use dict indexing/get for ctx and state (ctx['last_execution'], never ctx.last_execution).
Motion feasibility is a numerical result for the configured robot/static-scene model.
Current camera surfaces are not inserted as obstacles into that model; use camera geometry
to assess the tool and held material's surrounding space.
The query record also reports solver_tolerances[arm].position_m and orientation_rad
from the live inverse-kinematics configuration. Each trial also keeps solver_errors
per arm: position_error_m and orientation_error_rad for its returned numerical
solution when available. These are solution errors, not measured physical tracking
errors. Compare them with the subsequent execution_receipt, including joint_response,
before deciding whether difficulty comes from solving, execution or observed contact.
The configured tolerances are numerical acceptance
thresholds, not guaranteed executed accuracy or minimum useful command sizes.
'exact' means the requested increment was queried without shortening; solver success
still allows residual pose error. Judge remaining error against the current goal and
actual receipts. Little response to an already-small correction alone does not prove
collision, obstruction or grasp loss; do not silently relax the task's requirements.
It does NOT move, prove contact or prove task success. A prefix follows the same
SE(3) increment direction and keeps grippers unchanged. Gripper-change commands
are indivisible and receive exact queries only. The default cold delegate uses
the last dispatched gripper targets for this distinction: observed executor targets or aperture
aperture can stay nonzero around an object despite an unchanged close command.
ctx.commanded_grippers holds model targets separately from executor targets and measured aperture.
It uses
this numerical capability, not a learned recovery policy. A partial realization
discards the unverified suffix and never commits the original completion state.
A learned program chooses whether to use a query and how to interpret its result.
ctx.motion_evidence retains numerical/rejection evidence separately from executed
history; local full records and model compact views preserve evidence identities.
No tested prefix is not proof of global unreachability; change pose/orientation or
intent using current evidence. Do not lower execution limits or hardcode a task.
measure_patch(camera,bbox,reference_step) compares a normalized camera-local RGB
ROI between two real observations. It returns pixel change and mean BGR values;
occlusion, lighting and camera motion also cause changes. Design an observable
effect test and a stateful confirmation method, not an execution-count proxy.
Optional effects on an act proposal or decide result are a list of objects:
{id,name,status,evidence_ids,reason}, status pending/confirmed/refuted/unknown.
An id identifies ONE task event in this episode; repeated observations update it
instead of incrementing it. evidence_ids must name actual eN:sK observations.
ctx.effects and ctx.effect_counts expose these revisable semantic assessments;
they are not native truth. Use this ledger for claimed task effects and counts.
on_execution is only command receipt state, never automatic effect confirmation.
Pending/unknown effects retain their reference observations for later comparison.
Programs may define def observe(ctx,state), returning only state_update, effects
and/or reason. It runs once per observation/program/input revision, including the final
physical observation, before the next action. Accepting teacher parameters/bindings or
a changed local state update refreshes it on the same image; its own state output does
not trigger a refresh. Use observation/event IDs for idempotent assessments and counts.
Old control programs with an earlier interface keep once-per-observation behavior until explicitly reproposed.
It may use measure_patch, measure_region and numeric helpers,
but cannot request or execute motion. Use it for task-effect state and event
confirmation; decide then chooses control using that updated observation state.
blocked ends the current episode and scores its present state. A later authorized
episode is a new rollout, not a resumption. It is distinct from a transport pause.
'''+VISUAL_GUIDANCE+SEGMENT_GUIDANCE+GEOMETRY_GUIDANCE+RGB_GUIDANCE

class MechanismBlocked(RuntimeError):pass

def empty_memory(task):
    from .builtin_operations import initial_operation_knowledge
    return dict(schema=SCHEMA,task=task,programs=[],program_abis={},active=None,
        transitions=[],episodes=[],improvement_history=[],
        operation_knowledge=initial_operation_knowledge())

def sample_experience(transitions):
    indices=sorted(set(np.linspace(0,len(transitions)-1,min(48,len(transitions)),dtype=int))) if transitions else []
    return deepcopy([transitions[i] for i in indices])

def proposal_schema(horizon=5):
    def obj(p):return dict(type='object',additionalProperties=False,properties=p,required=list(p))
    s=dict(type='string')
    program=obj(dict(source=source_schema(),description=s,evidence_ids=dict(type='array',items=s)))
    binding=obj(dict(name=s,description=s,bbox=dict(type='array',minItems=4,maxItems=4,items=dict(type='number'))))
    binding['properties']['text']=dict(anyOf=[s,dict(type='null')])
    binding['properties']['camera']=dict(type='string',enum=['cam_head','cam_left_wrist','cam_right_wrist'])
    binding['properties']['prompt_mode']=dict(type='string',enum=['concept','joint'])
    return obj(dict(actions=dict(type='array',minItems=0,maxItems=horizon,
        description='Each 14-number row is one environment step for both arms together: '
            '[left dx,dy,dz,rx,ry,rz,gripper, right dx,dy,dz,rx,ry,rz,gripper]. '
            'Choose useful motion or a purposeful hold for each arm. Six zero pose '
            'increments retain an arm pose target; grippers are absolute targets. '
            'Use physical waiting for an expected response with a completion '
            'condition, not to advance an unchanged scene. An empty array supplies '
            'no teacher steps; local continue can select the next objective.',
        items=dict(type='array',minItems=14,maxItems=14,items=dict(type='number'))),
        execution_note=s,bindings=dict(type='array',maxItems=8,items=binding),program=dict(anyOf=[program,dict(type='null')]),
        state_update=dict(anyOf=[dict(type='object'),dict(type='null')]),intent=intent_schema()))

class Model(Transport):
    def __init__(self,cfg):
        self.seek_frozen=os.environ.get('SEEK_FREEZE_LEARNING')=='1'
        self.seek_episode_adaptation=os.environ.get('SEEK_EPISODE_ADAPTATION')=='1'
        if self.seek_episode_adaptation and not self.seek_frozen:
            raise ValueError('Episode adaptation requires fixed incoming/output memory')
        self.seek_code_frozen=self.seek_frozen and not self.seek_episode_adaptation
        self.seek_input=Path(os.environ['SEEK_MEMORY']);self.seek_output=Path(os.environ['SEEK_OUTPUT'])
        self.seek_base=json.loads(self.seek_input.read_text());self.seek_input_sha=hashlib.sha256(self.seek_input.read_bytes()).hexdigest()
        self.seek_episode=int(os.environ['SEEK_EPISODE']);self.seek_trackers={}
        if self.seek_base.get('schema')!=SCHEMA:raise ValueError('SEEK requires seek-memory; use tools/migrate_plan_memory.py for explicit evidence migration')
        if self.seek_base.get('active'):require_compatible(self.seek_base,self.seek_base['active'])
        if any(e['episode']>=self.seek_episode for e in self.seek_base['episodes']):raise ValueError('Future episode in memory')
        super().__init__(cfg)
        c=self.dialogue_config
        previous=(f'Predict exactly {c.predict_horizon} steps per act; execute at most {c.execute_horizon}. '
            if c.fixed_predict_horizon else f'Predict 1..{c.predict_horizon} steps per act; execute at most {c.execute_horizon}. ')
        self.seek_prediction_contract=prediction_contract(c.predict_horizon)
        self.prompt=self.prompt.replace(previous,self.seek_prediction_contract)
        self.prompt=self.prompt.replace(
            'Control dual ARX X5 from RGB, task instruction and proprioception using exactly one act call per turn.',
            'Control dual ARX X5 through the action and program interfaces below.')
        self.prompt=assemble_instructions(ACTOR_ROLE, [
            ('Robot commands, executor limits and planning strategies', self.prompt),
            ('Executable program interface', ABI),
            ('Feedback and measurement reference', FEEDBACK_ABI),
            ('Reusable operations and evidence memory', OPERATION_GUIDANCE+GUIDANCE+LESSON_GUIDANCE+EXPERIENCE_GUIDANCE),
            ('Candidate batches and observation boundaries', CONDITION_GUIDANCE+HORIZON_GUIDANCE+PROGRESS_GUIDANCE),
            ('Evidence access and request layout', ACCESS_GUIDANCE+OBSERVATION_GUIDANCE+ACTION_GUIDANCE+CONTEXT_GUIDANCE),
        ])
        if self.seek_frozen:
            from .frozen_evaluation import GUIDANCE as FROZEN_GUIDANCE, EPISODE_ADAPTATION_GUIDANCE
            if self.seek_episode_adaptation:
                self.prompt=self.prompt.replace('Code persists across episodes; state and role bindings reset.',
                    'Code revisions apply only within this episode; the next episode restores the incoming memory.')
            self.prompt += '\n' + (EPISODE_ADAPTATION_GUIDANCE if self.seek_episode_adaptation else FROZEN_GUIDANCE)
        self.stable_prefix_hash=digest(self.prompt);self.frozen_tools=self._tools()

    def reset(self):
        for v in getattr(self,'seek_trackers',{}).values():v.close()
        super().reset()
        self.seek_bank=deepcopy(self.seek_base);self.seek_state={};self.seek_plan=None;self.seek_plan_number=0;self.seek_program=Program(DEFAULT_PROGRAM)
        archives=self.seek_bank.setdefault('observation_archives',{})
        archives.update(lineage_archives(self.seek_input))
        archives[str(self.seek_episode)]=str(self.episode_dir/'observations.jsonl')
        self.seek_experience=ExperienceControl(self.seek_bank,self.seek_episode,frozen=self.seek_frozen)
        self.seek_operations=OperationKnowledge(self.seek_bank,self.seek_episode)
        self.seek_trackers={};self.seek_bindings={};self.seek_roles={};self.seek_last_observed=-1;self.seek_context=None
        self.seek_pending=None;self.seek_history=[];self.seek_error=None;self.seek_force_model=False;self.seek_finalized=False
        self.seek_sources={};self.seek_teacher_step=None;self.seek_execution=None;self.seek_execution_number=0
        self.seek_lookups=0;self.seek_context_lookups=0;self.seek_lookup_history=[]
        self.seek_source_working_set=set()
        self.seek_teacher_proposal=None;self.seek_teacher_note=None
        self.seek_using_lessons=[];self.seek_lesson_reference=None
        self.seek_bank.setdefault('teacher_updates',[]);self.seek_bank.setdefault('errors',[])
        self.seek_requests_seen=set();self.seek_initial_saved=False
        self.seek_queries={};self.seek_query_pending=None;self.seek_terminal=None;self.seek_observers_seen=set()
        self.seek_observer_revision=0
        self.seek_online_contexts=[]
        self.seek_prediction_ready=False
        self.seek_progress_guard=ExecutionProgressGuard()
        self.seek_delegate=Program(DEFAULT_PROGRAM)
        self.seek_teacher_plan=None;self.seek_steps_since_teacher=0
        self.seek_local_key=None;self.seek_local_seen=set();self.seek_local_count=0
        self.seek_bank.setdefault('motion_evidence',[]);self.seek_bank.setdefault('effects',[])
        self.seek_motion_index=MotionEvidenceIndex()
        self.seek_bank.setdefault('episode_outcomes',[])
        self.seek_bank.setdefault('plans',[])
        if self.seek_bank.get('active'):
            self.seek_program=Program(next(p['source'] for p in self.seek_bank['programs'] if p['id']==self.seek_bank['active']))
        self.seek_main_program=self.seek_program

    def _checkpoint_memory(self,reason):
        started=time.perf_counter()
        if self.seek_frozen:
            from .frozen_evaluation import persist
            persist(self,atomic_json)
        else:
            atomic_json(self.seek_output,self.seek_bank)
        elapsed=time.perf_counter()-started
        self._log('seek_memory_checkpoints.jsonl',dict(reason=reason,step=self.step,
            elapsed_s=elapsed,bytes=self.seek_output.stat().st_size,
            transitions=len(self.seek_bank['transitions']),motion_evidence=len(self.seek_bank['motion_evidence'])))

    def _tools(self):
        schema=proposal_schema(self.dialogue_config.predict_horizon)
        schema['properties']['effects']=dict(type='array',items=dict(type='object',properties=dict(
            id=dict(type='string'),name=dict(type='string'),status=dict(type='string',enum=['pending','confirmed','refuted','unknown']),
            evidence_ids=dict(type='array',items=dict(type='string')),reason=dict(type='string')),
            required=['id','name','status','evidence_ids','reason'],additionalProperties=False))
        schema['properties']['candidate_guard']=guard_schema()
        schema['properties']['operations']=updates_schema()
        schema['properties']['operation']=invocation_schema()
        if self.seek_code_frozen:
            schema['properties']['program']={'type':'null'}
            schema['properties']['operations']['maxItems']=0
        schema['properties']['using_lessons']=dict(type='array',items=dict(type='string'))
        return [dict(type='function',name='act',strict=False,description='Provide SEEK actions, intent, candidate conditions and/or executable mechanism.',parameters=schema),recall_tool(),observation_tool()]

    def _configure_prediction_horizon(self):
        if self.seek_prediction_ready:return
        c=self.dialogue_config;native=self.archive[0].get('max_episode_steps')
        horizon=prediction_horizon(native) if c.experience_horizon_from_budget else c.predict_horizon
        contract=prediction_contract(horizon)
        self.prompt=self.prompt.replace(self.seek_prediction_contract,contract)
        self.seek_prediction_contract=contract
        self.dialogue_config=replace(c,predict_horizon=horizon,execute_horizon=horizon,fixed_predict_horizon=False)
        self.max_chunk=horizon;self.frozen_tools=self._tools();self.stable_prefix_hash=digest(self.prompt)
        self.seek_prediction_ready=True
        self._log('prediction_policy.jsonl',dict(native_step_limit=native,predict_max=horizon,execute_max=horizon,
            physical_feedback_every=1,source='native_step_budget' if c.experience_horizon_from_budget else 'explicit_configuration',
            length='model selects 0..maximum; remaining native steps may reduce the current maximum',
            target_semantics='SEEK fixed plan targets; no per-step rebasing of teacher intent'))

    def _motion(self,action,mode):
        self._convert_tensor([action],self.archive[self.step]['state'])
        if mode not in ('exact','prefix'):raise ProgramError('Use exact or prefix motion mode')
        if not self.archive[self.step].get('motion_query_available',False):
            raise ProgramError('Execution process did not advertise motion queries')
        intent={}
        plan=self.seek_plan
        if plan is not None and plan.active and np.allclose(action,plan.actions(self.archive[self.step]['state'])[0],rtol=0,atol=1e-12):
            intent=dict(plan_reference=plan.reference(),target=deepcopy(plan.targets[plan.cursor]))
        key=digest(dict(observation=f'e{self.seek_episode}:s{self.step}',action=action,mode=mode,**intent))
        if key in self.seek_queries:return deepcopy(self.seek_queries[key])
        query=dict(kind='motion_query',id=key,observation_id=f'e{self.seek_episode}:s{self.step}',
                   step=self.step,action=deepcopy(action),mode=mode,**intent)
        self.seek_query_pending=query
        raise CapabilityRequest(query)

    def notify_motion_result(self,obs):
        query=self.seek_query_pending
        if (query is None or obs.get('id')!=query['id'] or obs.get('observation_id')!=query['observation_id']
                or obs.get('step')!=self.step or obs.get('executed') is not False):
            raise ValueError('Motion measurement must match the outstanding observation/query')
        self.seek_queries[query['id']]=deepcopy(obs);self.seek_query_pending=None
        record=dict(deepcopy(obs),episode=self.seek_episode,robot=deepcopy(self.archive[self.step]['state']))
        self.seek_bank['motion_evidence'].append(record)
        self._log('seek_motion_evidence.jsonl',record)
        return dict(recorded=True,executed=False)

    def _joint_motion(self, joint_deltas, grippers):
        from .joint_motion import validate_proposal
        validate_proposal(joint_deltas, grippers)
        if not self.archive[self.step].get('motion_query_available',False):
            raise ProgramError('Execution process did not advertise motion queries')
        spec=dict(space='joint',joint_deltas=deepcopy(joint_deltas),grippers=deepcopy(grippers),
            limits=dict(translation_m=self.dialogue_config.max_translation_m,
                        rotation_rad=self.dialogue_config.max_rotation_rad))
        key=digest(dict(observation=f'e{self.seek_episode}:s{self.step}',**spec))
        if key in self.seek_queries:return deepcopy(self.seek_queries[key])
        query=dict(kind='motion_query',id=key,observation_id=f'e{self.seek_episode}:s{self.step}',
                   step=self.step,mode='exact',**spec)
        self.seek_query_pending=query
        raise CapabilityRequest(query)

    def _rgb_measure(self, function, *args):
        result = function(self.archive, self.step, *args)
        self._log('rgb_perception.jsonl', dict(operation=function.__name__, arguments=plain(args), result=plain(result)))
        return result

    def _capabilities(self,ctx=None,experience=None):
        return {'measure_rgb':lambda views,support_z,reference_pose=None:self._rgb_measure(measure_rgb,views,support_z,reference_pose,TOOL_GEOMETRY),
                'segment_rgb':lambda camera,text:self._rgb_measure(segment_rgb,camera,text),
                'locate_rgb_point':lambda samples,plane_z=None:self._rgb_measure(locate_rgb_point,samples,plane_z),
                'recall_experience':self._recall,
                'control_status':lambda:(experience or self.seek_experience).assess(ctx or self.seek_context),
                'realize_motion':self._motion,
                'realize_joint_motion':self._joint_motion,
                'measure_patch':lambda camera,box,reference:measure_patch(self.archive,self.step,camera,box,reference),
                'measure_region':lambda camera,box,lo,hi,depth_range_m=None,reference_pose=None:measure_region(self.archive,self.step,camera,box,lo,hi,depth_range_m,reference_pose)}

    def _effects(self,updates,source):
        observations={f'e{self.seek_episode}:s{s}' for s in self.archive}
        return update_effects(self.seek_bank['effects'],updates,self.seek_episode,observations,source)

    def _commit_effects(self,records):
        for record in records[len(self.seek_bank['effects']):]:self._log('seek_effects.jsonl',record)
        self.seek_bank['effects']=records

    def _observe_program(self,ctx):
        operation=self.seek_operations.record()
        abi=operation['abi'] if operation else self.seek_bank.get('program_abis',{}).get(self.seek_bank.get('active'))
        key=(self.step,self.seek_program.sha,self.seek_observer_revision if abi in ('seek-control-observed',PROGRAM_ABI) else 0)
        if key in self.seek_observers_seen or 'observe' not in self.seek_program.functions:return
        self.seek_observers_seen.add(key)
        before=deepcopy(self.seek_state)
        try:
            update=self.seek_program.observe(ctx,self.seek_state,{'measure_patch':self._capabilities()['measure_patch'],
                'measure_region':self._capabilities()['measure_region'],
                'measure_rgb':self._capabilities()['measure_rgb'],
                'segment_rgb':self._capabilities()['segment_rgb'],
                'locate_rgb_point':self._capabilities()['locate_rgb_point'],
                'recall_experience':self._recall})
            effects=self._effects(update.get('effects',[]),'mechanism_observation_assessment')
            if update.get('state_update') is not None:self.seek_state=deepcopy(update['state_update'])
            if update.get('effects'):self._commit_effects(effects)
            self._log('seek_observer.jsonl',dict(observation_id=ctx['id'],program=self._current_program_id(),
                state_before=before,state_after=self.seek_state,assessment=update))
        except (ProgramError,ValueError,TypeError,KeyError) as error:
            self.seek_error=dict(kind='observation_program_error',step=self.step,detail=str(error),executed=False)
            self.seek_force_model=True;self._remember_error(self.seek_error)
        ctx['effects']=current_effects(self.seek_bank['effects'],self.seek_episode)
        ctx['effect_counts']=effect_counts(ctx['effects']);ctx['last_error']=deepcopy(self.seek_error)

    def update_obs(self,obs):
        result=super().update_obs(obs)
        meta=obs.get('model_meta',{})
        self.archive[self.step]['motion_query_available']=bool(meta.get('motion_query_available'))
        self.archive[self.step]['commanded_grippers']=deepcopy(meta.get('commanded_grippers'))
        self.archive[self.step]['tracking_guard_enabled']=meta.get('tracking_guard_enabled',True)
        return result

    def _make_plan(self,actions,ctx,origin,guard=None):
        targets=plain(self._convert_tensor(actions,ctx['robot'])) if actions else []
        return CandidatePlan(f'e{self.seek_episode}:p{self.seek_plan_number+1}',ctx['id'],actions,targets,origin,
                             self.archive[self.step].get('tracking_guard_enabled',True),guard=guard)

    def _save_plan(self):
        record=self.seek_plan.snapshot()
        records=self.seek_bank['plans']
        index=next((i for i,r in enumerate(records) if r['id']==record['id']),len(records))
        if index==len(records):records.append(record)
        else:records[index]=record
        self._log('seek_plans.jsonl',record)

    def _teacher_execution(self):
        plan=self.seek_teacher_plan
        if plan is None:return None
        return dict(teacher_proposal_id=self.seek_teacher_proposal,anchor_observation_id=plan.anchor,
            candidate_rows_proposed=len(plan.rows),candidate_rows_executed=plan.cursor,
            candidate_status=plan.status,discarded_candidates=(plan.event or {}).get('discarded_candidates',0),
            physical_steps_since_reply=self.seek_steps_since_teacher,
            other_executed_steps=self.seek_steps_since_teacher-plan.cursor,
            scope='Actual receipts since this reply; local action source is not independent semantic decision-making')

    def _replace_plan(self,plan):
        if self.seek_plan is not None and self.seek_plan.active:
            self.seek_plan.close('replaced','A new policy decision replaces the remaining candidates');self._save_plan()
        self.seek_plan=plan;self.seek_plan_number+=1;self._save_plan()

    def _plan_context(self,current):
        plan=self.seek_plan
        robot=current['robot']
        if plan is not None and plan.active:
            reason=guard_reason(plan.guard,current)
            if reason is not None:
                plan.close('conditions_ended',reason);self._save_plan()
                self.seek_error=dict(kind='candidate_conditions_ended',step=self.step,detail=reason,
                    plan_reference=plan.reference(),executed=False)
                self._remember_error(self.seek_error)
        actions=plan.actions(robot) if plan is not None else []
        if actions:
            try:self._convert_tensor([actions[0]],robot)
            except ValueError as error:
                plan.close('invalidated',str(error));self._save_plan();actions=[]
                self.seek_error=dict(kind='plan_target_outside_step_bounds',step=self.step,detail=str(error),
                                   plan_reference=plan.reference(),executed=False)
                self._remember_error(self.seek_error)
        return dict(buffered_actions=actions,candidate_plan=plan.view() if plan is not None else None)

    def _measure_roles(self,trackers,bindings):
        row=self.archive[self.step];roles={}
        for name,tracker in trackers.items():
            camera=tracker.camera;height,width=frame(row,camera).shape[:2]
            result=tracker.observe(dict(row,instruction=tracker.instruction),allow_ground=False)
            selected={k:plain(result.get(k)) for k in ('valid','center','area','method','identity_reason','relation_evidence')}
            selected.update(camera=camera,generation=bindings[name]['generation'],description=bindings[name]['description'])
            selected.update(u=float(selected['center'][0])/width if selected.get('valid') else None,
                v=float(selected['center'][1])/height if selected.get('valid') else None,
                normalized_area=float(selected.get('area') or 0)/(width*height) if selected.get('valid') else None)
            if selected.get('valid') and camera in row.get('depth', {}):
                selected['surface'] = center_surface(row, camera, selected['center'])
            if 'mask' in result:
                selected['segment_geometry'] = segment_geometry(row, camera, result)
            roles[name]=selected
        return roles

    def _role_observations(self):
        if self.seek_last_observed==self.step:return
        self.seek_roles=self._measure_roles(self.seek_trackers,self.seek_bindings)
        self.seek_last_observed=self.step

    def _snapshot(self):
        self._role_observations();robot=deepcopy(self.archive[self.step]['state']);features={}
        for arm in ('left','right'):
            pose=robot[arm+'_ee_pose'];rot=Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_rotvec()
            for key,value in zip(('x','y','z','rx','ry','rz'),list(pose[:3])+rot.tolist()):features[arm+'.'+key]=float(value)
            target=float(robot.get(arm+'_gripper_target',robot[arm+'_ee_joint_state'])[0])
            features[arm+'.gripper']=target
            features[arm+'.gripper_target']=target
            aperture=robot.get(arm+'_gripper_aperture')
            features[arm+'.gripper_aperture']=float(aperture[0]) if aperture is not None else None
        for name,r in self.seek_roles.items():
            for key,field in (('u','u'),('v','v'),('area','normalized_area')):features['role.'+name+'.'+key]=r[field]
        features=measured_features(dict(features=features,roles=self.seek_roles))
        commanded=self.archive[self.step].get('commanded_grippers')
        if commanded is None:
            previous=next((r['execution_receipt']['command'] for r in reversed(self.ledger)
                if r['resulting_step']==self.step and r.get('execution_receipt')),None)
            commanded={arm:float((previous or robot)[arm+'_ee_joint_state'][0]) for arm in ('left','right')}
        return dict(id=f'e{self.seek_episode}:s{self.step}',episode=self.seek_episode,step=self.step,
            instruction=self.archive[self.step]['instruction'],robot=robot,roles=deepcopy(self.seek_roles),features=features,
            commanded_grippers=deepcopy(commanded))

    def _context(self):
        self._configure_prediction_horizon()
        current=self._snapshot()
        if self.seek_pending and self.step>self.seek_pending['before']['step']:
            p=self.seek_pending;start=p['before']['step'];receipts=[r for r in self.ledger if start<r['resulting_step']<=self.step]
            if len(receipts)!=1 or self.step!=start+1:raise RuntimeError('SEEK requires one real receipt per action')
            execution=deepcopy(receipts[0]['execution_receipt'])
            if execution['schema']!=RECEIPT_SCHEMA or execution['executed'] is not True or execution['resulting_step']!=self.step:
                raise ValueError('Expected one executed seek-execution-receipt receipt at the new step')
            for key,value in p['before']['robot'].items():
                if not np.array_equal(value,execution['state_before'][key]):raise ValueError('Receipt pre-state differs from dispatch observation')
            for key,value in current['robot'].items():
                if not np.array_equal(value,execution['state_after'][key]):raise ValueError('Receipt post-state differs from current observation')
            event=self.seek_plan.consume(execution)
            execution['controller_event']=event;self._save_plan()
            self.seek_state=p['next_state']
            record=dict(id=f'e{self.seek_episode}:t{self.step}',episode=self.seek_episode,start=start,end=self.step,
                before=p['before'],after=deepcopy(current),action=executed_delta(execution['command'],p['before']['robot']),
                requested_action=p['requested_action'],actual_command=execution['command'],
                plan_reference=p['plan_reference'],intended_target=p['intended_target'],
                execution_receipt=execution,controller_event=event,
                state_before=p['state_before'],state_after=deepcopy(p['next_state']),
                tracking=execution['tracking'],program=p['program'],source=p['source'],
                matched_fit_evidence=p.get('matched_fit_evidence',[]),
                experience_invocation=p.get('experience_invocation'),executed=True)
            record.update(teacher_proposal_id=p.get('teacher_proposal_id'),execution_note=p.get('execution_note'),
                operation_invocation=p.get('operation_invocation'),using_lessons=p.get('using_lessons',[]),
                lesson_reference=deepcopy(p.get('lesson_reference')))
            if p.get('motion_request') is not None:
                record['motion_request']=deepcopy(p['motion_request'])
            if 'motion_evidence_id' in p:
                record.update(motion_evidence_id=p['motion_evidence_id'],intended_action=p['intended_action'],queried_intent=p['queried_intent'])
            self.seek_history.append(record);self.seek_bank['transitions'].append(record)
            if self.seek_teacher_plan is not None:self.seek_steps_since_teacher+=1
            self.seek_experience.observe(record)
            self.seek_operations.observe(record)
            self.seek_sources[record['source']]=self.seek_sources.get(record['source'],0)+1
            self._log('seek_transitions.jsonl',record);self.seek_pending=None;self.seek_error=None
            self._execution_event('executed',record['action'],receipt=record['id'],receipt_id=record['id'],executed=True,
                plan_reference=p['plan_reference'],controller_event=event,tracking=execution['tracking'],
                source=record['source'],experience_invocation=record['experience_invocation'],
                detail=dict(type='action_executed',receipt_id=record['id'],tracking=deepcopy(execution['tracking']),controller_event=event))
            progress_event=self.seek_progress_guard.observe(record)
            if progress_event is not None:
                if self.seek_plan is not None and self.seek_plan.active:
                    self.seek_plan.close('supervision_review',progress_event);self._save_plan()
                self.seek_error=dict(kind='execution_progress_review',step=self.step,
                    detail=progress_event,executed=False,
                    reason='Pause further dispatch and obtain model review of actual motion evidence')
                self.seek_force_model=True
                self._remember_error(self.seek_error)
                self._log('execution_progress.jsonl',progress_event)
            self.seek_requests_seen.clear()
            self.seek_lookups=0;self.seek_context_lookups=0;self.seek_lookup_history=[]
        prior=[t for t in self.seek_bank['transitions'] if t['episode']<self.seek_episode]
        current.update(tool_geometry=deepcopy(TOOL_GEOMETRY),
            tool_base_relation=tool_base_relation(current['robot']),
            camera_calibration=deepcopy(self.archive[self.step].get('camera_calibration',{})),
            perception_mode='rgb_only',
            history=deepcopy(self.seek_history[-24:]),experience=sample_experience(prior),
            **self._plan_context(current),last_error=deepcopy(self.seek_error),
            last_execution=deepcopy(self.seek_execution),
            limits=dict(translation_m=self.dialogue_config.max_translation_m,rotation_rad=self.dialogue_config.max_rotation_rad),
            native_step_limit=self.archive[0].get('max_episode_steps'),
            prediction_budget=prediction_budget(self.dialogue_config.predict_horizon,self.step,self.archive[0].get('max_episode_steps')))
        effects=current_effects(self.seek_bank['effects'],self.seek_episode)
        current.update(capabilities=dict(motion_query=self.archive[self.step].get('motion_query_available',False),pixel_measurement=True,region_measurement=True,
            camera_depth=[],rgb_shape_measurement=True,rgb_point_triangulation=True),
            commanded_grippers=deepcopy(current['commanded_grippers']),
            motion_evidence=self.seek_motion_index.select(self.seek_bank['motion_evidence'],current['robot']),
            motion_evidence_total=len(self.seek_bank['motion_evidence']),effects=effects,effect_counts=effect_counts(effects))
        current['experience_control']=self.seek_experience.context()
        current['progress_guard']=self.seek_progress_guard.context()
        current['teacher_execution']=self._teacher_execution()
        self.seek_context=current
        if not self.seek_operations.started and self.seek_operations.bank['entry'] is not None:
            self._select_operation(dict(name=self.seek_operations.bank['entry'],parameters={}),current,
                'learned_entry','Start retained operation on fresh observations')
        current['operation_knowledge']=self.seek_operations.context()
        self._observe_program(current)
        if self.step==0 and not self.seek_initial_saved:
            atomic_json(self.episode_dir/'seek_initial_context.json',current);self.seek_initial_saved=True
        return current

    def _execution_event(self,kind,action,**fields):
        self.seek_execution_number+=1
        self.seek_execution=dict(id=f'e{self.seek_episode}:x{self.seek_execution_number}',kind=kind,
            step=self.step,action=deepcopy(action),**deepcopy(fields))
        self._log('seek_execution_events.jsonl',self.seek_execution)

    def _recall(self,query,evidence_ids,limit,filters=None,context_ref=None,path=(),offset=0,context_reads=None):
        if context_ref is not None or context_reads is not None:
            return ContextArchive(self.episode_dir/'context_index').read_request(
                query=query,evidence_ids=evidence_ids,limit=limit,filters=filters,
                context_ref=context_ref,path=path,offset=offset,context_reads=context_reads)
        return lookup(self.seek_bank,query,evidence_ids,limit,filters=filters)

    def _remember_error(self,error):
        record=dict(deepcopy(error),id=f'e{self.seek_episode}:error{len(self.seek_bank["errors"])+1}')
        self.seek_bank['errors'].append(record)
        self._log('seek_errors.jsonl',record);self._checkpoint_memory('error')

    def _preview_observer(self,update,ctx):
        observations={f'e{self.seek_episode}:s{s}' for s in self.archive}
        effects=update_effects(ctx.get('effects',[]),update.get('effects',[]),
            self.seek_episode,observations,'candidate_observation_assessment')
        result=deepcopy(ctx)
        result['effects']=current_effects(effects,self.seek_episode)
        result['effect_counts']=effect_counts(result['effects'])
        return result

    def _prepare_program(self,proposal,ctx,state=None,experience=None):
        if not isinstance(proposal,dict) or set(proposal)!={'source','description','evidence_ids'}:raise ProgramError('program requires source, description, evidence_ids')
        if not isinstance(proposal['description'],str) or not isinstance(proposal['evidence_ids'],list) or any(not isinstance(x,str) for x in proposal['evidence_ids']):
            raise ProgramError('program description must be text and evidence_ids a list of strings')
        known={ctx['id']}|{t['id'] for t in self.seek_bank['transitions']}|{t[s]['id'] for t in self.seek_bank['transitions'] for s in ('before','after')}
        known.update(r['id'] for r in self.seek_bank['motion_evidence'])
        known.update(r['observation_id'] for r in self.seek_bank['motion_evidence'] if 'observation_id' in r)
        known.update(available_ids(self.seek_bank))
        if not set(proposal['evidence_ids'])<=known:raise ProgramError('Unavailable evidence IDs')
        try:
            check=validate_candidate(proposal['source'],[ctx],self._capabilities(ctx,experience),state=state,
                decision_validator=lambda d,c:self._prepare_local_intent(d,c,experience),
                observer_validator=self._preview_observer)
        except CapabilityRequest:
            check=validate_candidate(proposal['source'],[])
            check['runtime_validation']='Deferred until actual motion query returns'
            self.seek_query_pending=None
        return dict(id=check['sha256'],**proposal,abi=PROGRAM_ABI,origin_episode=self.seek_episode,origin_step=self.step,status='provisional',validation=check)

    def _prepare_local_intent(self,decision,ctx,experience=None):
        return (experience or self.seek_experience).prepare(decision.get('intent'),ctx,
            has_actions=False,has_program=True)

    def _local_control_key(self):
        active=self.seek_experience.active
        intent=None if active is None else dict(spec={k:v for k,v in active['spec'].items() if k!='reason'},roles=active['roles'])
        operation=self.seek_operations.record()
        return digest(dict(state=self.seek_state,intent=intent,operation=operation['id'] if operation else None,
            effects=current_effects(self.seek_bank['effects'],self.seek_episode)))

    def _current_program_id(self):
        record=self.seek_operations.record()
        return record['id'] if record else self.seek_bank.get('active')

    def _select_operation(self,value,ctx,source,reason):
        if not self.seek_operations.select(value,ctx,source,reason):return False
        if self.seek_experience.active is not None:
            if self.seek_experience.active['status']=='active':
                self.seek_experience.active['status']='operation_changed'
            self.seek_experience.active=None
        if self.seek_plan is not None and self.seek_plan.active:
            self.seek_plan.close('operation_changed','Selected a different learned operation');self._save_plan()
        record=self.seek_operations.record()
        self.seek_program=Program(record['source']) if record else self.seek_main_program
        self.seek_state=deepcopy(value['parameters']) if value else {}
        self.seek_observer_revision+=1
        ctx.update(experience_control=self.seek_experience.context(),**self._plan_context(ctx))
        ctx['operation_knowledge']=self.seek_operations.context()
        self._log('seek_operations.jsonl',dict(step=self.step,source=source,reason=reason,
            operation=deepcopy(self.seek_operations.active)))
        return True

    def _commit_program(self,record):
        if self.seek_code_frozen:raise ProgramError('Frozen evaluation prohibits code installation')
        accept_program(self.seek_bank,record);self.seek_main_program=Program(record['source'])
        if self.seek_operations.active is None:self.seek_program=self.seek_main_program
        self._log('seek_programs.jsonl',dict(event='installed',episode=self.seek_episode,step=self.step,**record))

    def _install(self,proposal,ctx,state=None):
        self._commit_program(self._prepare_program(proposal,ctx,state))
        self._checkpoint_memory('program_install')

    def _stage_bindings(self,bindings):
        if not isinstance(bindings,list) or len(bindings)>8:raise ProgramError('At most eight role bindings per update')
        names=set()
        for item in bindings:
            if not {'name','description','bbox'}<=set(item) or set(item)-{'name','description','bbox','text','camera','prompt_mode'}:
                raise ProgramError('Binding needs name, description, bbox, with optional text, camera and prompt_mode')
            name=item['name'];box=item['bbox']
            if not isinstance(name,str) or not name.isidentifier() or name.startswith('_') or name in names:raise ProgramError('Unique plain role names required')
            names.add(name)
            if not isinstance(item['description'],str) or not item['description'].strip():raise ProgramError('Role description required')
            camera=item.get('camera','cam_head')
            if camera not in ('cam_head','cam_left_wrist','cam_right_wrist') or camera not in self.archive[self.step]['images']:
                raise ProgramError('Binding camera must select an available current RGB view')
            if item.get('text') is not None and (not isinstance(item['text'],str) or not item['text'].strip()):
                raise ProgramError('Binding text must be a nonempty visual concept or null')
            if 'prompt_mode' in item and (item['prompt_mode'] not in ('concept','joint') or not item.get('text')):
                raise ProgramError('Binding prompt_mode requires text and must be concept or joint')
            if len(box)!=4 or any(type(x) not in (int,float) or not np.isfinite(x) for x in box) or not(0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ProgramError('Normalized xyxy bbox required')
        staged={};definitions={}
        try:
            for item in bindings:
                name=item['name'];generation=self.seek_bindings.get(name,{}).get('generation',0)+1;camera=item.get('camera','cam_head')
                folder=self.episode_dir/'seek_vision'/f'{name}_{generation}_turn{self.turns}'
                description='Locate this visual role: '+item['description'];row=dict(self.archive[self.step],instruction=description)
                tracker=UnifiedVision({'task':'visual_role','localization':[]},row,lambda:'unavailable',self.effort,folder,
                    os.environ.get('SEEK_SIM_PYTHON','python'),model_request_budget=self.model_request_budget,camera=camera)
                staged[name]=tracker;definitions[name]=dict(item,generation=generation)
                path=folder/'binding.jpg';cv2.imwrite(str(path),frame(row,camera))
                ref=dict(image=str(path),camera=camera,bbox=item['bbox'],instruction=description,source_step=self.step,episode=self.seek_episode,
                    grounding_request=None,semantic_binding_source='act',id=f'e{self.seek_episode}:s{self.step}:{name}:{generation}')
                if 'text' in item:ref['text']=item['text']
                if 'prompt_mode' in item:ref['prompt_mode']=item['prompt_mode']
                if not tracker.add_reference(ref):raise ProgramError('Binding patch cannot be tracked')
                tracker.program=tracker.make_program([dict(id=ref['id'],episode=self.seek_episode,source_step=self.step)])
                tracker.program.update(evidence='current_act_visual_hypothesis',definition_origin='fixed_measurement_adapter',
                    learned_information='none; geometric template, flow and SAM are fixed measurement procedures')
                tracker.program_origin='current_act_binding';tracker.program_reused=False;tracker.references.append(ref)
            return staged,definitions,self._measure_roles(staged,definitions)
        except Exception:
            for tracker in staged.values():tracker.close()
            raise

    def _commit_bindings(self,staged,definitions,roles):
        old=[self.seek_trackers[name] for name in staged if name in self.seek_trackers]
        self.seek_trackers.update(staged);self.seek_bindings.update(definitions);self.seek_roles.update(roles)
        for item in definitions.values():self._log('seek_bindings.jsonl',dict(step=self.step,**item))
        for tracker in old:tracker.close()

    def _bind(self,bindings):
        self._role_observations();self._commit_bindings(*self._stage_bindings(bindings))

    def build_payload(self):
        ctx=self._context();active=next((p for p in self.seek_bank['programs'] if p['id']==self.seek_bank.get('active')),None)
        active=self.seek_operations.record() or active
        if self.seek_operations.record() is not None:
            self.seek_source_working_set.add(self.seek_operations.record()['id'])
        packet=dict(task=ctx['instruction'],context=ctx,mechanism_state=self.seek_state,active_program=active,
            task_lessons=lesson_context(self.seek_bank),
            improvement_history=improvement_context(self.seek_bank),
            prior_episode_results=self.seek_bank['episodes'],official_task_prompt=self.dialogue_config.official_task_prompt,
            camera_calibration=self.head_calibration_context)
        packet['experience_access']=dict(lookup_responses_remaining=max(0,LOOKUP_LIMIT-self.seek_lookups),
            context_read_responses_remaining=max(0,CONTEXT_LOOKUP_LIMIT-self.seek_context_lookups),
            visual_observation_episodes=sorted(int(e) for e in self.seek_bank['observation_archives']),
            stored_transitions=len(self.seek_bank['transitions']),stored_teacher_proposals=len(self.seek_bank['teacher_updates']))
        packet['learning_agenda']=learning_agenda(self.seek_bank)
        packet['learning_agenda']['current_episode']=online_learning_agenda(
            self.seek_bank,self.seek_episode,self.seek_online_contexts,ctx)
        steps=sorted({self.step,*([self.seek_teacher_step] if self.seek_teacher_step is not None else [])})
        pending=[e for e in ctx['effects'] if e['status'] in ('pending','unknown')]
        anchors=[int(i.split(':s')[-1]) for e in pending for i in e['evidence_ids']]
        steps=sorted(set(steps+anchors[-2:])|referenced_observations(
            self.seek_state,self.seek_episode,self.archive,self.step))
        inputs=action_inputs(action_view(packet),self._observations(steps))
        payload=with_cache_key(dict(model=self.model_name,instructions=self.prompt,input=inputs,tools=self.frozen_tools,
            tool_choice='required' if self.seek_lookups<LOOKUP_LIMIT or self.seek_context_lookups<CONTEXT_LOOKUP_LIMIT else dict(type='function',name='act'),
            parallel_tool_calls=False,store=False,reasoning=dict(effort=self.effort)))

        if self.dialogue_config.cache_mode=='implicit':
            payload['prompt_cache_options']={'mode':'implicit','ttl':'30m'}
        return payload

    def _ask(self):
        args=None;staged={};lookup_history=self.seek_lookup_history
        boundary=capture_boundary(self._context(),self.seek_state,self.seek_error,self.turns)
        self._log('seek_learning_contexts.jsonl',boundary)
        self.seek_online_contexts=(self.seek_online_contexts+[boundary])[-24:]
        self._log('seek_handoffs.jsonl',dict(step=self.step,reason=deepcopy(self.seek_error),
            teacher_execution=self._teacher_execution(),program=self._current_program_id()))
        self._checkpoint_memory('model_handoff')
        try:
            while True:
                payload=self.build_payload();payload['input'].extend(deepcopy(lookup_history))
                payload,context_metrics=bound_payload(payload,ContextArchive(self.episode_dir/'context_index'),
                    source_working_set=self.seek_source_working_set)
                self._log('context_budget.jsonl',dict(step=self.step,turn=self.turns+1,**context_metrics))
                self.payload_metrics(payload);response=self._request(payload);self.turns+=1
                self._log('exchanges.jsonl',dict(turn=self.turns,step=self.step,request=payload,response=response))
                calls=[o for o in response.get('output',[]) if o.get('type')=='function_call']
                if len(calls)!=1 or calls[0].get('name') not in ('recall_experience','inspect_observations'):break
                call=calls[0]
                if not isinstance(call.get('call_id'),str) or not call['call_id'] or call['call_id'] in self.seen_call_ids:
                    raise ProgramError('Experience lookup requires a new call_id')
                self.seen_call_ids.add(call['call_id'])
                try:lookup_arguments=json.loads(call['arguments'])
                except (ValueError,TypeError):lookup_arguments=None
                context_read=call['name']=='recall_experience' and is_context_lookup(lookup_arguments)
                if context_read:
                    if self.seek_context_lookups>=CONTEXT_LOOKUP_LIMIT:raise ProgramError('Deferred-context read allowance exhausted; use returned evidence or return act')
                    self.seek_context_lookups+=1
                else:
                    if self.seek_lookups>=LOOKUP_LIMIT:raise ProgramError('Experience lookup allowance exhausted; deferred-context reads use their separate allowance')
                    self.seek_lookups+=1
                visual_images=[]
                try:
                    if call['name']=='inspect_observations':
                        result,visual_images=inspect_observations(lookup_arguments,self.seek_episode,
                            self.episode_dir/'observations.jsonl',image_config=self.dialogue_config,
                            archives=self.seek_bank['observation_archives'])
                    else:
                        query=validate_lookup_arguments(lookup_arguments)
                        result=self._recall(**query)
                except (ValueError,TypeError,KeyError,IndexError,FileNotFoundError) as error:result=dict(error=str(error),executed=False)
                self.seek_source_working_set.update(operation_source_ids(result,self.seek_operations.bank['versions']))
                lookup_history.extend(canonical_tool_history([call]))
                lookup_history.append(dict(type='function_call_output',call_id=call['call_id'],output=json.dumps(result,ensure_ascii=False)))
                if visual_images:lookup_history.append(dict(role='user',content=visual_images))
                self._log('experience_lookups.jsonl',dict(step=self.step,turn=self.turns,call_id=call['call_id'],
                    lookup_kind='deferred_context' if context_read else 'observations' if call['name']=='inspect_observations' else 'experience',
                    paid_model_response=True,local_lookup=True,executed=False,arguments=lookup_arguments,result=result))
            calls=[o for o in response.get('output',[]) if o.get('type')=='function_call']
            if len(calls)!=1 or calls[0].get('name')!='act':raise ProgramError('Return exactly one act call')
            args=json.loads(calls[0]['arguments'])
            if self.seek_code_frozen:
                from .frozen_evaluation import validate_proposal
                validate_proposal(args)
            if set(args)-{'effects','candidate_guard','operations','operation','using_lessons'}!=set(proposal_schema(self.dialogue_config.predict_horizon)['properties']):raise ProgramError('act fields must match SEEK intent contract plus optional effects/candidate_guard')
            json.dumps(args,allow_nan=False)
            guard=validate_guard(args.get('candidate_guard'))
            actions=args['actions']
            maximum=prediction_budget(self.dialogue_config.predict_horizon,self.step,self.archive[0].get('max_episode_steps'))['max_actions_this_call']
            if not isinstance(actions,list) or len(actions)>maximum:raise ProgramError(f'actions must contain 0..{maximum} rows at the current step')
            if not actions and not args['bindings'] and args['program'] is None and args['state_update'] is None and args['intent'] is None and not args.get('effects') and not args.get('operations') and 'operation' not in args:
                raise ProgramError('Empty act must update intent, mechanism, binding or current evidence')
            current=args['state_update']
            if current is not None and not isinstance(current,dict):raise ProgramError('state_update must be object/null')
            ctx=self._context();plan=self._make_plan(actions,ctx,'teacher_previous_target',guard)
            staged,definitions,roles=self._stage_bindings(args['bindings'])
            effects=self._effects(args.get('effects',[]),'model_observation_assessment')
            ctx=deepcopy(ctx);ctx['roles'].update(roles)
            ctx['effects']=current_effects(effects,self.seek_episode)
            ctx['effect_counts']=effect_counts(ctx['effects'])
            for name,role in roles.items():
                for key,field in (('u','u'),('v','v'),('area','normalized_area')):ctx['features']['role.'+name+'.'+key]=role[field]
            ctx['features'].update(measured_features(dict(roles=roles)))
            ctx.update(buffered_actions=plan.actions(ctx['robot']),candidate_plan=plan.view())
            intent=self.seek_experience.prepare(args['intent'],ctx,has_actions=bool(actions),
                has_program=args['program'] is not None or self._current_program_id() is not None)
            record=None
            if args['program'] is not None:
                staged_experience=deepcopy(self.seek_experience);staged_experience.activate(intent,ctx)
                ctx['experience_control']=staged_experience.context()
                record=self._prepare_program(args['program'],ctx,state=current if current is not None else self.seek_state,
                    experience=staged_experience)
            known={ctx['id']}|available_ids(self.seek_bank)
            known.update(t['id'] for t in self.seek_bank['transitions'])
            known.update(t[s]['id'] for t in self.seek_bank['transitions'] for s in ('before','after'))
            operations=prepare_updates(args.get('operations',[]),known,self.seek_episode,self.step,
                self.seek_operations.bank['versions'])
            selection=self.seek_operations.prepare_invocation(args.get('operation'),operations)
            if selection is not None:
                preview_operations=deepcopy(self.seek_operations);preview_operations.install(operations)
                changed=preview_operations.select(selection,ctx,'candidate','Read-only observer validation')
                preview_ctx=deepcopy(ctx);preview_ctx['operation_knowledge']=preview_operations.context()
                preview_experience=deepcopy(self.seek_experience)
                if changed:
                    if preview_experience.active is not None and preview_experience.active['status']=='active':
                        preview_experience.active['status']='operation_changed'
                    preview_experience.active=None
                preview_experience.activate(intent,preview_ctx)
                preview_ctx['experience_control']=preview_experience.context()
                observer_state=invocation_state(selection,changed,self.seek_state,current)
                capabilities=self._capabilities(preview_ctx,preview_experience)
                update=Program(preview_operations.record()['source']).observe(preview_ctx,observer_state,
                    {key:capabilities[key] for key in ('measure_patch','measure_region','measure_rgb','segment_rgb','locate_rgb_point','recall_experience')})
                if update is not None:self._preview_observer(update,preview_ctx)

            old_state=deepcopy(self.seek_state)
            self._commit_bindings(staged,definitions,roles);staged={}
            if record is not None:self._commit_program(record)
            self.seek_operations.install(operations)
            self.seek_source_working_set.update(r['id'] for r in operations)
            operation_changed=False
            if 'operation' in args or record is not None:
                operation_changed=self._select_operation(selection,ctx,'teacher',args['execution_note'])
            self.seek_experience.activate(intent,ctx)
            self.seek_state=invocation_state(selection,operation_changed,self.seek_state,current)

            self.seek_observer_revision+=1
            self._commit_effects(effects)
            self._replace_plan(plan)
            self.seek_teacher_plan=plan;self.seek_steps_since_teacher=0
            self.seek_local_key=None;self.seek_local_seen.clear();self.seek_local_count=0
            self.seek_teacher_proposal=f'e{self.seek_episode}:teacher{self.turns}';self.seek_teacher_note=args['execution_note']
            lesson_ids={item['id'] for item in lesson_context(self.seek_bank)}
            claimed=args.get('using_lessons',[])
            self.seek_using_lessons=list(dict.fromkeys(i for i in claimed if isinstance(i,str) and i in lesson_ids)) if isinstance(claimed,list) else []
            self.seek_lesson_reference=dict(teacher_proposal_id=self.seek_teacher_proposal,
                observation_id=ctx['id'],using_lessons=deepcopy(self.seek_using_lessons),
                program=self._current_program_id(),
                operation_invocation=self.seek_operations.active['id'] if self.seek_operations.active else None,
                experience_invocation=self.seek_experience.active['id'] if self.seek_experience.active else None)
            if self.seek_operations.active is not None:
                self.seek_operations.active.setdefault('lesson_references',[]).append(deepcopy(self.seek_lesson_reference))
            self._log('seek_lesson_references.jsonl',self.seek_lesson_reference)
            self.seek_bank['teacher_updates'].append(dict(id=self.seek_teacher_proposal,episode=self.seek_episode,
                step=self.step,turn=self.turns,state_before=old_state,**deepcopy(args)))
            self.seek_teacher_step=self.step;self.seek_force_model=False
            self.seek_progress_guard.model_review_accepted()
            self._checkpoint_memory('teacher_update')
            self._log('seek_teacher.jsonl',dict(step=self.step,turn=self.turns,**args));return actions
        except (ValueError,TypeError,KeyError,IndexError) as e:
            self.seek_error=dict(kind='proposal_error',detail=str(e)[:1200],step=self.step,
                proposed_program=args.get('program') if isinstance(args,dict) else None,
                proposed_operations=args.get('operations') if isinstance(args,dict) else None,
                proposed_operation=args.get('operation') if isinstance(args,dict) else None,executed=False)
            self._remember_error(self.seek_error);self.seek_force_model=True;return None
        finally:
            for tracker in staged.values():tracker.close()

    def _execution_lesson_reference(self,source):
        reference=self.seek_lesson_reference
        if reference is None:return None
        if source in ('model','astra','teacher_buffer'):return deepcopy(reference)
        operation=self.seek_operations.active
        invocation=self.seek_experience.active
        if reference['program']!=self._current_program_id():return None
        if reference['operation_invocation'] is not None:
            if operation is None or operation['id']!=reference['operation_invocation']:return None
        elif operation is not None or reference['experience_invocation']!=(invocation['id'] if invocation else None):
            return None
        return deepcopy(reference)

    def _dispatch(self,action,ctx,next_state,source,reason,plan=None,query_id=None):
        if query_id is not None:
            chosen=self.seek_queries.get(query_id)
            target=self._convert_tensor([action],ctx['robot'])[0]
            if (chosen is None or chosen['observation_id']!=ctx['id']
                    or chosen['result']['status']!='feasible'
                    or not np.allclose(chosen['result']['action'],action,rtol=0,atol=1e-12)
                    or any(not np.array_equal(chosen['result']['command'][k],v) for k,v in target.items())):
                raise ProgramError('query_id must select this action at the current observation')
        if plan is None:
            plan=self._make_plan([action],ctx,'program_current_observation');self._replace_plan(plan)
        converted=deepcopy(plan.targets[plan.cursor])
        reference=plan.reference()
        lesson_reference=self._execution_lesson_reference(source)
        before={k:deepcopy(ctx[k]) for k in ('id','episode','step','instruction','robot','roles','features','commanded_grippers')}
        self.seek_pending=dict(before=before,action=deepcopy(action),state_before=deepcopy(self.seek_state),
            motion_request=selected_motion_request(self.seek_queries.values(),action,ctx['id'],query_id),
            requested_action=deepcopy(plan.rows[plan.cursor]),plan_reference=reference,intended_target=deepcopy(converted),
            next_state=deepcopy(next_state),program=self._current_program_id(),source=source,
            operation_invocation=self.seek_operations.active['id'] if self.seek_operations.active else None,
            using_lessons=deepcopy(lesson_reference['using_lessons']) if lesson_reference else [],
            lesson_reference=lesson_reference,
            experience_invocation=self.seek_experience.active['id'] if self.seek_experience.active else None,
            teacher_proposal_id=self.seek_teacher_proposal if source in ('model','astra','teacher_buffer') else None,
            execution_note=self.seek_teacher_note if source in ('model','astra','teacher_buffer') else None,
            matched_fit_evidence=[])
        self._log('seek_dispatch.jsonl',dict(step=self.step,source=source,program=self._current_program_id(),
            action=action,reason=reason,plan_reference=reference,target=converted))
        packet=dict(kind='planned_action',reference=reference,action=converted)
        proof=next((q for q in reversed(list(self.seek_queries.values()))
            if q['observation_id']==ctx['id'] and q['result']['status']=='feasible'
            and (q['id']==query_id if query_id is not None else q.get('space')!='joint')
            and np.allclose(q['result']['action'],action,rtol=0,atol=1e-12)
            and all(np.array_equal(q['result']['command'][key],value) for key,value in converted.items())),None)
        if proof:
            self.seek_pending['motion_evidence_id']=proof['id']
            self.seek_pending['intended_action']=deepcopy(proof['requested_action'])
            self.seek_pending['queried_intent']=dict(plan_reference=proof.get('plan_reference'),target=proof.get('target'))
            if proof.get('space')=='joint':
                self.seek_pending['queried_intent'].update(space='joint',
                    joint_deltas=deepcopy(proof['requested_joint_deltas']),
                    grippers=deepcopy(proof['requested_grippers']))
            packet['query_id']=proof['id']
        return packet

    def get_action(self):
        if self.seek_terminal:return deepcopy(self.seek_terminal)
        try:return self._get_action()
        except CapabilityRequest as request:return request.query
        except MechanismBlocked as blocked:
            self.seek_terminal=dict(kind='episode_outcome',termination='blocked',reason=str(blocked),
                                 episode=self.seek_episode,step=self.step)
            self.seek_bank['episode_outcomes'].append(deepcopy(self.seek_terminal))
            self.finalize()
            return deepcopy(self.seek_terminal)

    def _get_action(self):
        while True:
            ctx=self._context()
            if self.seek_program is not None and not self.seek_force_model:
                try:
                    capabilities=self._capabilities()
                    def assess_control():
                        result=self.seek_experience.assess(ctx)
                        self.seek_experience.last_decision=deepcopy(result)
                        return result
                    capabilities['control_status']=assess_control
                    before_local=self._local_control_key()
                    d=self.seek_program.run(ctx,self.seek_state,capabilities)
                    effects=self._effects(d.get('effects',[]),'mechanism_observation_assessment')
                    intent=self._prepare_local_intent(d,ctx)
                    selection=self.seek_operations.prepare_invocation(d.get('operation'))
                    self._log('seek_decisions.jsonl',dict(step=self.step,program=self._current_program_id(),decision=d))
                    operation_changed=False
                    if 'operation' in d:
                        operation_changed=self._select_operation(selection,ctx,'mechanism',d.get('reason','Learned operation transition'))
                    if self.seek_experience.activate(intent,ctx,source='mechanism',program=self._current_program_id()):
                        self.seek_observer_revision+=1
                        if self.seek_plan is not None and self.seek_plan.active:
                            self.seek_plan.close('objective_changed','Local program selected a new objective; previous candidates are not evidence for it')
                            self._save_plan()
                        ctx.update(experience_control=self.seek_experience.context(),**self._plan_context(ctx))
                    if d.get('effects'):
                        if effects!=self.seek_bank['effects']:self.seek_observer_revision+=1
                        self._commit_effects(effects)
                    next_state=invocation_state(selection,operation_changed,self.seek_state,d.get('state_update'))
                    if next_state!=self.seek_state:self.seek_observer_revision+=1
                    self.seek_state=next_state
                    if d['kind']=='continue':
                        local_key=ctx['id']
                        if self.seek_local_key!=local_key:
                            self.seek_local_key=local_key;self.seek_local_seen.clear();self.seek_local_count=0
                        self.seek_local_seen.add(before_local);self.seek_local_count+=1
                        if self._local_control_key() in self.seek_local_seen or self.seek_local_count>16:
                            raise ProgramError('Local continuation made no new state, cycled, or exceeded its scheduling budget; repair locally executable transitions')
                        continue
                    delegated=d['kind']=='delegate'
                    if delegated:
                        d=self.seek_delegate.run(ctx,self.seek_state,capabilities)
                        self._log('seek_delegation.jsonl',dict(step=self.step,decision=d))
                        self._log('seek_decisions.jsonl',dict(step=self.step,program=self._current_program_id(),
                            decision=d,decision_route='explicit_local_delegation'))
                    if d['kind']=='blocked':self._blocked(d.get('reason','Missing external capability'))
                    if d['kind']=='act':
                        after=d.get('on_execution')
                        source='mechanism' if self._current_program_id() and not delegated else 'motion_adapter'
                        return self._dispatch(d['action'],ctx,self.seek_state if after is None else after,source,d.get('reason',''),query_id=d.get('query_id'))
                    if d['kind']=='cached' and ctx['buffered_actions']:
                        action=ctx['buffered_actions'][0];after=d.get('on_execution')
                        source='model' if self.seek_teacher_step==self.step else 'teacher_buffer'
                        return self._dispatch(action,ctx,self.seek_state if after is None else after,source,d.get('reason',''),plan=self.seek_plan)
                    self.seek_error=dict(kind='mechanism_request',detail=d.get('reason','Buffer empty'),diagnosis=d.get('diagnosis'),step=self.step)
                except (ProgramError,ValueError,TypeError,KeyError) as e:
                    self.seek_error=dict(kind='mechanism_runtime_error',detail=str(e),program=self._current_program_id(),step=self.step,executed=False)
                    self._remember_error(self.seek_error)
            ctx=self._context()
            key=digest(dict(observation=ctx['id'],features=ctx['features'],robot=ctx['robot'],
                program=self.seek_program.sha,state=self.seek_state,actions=ctx['buffered_actions'],
                intent=self.seek_experience.active['spec'] if self.seek_experience.active else None,
                bindings=[{k:b[k] for k in ('name','description','bbox','text','camera','prompt_mode') if k in b} for b in self.seek_bindings.values()],
                execution=semantic_feedback(self.seek_execution),request=semantic_feedback(self.seek_error),
                motion=sorted({digest(semantic_feedback(r)) for r in ctx['motion_evidence']}),
                effects=current_effects(self.seek_bank['effects'],self.seek_episode)))
            if key in self.seek_requests_seen:self._blocked('Repeated request with unchanged observation, mechanism, state and feedback')
            self.seek_requests_seen.add(key)

            self._ask()

    def _blocked(self,reason):
        self._log('seek_blocked.jsonl',dict(step=self.step,reason=reason,last_execution=self.seek_execution,last_error=self.seek_error))
        raise MechanismBlocked(reason)

    def notify_action_rejection(self,obs):
        if obs.get('step')!=self.step or not isinstance(obs.get('event'),dict) or self.seek_pending is None:
            raise ValueError('Execution rejection requires the current step, event and pending command')
        event=deepcopy(obs['event']);rejected=deepcopy(self.seek_pending['action'])
        reference=self.seek_pending['plan_reference']
        if obs.get('reference')!=reference:raise ValueError('Rejection must identify the pending candidate')
        source=self.seek_pending['source'];invocation=self.seek_pending.get('experience_invocation')
        self.seek_plan.close('rejected',plain(event));self._save_plan()
        self.seek_pending=None;self.seek_force_model=False
        self._execution_event('rejected',rejected,detail=plain(event),receipt_id=None,executed=False,plan_reference=reference,
                              source=source,experience_invocation=invocation)
        record=dict(deepcopy(self.seek_execution),episode=self.seek_episode,observation_id=f'e{self.seek_episode}:s{self.step}',
                    source='execution_rejection',robot=deepcopy(self.archive[self.step]['state']))
        self.seek_bank['motion_evidence'].append(record)
        self._log('seek_motion_evidence.jsonl',record)
        self.seek_error=None
        self._remember_error(dict(kind='execution_rejected',step=self.step,detail=plain(event),executed=False))

    def finalize(self):
        if self.seek_finalized:return dict(step=self.step,turns=self.turns)
        if self.step is not None:self._context()
        self.seek_experience.finish()
        if self.seek_context is not None:self.seek_operations.finish(self.seek_context)
        if self.seek_plan is not None and self.seek_plan.active:
            self.seek_plan.close('episode_ended','Episode finalized before the remaining candidates executed');self._save_plan()
        if hashlib.sha256(self.seek_input.read_bytes()).hexdigest()!=self.seek_input_sha:raise RuntimeError('Input memory modified')
        self._checkpoint_memory('finalize')
        for tracker in self.seek_trackers.values():tracker.close()
        report=dict(episode=self.seek_episode,step=self.step,turns=self.turns,sources=self.seek_sources,active_program=self.seek_bank.get('active'),active_operation=deepcopy(self.seek_operations.active),transitions=len(self.seek_history))
        self._log('seek_episode.jsonl',report);self.seek_finalized=True;return report
