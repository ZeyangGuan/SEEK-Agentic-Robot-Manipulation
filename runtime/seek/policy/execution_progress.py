from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


GUIDANCE = '''
Execution supervision is independent of your program and plan IDs. A completed
RPC/receipt is not evidence of physical progress. context.progress_guard reports
the supervisor limits, actual receipt IDs and the latest model-review event.
Eight sustained motion commands with no measurable pose change or twelve-step
back-and-forth motion with little net displacement can require a new model assessment.
Twelve sustained commands whose signed motion along the commanded direction and
reduction of command-target error are both below 10% of the requested motion also require assessment, unless
an actively commanded pose channel on either arm is making non-oscillatory progress.
This also defers no-motion and oscillation reviews: one arm's small holding corrections
must not interrupt the other arm's progressing maneuver. Once that progress ends,
the rolling receipt window can reveal a persistent stall. Translation and rotation
use separate units; uncommanded drift is not progress.
The non-oscillation check includes the whole receipt window, including drift
between corrections; repeatedly restoring a pose is not cumulative progress.
Targets are compared before/after each receipt, not assumed fixed across commands.
For a dispatched numerical prefix, supervision also compares actual response
with the original unshortened query, using the same twelve-receipt/10% limits.
low_intended_motion_response distinguishes a well-tracked tiny accepted command
from useful response to that original request. Its accepted_channels and
intended_channels keep both scales visible. An active progressing channel on
either arm still defers review. Deliberately small requests remain small requests.
motion_request links only the selected current-observation query; rounded_action
is a diagnostic association, never permission to reuse an approximate IK solve.
Return query results unchanged with query_id where possible to preserve provenance.
Review the current route, posture or intended slow contact process; no direction,
height change or task stage is prescribed by this diagnostic.
Pending plan
candidates are discarded; the episode and actual receipts remain valid.
Inspect measured motion, command scale, constraints, targets and fresh images.
Revise the method or explain why bounded waiting/repetition is necessary. These
events are numerical supervision evidence, not contact or task-success labels.
Stationary observation and gripper holding are not classified as stalled arm
motion. There is no unconditional step-count review: measured progress may
continue without a model call, within the environment's native episode limit.
'''


@dataclass(frozen=True)
class ProgressLimits:
    no_progress_steps: int = 8
    low_response_steps: int = 12
    low_response_fraction: float = .1
    oscillation_steps: int = 12
    command_position_m: float = .00025
    command_rotation_rad: float = .0025
    observed_position_m: float = .0001
    observed_rotation_rad: float = .001
    oscillation_position_path_m: float = .004
    oscillation_rotation_path_rad: float = .04
    oscillation_net_fraction: float = .2
    oscillation_reversals: int = 4


def _rotation(pose):
    return Rotation.from_quat(np.asarray(pose, float)[[4, 5, 6, 3]])


def _distance(before, after):
    return dict(position_m=float(np.linalg.norm(np.asarray(after[:3])-before[:3])),
                rotation_rad=float((_rotation(after)*_rotation(before).inv()).magnitude()))


def _path(vectors, epsilon):
    significant=[np.asarray(v, float) for v in vectors if np.linalg.norm(v)>epsilon]
    reversals=sum(float(np.dot(a,b)) < -.5*float(np.linalg.norm(a)*np.linalg.norm(b))
                  for a,b in zip(significant,significant[1:]))
    return float(sum(np.linalg.norm(v) for v in vectors)), reversals


class ExecutionProgressGuard:
    def __init__(self, limits=None):
        self.limits=limits or ProgressLimits()
        if any(not math.isfinite(v) or v<=0 for v in asdict(self.limits).values()):
            raise ValueError('Progress supervision limits must be finite and positive')
        self.rows=deque(maxlen=max(self.limits.no_progress_steps,self.limits.oscillation_steps,self.limits.low_response_steps))
        self.steps=0
        self.last_step=None
        self.pending=False
        self.last_event=None
        self.events=0

    def context(self):
        return dict(limits=asdict(self.limits),steps_since_model_review=self.steps,
                    model_review_required=self.pending,last_event=deepcopy(self.last_event),
                    scope='Observed numerical motion; not verified task progress or success')

    def model_review_accepted(self):
        self.rows.clear()
        self.steps=0
        self.pending=False

    def _event(self, kind, rows, **evidence):
        self.events+=1
        self.pending=True
        self.last_event=dict(id='progress-review-'+str(self.events),kind=kind,
            first_step=rows[0]['step'],last_step=rows[-1]['step'],
            receipt_ids=[r['receipt'] for r in rows],steps_since_model_review=self.steps,
            requires_model_review=True,task_failure=False,**evidence)
        return deepcopy(self.last_event)

    def _channels(self, rows, arm, intended=False):
        limits=self.limits
        channels={}
        for field,unit,minimum in (
            ('translation','position_m',limits.command_position_m),
            ('rotation','rotation_rad',limits.command_rotation_rad)):
            selected=[r['intended_arms'][arm] if intended else r['arms'][arm] for r in rows]
            active=[r for r in selected if r['command'][unit]>=minimum]
            if not active:
                continue
            requested=sum(r['command'][unit] for r in active)
            measured=sum(float(np.linalg.norm(r[field])) for r in active)
            directed=sum(float(np.dot(r[field],r['command_'+field]))/r['command'][unit]
                         for r in active)
            reduction=sum(r['command'][unit]-r['target_error_after'][unit] for r in active)
            channels[unit]=dict(active_steps=len(active),requested=requested,
                measured_path=measured,directed_motion=directed,command_error_reduction=reduction,
                measured_net=float(np.linalg.norm(np.sum([r[field] for r in active],axis=0))),
                motion_fraction=measured/requested,directed_motion_fraction=directed/requested,
                error_reduction_fraction=reduction/requested)
        return channels

    def _below_response(self, value):
        return value<self.limits.low_response_fraction and not math.isclose(
            value,self.limits.low_response_fraction,rel_tol=1e-9)

    def _commanded_progress(self, rows, intended=False):
        limits=self.limits
        for arm in ('left','right'):
            actual=[r['arms'][arm] for r in rows]
            net=_distance(actual[0]['before'],actual[-1]['after'])
            for unit,c in self._channels(rows,arm,intended).items():
                minimum=(limits.observed_position_m if unit=='position_m'
                         else limits.observed_rotation_rad)
                field='translation' if unit=='position_m' else 'rotation'
                path=sum(float(np.linalg.norm(r[field])) for r in actual)
                if (c['measured_net']>minimum
                        and net[unit]>minimum
                        and net[unit]>path*limits.oscillation_net_fraction
                        and (not self._below_response(c['directed_motion_fraction'])
                             or not self._below_response(c['error_reduction_fraction']))):
                    return True
        return False

    def _low_response(self, arm):
        limits=self.limits
        rows=list(self.rows)[-limits.low_response_steps:]
        if len(rows)!=limits.low_response_steps:
            return None
        channels=self._channels(rows,arm)
        weak=[unit for unit,c in channels.items() if c['active_steps']==len(rows)
              and self._below_response(c['directed_motion_fraction'])
              and self._below_response(c['error_reduction_fraction'])]
        if weak and not self._commanded_progress(rows):
            return self._event('low_motion_response',rows,arm=arm,weak_channels=weak,
                channels=channels,
                interpretation='Sustained commanded motion has little response along its direction or '
                    'command-target error reduction; reassess reach, posture and constraints. '
                    'This is not a collision, grasp-loss or task-failure diagnosis.')
        return None

    def _low_intended_response(self, arm):
        rows=list(self.rows)[-self.limits.low_response_steps:]
        if (len(rows)!=self.limits.low_response_steps
                or not all(r.get('motion_request') and r['motion_request']['mode']=='prefix'
                           and 0<r['motion_request']['fraction']<1 for r in rows)):
            return None
        channels=self._channels(rows,arm,intended=True)
        weak=[unit for unit,c in channels.items() if c['active_steps']==len(rows)
              and self._below_response(c['directed_motion_fraction'])
              and self._below_response(c['error_reduction_fraction'])]
        if weak and not self._commanded_progress(rows,intended=True):
            return self._event('low_intended_motion_response',rows,arm=arm,
                weak_channels=weak,intended_channels=channels,
                accepted_channels=self._channels(rows,arm),
                motion_requests=[deepcopy(r['motion_request']) for r in rows],
                interpretation='Actual motion is small relative to sustained unshortened requests. '
                    'Reassess the current route/posture or explain the intended slow process; '
                    'this does not establish collision, unreachable goals or task failure.')
        return None

    def observe(self, transition):
        receipt=transition.get('execution_receipt',{})
        if receipt.get('executed') is not True:
            return None
        step=receipt['resulting_step']
        if self.last_step is not None and step<=self.last_step:
            return None
        self.last_step=step
        self.steps+=1
        request=transition.get('motion_request')
        row=dict(step=step,receipt=transition['id'],arms={},intended_arms={},
                 motion_request=deepcopy(request))
        for arm in ('left','right'):
            key=arm+'_ee_pose'
            before,after,target=(receipt[k][key] for k in ('state_before','state_after','command'))
            row['arms'][arm]=dict(before=list(before),after=list(after),
                command=_distance(before,target),
                command_translation=(np.asarray(target[:3])-before[:3]).tolist(),
                command_rotation=(_rotation(target)*_rotation(before).inv()).as_rotvec().tolist(),
                target_error_after=_distance(after,target),
                translation=(np.asarray(after[:3])-before[:3]).tolist(),
                rotation=(_rotation(after)*_rotation(before).inv()).as_rotvec().tolist())
            intended=deepcopy(row['arms'][arm])
            if request is not None:
                offset=0 if arm=='left' else 7
                action=np.asarray(request['requested_action'][offset:offset+6],float)
                rotation=Rotation.from_rotvec(action[3:])*_rotation(before)
                quaternion=rotation.as_quat()
                target=[*(np.asarray(before[:3])+action[:3]),
                        quaternion[3],*quaternion[:3]]
                intended.update(command=_distance(before,target),
                    command_translation=action[:3].tolist(),command_rotation=action[3:].tolist(),
                    target_error_after=_distance(after,target))
            row['intended_arms'][arm]=intended
        self.rows.append(row)
        if self.pending:
            return None
        limits=self.limits
        for arm in ('left','right'):
            window=list(self.rows)[-limits.no_progress_steps:]
            if len(window)==limits.no_progress_steps:
                active=all(r['arms'][arm]['command']['position_m']>=limits.command_position_m or
                           r['arms'][arm]['command']['rotation_rad']>=limits.command_rotation_rad for r in window)
                poses=[window[0]['arms'][arm]['before']]+[r['arms'][arm]['after'] for r in window]

                spans=[_distance(a,b) for i,a in enumerate(poses) for b in poses[i+1:]]
                position=max(d['position_m'] for d in spans)
                rotation=max(d['rotation_rad'] for d in spans)
                if (active and position<=limits.observed_position_m and rotation<=limits.observed_rotation_rad
                        and not self._commanded_progress(window)):
                    return self._event('no_measured_motion',window,arm=arm,
                        observed_position_span_m=position,observed_rotation_span_rad=rotation,
                        latest_command=window[-1]['arms'][arm]['command'])
            window=list(self.rows)[-limits.oscillation_steps:]
            if len(window)==limits.oscillation_steps:
                net=_distance(window[0]['arms'][arm]['before'],window[-1]['arms'][arm]['after'])
                for field,unit,minimum,epsilon,command_min in (
                    ('translation','position_m',limits.oscillation_position_path_m,limits.observed_position_m,limits.command_position_m),
                    ('rotation','rotation_rad',limits.oscillation_rotation_path_rad,limits.observed_rotation_rad,limits.command_rotation_rad)):
                    path,reversals=_path([r['arms'][arm][field] for r in window],epsilon)
                    active=sum(r['arms'][arm]['command'][unit]>=command_min for r in window)
                    if (active>=len(window)//2 and path>=minimum and net[unit]<=path*limits.oscillation_net_fraction
                            and reversals>=limits.oscillation_reversals
                            and not self._commanded_progress(window)):
                        return self._event('oscillation_suspected',window,arm=arm,channel=field,
                            measured_path=path,measured_net=net[unit],reversals=reversals,
                            interpretation='Repeated reversals with little net motion; model must assess task relevance')
            response_event=self._low_response(arm)
            if response_event is not None:
                return response_event
            intended_event=self._low_intended_response(arm)
            if intended_event is not None:
                return intended_event
        return None
