import base64
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4
import cv2
import numpy as np
import requests
from .task_scope import task_binding, same_target
from .model_request_budget import ModelRequestBudget, request_budget
from . import transport_retry


PROGRAM_SCHEMA = 'seek-localization-program'
PROGRAM_STEPS = ('bind_task_target', 'match_target_reference', 'track_target_reference',
                 'ground_task_target', 'segment_target_reference')


def frame(row, camera='cam_head'):
    raw = base64.b64decode(row['images'][camera].split(',')[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def template_match(image, template):
    if (image is None or template is None or image.size == 0 or template.size == 0
            or image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]):
        return np.zeros(2), -1., 0.
    correlation = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(correlation)
    h, w = template.shape[:2]
    other = correlation.copy(); x, y = location
    other[max(0,y-h):y+h+1, max(0,x-w):x+w+1] = -1
    second = float(other.max())
    return np.array([x+w/2, y+h/2]), float(score), float(score-second)


class UnifiedVision:
    def __init__(self, memory, row0, token_provider, effort, folder, sim_python, model_request_budget=None, camera='cam_head'):
        if camera not in ('cam_head','cam_left_wrist','cam_right_wrist'):
            raise ValueError('Select a calibrated RGB camera')
        self.camera = camera
        self.model_request_budget = model_request_budget or ModelRequestBudget(folder=folder)
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.token_provider = token_provider; self.effort = effort; self.sim_python = sim_python
        self.instruction = row0['instruction']; self.references = []; self.templates = []
        self.task = memory.get('task', 'general_pickup')
        self.binding = task_binding(self.instruction, self.task)
        self.programs = []
        self.grounding_calls = 0; self.grounding_seconds = 0.; self.grounding_completed = 0
        self.worker = None; self.error_file = None
        self.center = None; self.size = None; self.previous = None; self.active = None
        self.last_step = -1; self.last_result = None
        self.temporal_id = uuid4().hex
        self.temporal_seeded = False
        for reference in memory.get('localization', []):
            if same_target(reference.get('instruction'), self.instruction, self.task):
                self.add_reference(reference)
        programs = [p for p in memory.get('programs', []) if self.valid_program(p)]
        self.program = dict(programs[0]) if programs else None
        self.program_origin = 'persisted_program' if self.program else 'bootstrap'
        if self.program is None:
            sources = []
            for reference in memory.get('localization', []):
                if (reference.get('visual_validation') in ('model_visible_target', 'astra_visible_target')
                        and reference.get('camera','cam_head') == self.camera
                        and reference.get('instruction')
                        and self.reference_asset(reference) is not None):
                    sources.append({k: reference.get(k) for k in
                                    ('id', 'episode', 'source_step', 'instruction')})
            if sources:
                self.program = self.make_program(sources, migration=True)
                self.programs.append(self.program)
                self.program_origin = 'migrated_prior_experience'
                self.log('program_registered', program=self.program)
        self.program_reused = self.program is not None
        self.log('program_loaded', program_id=self.program['id'] if self.program else None,
                 origin=self.program_origin, task=self.task, binding=self.binding,
                 compatible_templates=len(self.templates),
                 source_episode=self.program.get('source_episode', self.program.get('episode'))
                                if self.program else None)

    def valid_program(self, program):
        return (program.get('schema') == PROGRAM_SCHEMA and program.get('task') == self.task
                and program.get('camera','cam_head') == self.camera
                and program.get('operation') == 'locate_task_target'
                and tuple(program.get('steps', ())) == PROGRAM_STEPS
                and bool(program.get('source_references')))

    def make_program(self, sources, migration=False):
        return dict(id='seek-localization:'+self.task, schema=PROGRAM_SCHEMA, camera=self.camera,
                    kind='perception_program', task=self.task, operation='locate_task_target',
                    steps=list(PROGRAM_STEPS), source_references=sources,
                    source_episode=sources[0].get('episode'), source_step=sources[0].get('source_step'),
                    migration=migration,
                    evidence='persisted_model_grounded_references' if migration else
                             'observed_model_grounding_and_valid_identity_segmentation',
                    definition_origin='registered_generic_primitives',
                    learned_information='validated input/output binding and interaction provenance',
                    unseen_target_semantics='requires_model_when_no_compatible_visual_reference')

    @staticmethod
    def reference_asset(reference):
        if not reference.get('image') or 'bbox' not in reference:
            return None
        source = cv2.imread(reference['image'])
        if source is None: return None
        h,w = source.shape[:2]
        box = np.asarray(reference['bbox'], dtype=float)
        if box.shape != (4,) or not np.all(np.isfinite(box)): return None
        x1,y1,x2,y2 = (box*[w,h,w,h]).astype(int)
        if not (0<=x1<x2<=w and 0<=y1<y2<=h): return None
        template = source[y1:y2,x1:x2].copy()
        if min(template.shape[:2])<4: return None
        return dict(reference=reference, template=template, size=np.array([x2-x1,y2-y1],float))

    def log(self, event, **data):
        with (self.folder/'perception.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event, **data))+'\n')

    def add_reference(self, reference):
        if reference.get('camera','cam_head') != self.camera:
            return False
        item = self.reference_asset(reference)
        if item is None: return False
        self.templates.append(item)
        return True

    def ground(self, row):
        def can_retry(error):
            if isinstance(error, (transport_retry.TransportInterruption, requests.Timeout,
                                  requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
                request_budget(self).check_available()
                return True
            return False
        return transport_retry.request(lambda: self._ground_once(row), can_retry,
            lambda event, **fields: self.log('transport_' + event, step=row['step'], **fields))

    def _ground_once(self, row):
        request_budget(self).check_available()
        endpoint = os.environ.get('SEEK_PERCEPTION_ENDPOINT') or os.environ.get('ASTRA_ENDPOINT') or 'https://api.openai.com/v1/responses'
        token = self.token_provider()
        model = os.environ.get('SEEK_PERCEPTION_MODEL', 'gpt-6-astra')
        properties = {'visible':{'type':'boolean'}, 'bbox':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},
                      'description':{'type':'string'}}
        content = [{'type':'input_text','text':
            'Locate the single manipulated target for this task in the '+self.camera+' image. Instruction: '+self.instruction+
            ' Preserve object identity, including during grasping. Do not select a similar distractor or robot part. '
            'Other camera views are for identity context only. Return a tight visible selected-camera bbox, normalized xyxy. '
            'If the target is hidden or ambiguous in the selected camera, visible=false and bbox=[0,0,0,0]. '
            'This is perception; do not propose actions.'},
            {'type':'input_text','text':self.camera+' (selected image)'}, {'type':'input_image','image_url':row['images'][self.camera],'detail':'high'}]
        for name in ['cam_head','cam_left_wrist','cam_right_wrist']:
            if name != self.camera and name in row['images']:
                content.extend([{'type':'input_text','text':name+' (context only)'},
                    {'type':'input_image','image_url':row['images'][name],'detail':'high'}])
        payload = dict(model=model, reasoning={'effort':self.effort},store=False,stream=True,
            input=[{'role':'user','content':content}],tools=[{'type':'namespace','name':'perception','description':'Target identity',
                'tools':[{'type':'function','name':'locate','strict':True,'parameters':{'type':'object','properties':properties,
                    'required':list(properties),'additionalProperties':False}}]}],tool_choice='required')
        payload.setdefault('instructions', '')
        model_attempt = request_budget(self).reserve('perception')
        self.grounding_calls += 1; request = self.grounding_calls
        started = time.perf_counter()
        self.log('request_started', request=request, model_attempt=model_attempt, step=row['step'],
                 effort=self.effort, model=model)
        try:
            completed = None; items = []
            with requests.post(endpoint, headers={'Authorization':'Bearer '+token},
                               json=payload,stream=True,timeout=(30,600)) as response:
                if response.status_code!=200:
                    raise RuntimeError('Perception HTTP '+str(response.status_code))
                for line in response.iter_lines():
                    if not line.startswith(b'data: ') or line[6:]==b'[DONE]': continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        raise transport_retry.TransportInterruption('provider_stream_truncated') from None
                    if event.get('type')=='response.output_item.done': items.append(event['item'])
                    if event.get('type')=='response.completed':
                        completed=event['response']
                        break
                    if event.get('type') in ('response.failed','response.incomplete','error'):
                        raise RuntimeError('Perception response incomplete')
            if completed is None: raise transport_retry.TransportInterruption('provider_stream_truncated')
            self.grounding_completed += 1
            calls = [x for x in (completed.get('output') or items) if x.get('type')=='function_call']
            if len(calls)!=1: raise ValueError('Expected one grounding response')
            answer = json.loads(calls[0]['arguments'])
            self.log('request_completed', request=request,step=row['step'],answer=answer,usage=completed.get('usage'))
            if not answer['visible']: return False
            image = frame(row,self.camera); h,w = image.shape[:2]; box=np.array(answer['bbox'],float)
            if (box.shape != (4,) or not np.all(np.isfinite(box))
                    or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1)):
                raise ValueError('Invalid grounding box')
            pixels=(box*[w,h,w,h]).astype(int)
            if min(pixels[2:]-pixels[:2])<4: raise ValueError('Grounding box too small for identity tracking')
            path=self.folder/f'reference_{row["step"]}_{request}.jpg'
            if not cv2.imwrite(str(path),image): raise RuntimeError('Could not save grounding reference')
            reference=dict(image=str(path),camera=self.camera,bbox=box.tolist(),instruction=self.instruction,source_step=row['step'],
                kind='localization',identity_description=answer['description'],visual_validation='model_visible_target',
                task=self.task, target_key=self.binding.get('target_key'),
                target_description=self.binding.get('target_description'), task_binding=self.binding,
                grounding_request=request)
            if not self.add_reference(reference): raise ValueError('Grounding reference is unusable')
            self.references.append(reference)
            self.active=self.templates[-1]; self.size=self.active['size']
            self.center=(box[:2]+box[2:])*[w,h]/2; self.previous=image
            return True
        except (requests.RequestException,ValueError,KeyError,RuntimeError,json.JSONDecodeError) as error:
            self.log('request_failed',request=request,step=row['step'],error=type(error).__name__+': '+str(error))
            if transport_retry.enabled() and isinstance(error, (transport_retry.TransportInterruption,
                    requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
                raise
            return False
        finally:
            self.grounding_seconds += time.perf_counter()-started

    def segment(self, row, box, text=None, prompt_mode=None):
        from .segmentation_client import segment
        options = {'prompt_mode': prompt_mode} if prompt_mode is not None else {}
        return segment(row['images'][self.camera], box, text, **options)

    def seed_temporal(self, row, observed):
        if not observed.get('mask'):
            return
        from .segmentation_client import track
        result = track(row['images'][self.camera], self.temporal_id, row['step'], observed['mask'])
        self.temporal_seeded = bool(result.get('valid'))
        self.log('temporal_seed', step=row['step'], track_id=self.temporal_id,
                 valid=self.temporal_seeded, source='accepted_current_instance_mask')

    def temporal_observation(self, row):
        if not getattr(self, 'temporal_seeded', False):
            return None
        from .segmentation_client import track
        result = track(row['images'][self.camera], self.temporal_id, row['step'])
        if result.get('reason') in ('track_uninitialized', 'track_frame_discontinuity'):
            self.temporal_seeded = False
        self.log('temporal_observation', step=row['step'], track_id=self.temporal_id,
                 **{k:result.get(k) for k in ('valid','reason','presence_logit','tracked_frames','bbox')})
        return result

    def reference_hypotheses(self, image):
        candidates = []
        for item in self.templates:
            center,score,margin=template_match(image,item['template'])
            continuous=self.center is None or np.linalg.norm(center-self.center)<max(25.,float(np.linalg.norm(item['size'])))
            if score>.78 and margin>.045 and continuous:
                candidates.append(dict(item=item,center=center,score=score,margin=margin,method='reference_match'))
        return candidates

    @staticmethod
    def relation_tolerance(first_size, second_size):
        return max(10., .4 * min(float(np.linalg.norm(first_size)), float(np.linalg.norm(second_size))))

    def relation_hypothesis(self, references, tracked):
        def agrees(a, b):
            return np.linalg.norm(np.asarray(a['center'])-b['center']) < self.relation_tolerance(
                a['item']['size'], b['item']['size'])

        def near_previous(candidate):
            return (self.center is None or self.size is None or
                    np.linalg.norm(candidate['center']-self.center) < self.relation_tolerance(
                        candidate['item']['size'], self.size))

        selected, reason = None, 'no_coherent_target_relation'
        if tracked is not None:
            selected, reason = tracked, 'active_temporal_track'
        elif self.active is not None:
            active = [c for c in references if c['item'] is self.active and near_previous(c)]
            if active:
                selected, reason = max(active, key=lambda c: c['score']), 'continuous_active_reference'
        if selected is None and references:
            if (all(agrees(a, b) for a in references for b in references)
                    and all(near_previous(c) for c in references)):
                selected = max(references, key=lambda c: c['score'])
                reason = 'consistent_reference_relations'
            else:
                reason = 'conflicting_reference_relations_without_temporal_support'
        candidates = [*references, *([tracked] if tracked is not None else [])]
        hypotheses = []
        accepted_centers = []
        for candidate in candidates:
            consistent = selected is not None and agrees(candidate, selected)
            reference = candidate['item'].get('reference', {})
            hypotheses.append(dict(source=candidate['method'], center=np.asarray(candidate['center']).tolist(),
                reference_id=reference.get('id'), source_episode=reference.get('episode'),
                source_step=reference.get('source_step'), appearance_score=candidate['score'],
                identity_margin=candidate['margin'], agrees_with_selected=bool(consistent),
                temporal_evidence=candidate.get('temporal_evidence')))
            if consistent:
                accepted_centers.append(np.asarray(candidate['center']))
        conflict = selected is not None and any(not h['agrees_with_selected'] for h in hypotheses)
        if conflict:
            reason += '_resolves_reference_conflict'
        uncertainty = None
        if selected is not None:
            uncertainty = max(1., float(selected.get('uncertainty_px', 0.)),
                max((float(np.linalg.norm(c-selected['center'])) for c in accepted_centers), default=0.))
        evidence = dict(reason=reason, hypotheses=hypotheses,
            selected_source=selected['method'] if selected is not None else None,
            selected_reference_id=selected['item'].get('reference', {}).get('id') if selected is not None else None,
            uncertainty_px=uncertainty, reference_conflict=bool(conflict),
            identity_source='appearance_and_temporal_image_evidence',
            action_response_used_for_identity=False)
        return selected, evidence

    def track_reference(self, image):
        if self.previous is not None and self.active is not None:
            old=cv2.cvtColor(self.previous,cv2.COLOR_BGR2GRAY); gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            mask=np.zeros_like(old); lo=np.maximum((self.center-self.size*.4).astype(int),0);hi=np.minimum((self.center+self.size*.4).astype(int),[image.shape[1],image.shape[0]])
            mask[lo[1]:hi[1],lo[0]:hi[0]]=255
            points=cv2.goodFeaturesToTrack(old,40,.02,3,mask=mask)
            if points is not None and len(points)>=4:
                q,st,_=cv2.calcOpticalFlowPyrLK(old,gray,points,None,winSize=(21,21),maxLevel=3)
                back=sb=None
                if q is not None and st is not None:
                    back,sb,_=cv2.calcOpticalFlowPyrLK(gray,old,q,None,winSize=(21,21),maxLevel=3)
                ok=np.zeros(len(points),dtype=bool)
                if back is not None and sb is not None:
                    ok=(st[:,0]>0)&(sb[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<1.)
                if ok.sum()>=4:
                    shifts=q[:,0][ok]-points[:,0][ok]; shift=np.median(shifts,axis=0); center=self.center+shift
                    lo=np.maximum((center-self.size/2-8).astype(int),0);hi=np.minimum((center+self.size/2+8).astype(int),[image.shape[1],image.shape[0]])
                    patch=image[lo[1]:hi[1],lo[0]:hi[0]]; template=self.active['template']
                    if patch.shape[0]>=template.shape[0] and patch.shape[1]>=template.shape[1]:
                        match,score,margin=template_match(patch,template)
                        residual=float(np.median(np.linalg.norm(shifts-shift,axis=1)))
                        backward_error=float(np.median(np.linalg.norm(back[:,0][ok]-points[:,0][ok],axis=1)))
                        appearance_center=lo+match
                        appearance_error=float(np.linalg.norm(appearance_center-center))
                        if (score>.55 and np.linalg.norm(shift)<25 and residual<2.
                                and appearance_error<self.relation_tolerance(self.size,self.size)):
                            return dict(item=self.active,center=center,score=score,margin=margin,method='flow_with_reference',
                                uncertainty_px=max(backward_error,residual,appearance_error),
                                temporal_evidence=dict(points=int(ok.sum()), shift_px=shift.tolist(),
                                    flow_residual_px=residual, backward_error_px=backward_error,
                                    appearance_center=appearance_center.tolist(), appearance_error_px=appearance_error))
        return None

    def observe(self, row, allow_ground=True):
        if row['step']==self.last_step:
            upgrade = (allow_ground and not getattr(self, '_last_ground_allowed', True)
                       and not self.last_result.get('valid'))
            if not upgrade:
                return self.last_result
        self._last_ground_allowed = bool(allow_ground)
        image=frame(row,self.camera); best=None; result=None; grounded=False
        references=[]; relation_evidence=dict(reason='no_coherent_target_relation', hypotheses=[], uncertainty_px=None)
        selected_program = self.program
        operations = selected_program['steps'] if selected_program else PROGRAM_STEPS
        executed = []
        calls_before = self.grounding_calls

        def grounded_hypothesis():
            nonlocal relation_evidence
            relation_evidence=dict(relation_evidence, reason='fresh_model_target_binding',
                selected_source='model_grounding', selected_reference_id=self.active['reference'].get('id'),
                uncertainty_px=None, identity_source='fresh_model_target_binding',
                action_response_used_for_identity=False)
            return dict(item=self.active,center=self.center.copy(),score=1.,margin=1.,method='model_grounding')

        def segmented(candidate):
            size=candidate['item']['size']; center=candidate['center']; box=np.r_[center-size/2,center+size/2]
            reference=candidate['item']['reference']
            options={'text':reference.get('text')}
            if reference.get('prompt_mode') is not None:options['prompt_mode']=reference['prompt_mode']
            observed=self.segment(row,box,**options)
            displacement=float(np.linalg.norm(np.array(observed['center'])-center))
            valid=bool(observed['valid'] and displacement<self.relation_tolerance(size,size))
            evidence=dict(relation_evidence, segmentation_center=observed['center'],
                segmentation_anchor_displacement_px=displacement,
                uncertainty_px=max(1., float(relation_evidence.get('uncertainty_px') or 0.), displacement),
                uncertainty_kind='observed_image_spread_not_calibrated_confidence')
            if not valid:
                evidence['reason']='segmentation_conflicts_with_selected_relation'
            answer=dict(observed,step=row['step'],valid=valid,relation_consistent=valid,
                relation_evidence=evidence,confidence=candidate['score'] if valid else 0.,
                appearance_score=candidate['score'],identity_margin=candidate['margin'],method=candidate['method'],
                identity_reason=evidence['reason'], reference_id=candidate['item']['reference'].get('id'),
                reference_episode=candidate['item']['reference'].get('episode'),
                reference_source_step=candidate['item']['reference'].get('source_step'))
            if valid:
                self.center=np.asarray(observed['center']);self.size=size;self.active=candidate['item'];self.previous=image
                self.seed_temporal(row, observed)
            else:
                self.previous=None
            return answer

        for operation in operations:
            outcome = 'skipped'
            if operation == 'bind_task_target':
                self.binding = task_binding(self.instruction, self.task)
                outcome = 'bound' if self.binding.get('parsed') else 'instruction_fallback'
            elif operation == 'match_target_reference':
                references = self.reference_hypotheses(image)
                outcome = 'matched' if references else 'no_match'
            elif operation == 'track_target_reference':
                temporal = self.temporal_observation(row)
                if temporal is not None and temporal.get('valid') and self.active is not None:
                    self.center=np.asarray(temporal['center'])
                    bounds=np.asarray(temporal['bbox']);self.size=bounds[2:]-bounds[:2]
                    self.previous=image
                    reference=self.active['reference']
                    relation_evidence=dict(reason='temporal_segmentation_instance', hypotheses=[],
                        selected_source='temporal_segmentation', selected_reference_id=reference.get('id'),
                        uncertainty_px=None, reference_conflict=False,
                        identity_source='sam3_causal_instance_memory', action_response_used_for_identity=False,
                        initial_frame_id=temporal.get('initial_frame_id'), presence_logit=temporal.get('presence_logit'),
                        scope=temporal.get('scope'))
                    best=dict(item=self.active, center=self.center, score=temporal['score'],
                              margin=None, method='temporal_segmentation')
                    result=dict(temporal, step=row['step'], relation_consistent=True,
                        relation_evidence=relation_evidence, confidence=temporal['score'],
                        appearance_score=None, identity_margin=None, method='temporal_segmentation',
                        identity_reason='temporal_segmentation_instance', reference_id=reference.get('id'),
                        reference_episode=reference.get('episode'), reference_source_step=reference.get('source_step'))
                else:
                    tracked = self.track_reference(image)
                    best, relation_evidence = self.relation_hypothesis(references, tracked)
                outcome = 'coherent_relation' if best is not None else 'inconsistent_or_missing_relation'
            elif (operation == 'ground_task_target' and best is None
                    and allow_ground):
                grounded = self.ground(row)
                if grounded:
                    best=grounded_hypothesis()
                outcome = 'grounded' if grounded else 'unavailable'
            elif operation == 'segment_target_reference' and best is not None:
                if result is None:
                    result=segmented(best)
                outcome = 'valid' if result['valid'] else 'invalid'
            executed.append(dict(operation=operation, outcome=outcome))
        if result is not None and not result['valid'] and allow_ground and self.grounding_calls==calls_before:
            relation_evidence=result['relation_evidence']
            grounded=self.ground(row)
            executed.append(dict(operation='ground_task_target',outcome='grounded_after_relation_conflict' if grounded else 'unavailable'))
            if grounded:
                best=grounded_hypothesis();result=segmented(best)
        if result is None:
            self.previous=None
            result=dict(step=row['step'],valid=False,relation_consistent=False,relation_evidence=relation_evidence,
                center=self.center.tolist() if self.center is not None else [0.,0.],
                confidence=0.,identity_reason=relation_evidence['reason'])
        program_source = ((selected_program.get('source_episode') or selected_program.get('episode'))
                          if selected_program else None)
        result.update(camera=self.camera, target_key=self.binding.get('target_key'), target_binding=self.binding,
                      program_id=selected_program['id'] if selected_program else None,
                      program_reused=self.program_reused, program_origin=self.program_origin,
                      program_source_episode=program_source,
                      object_reference_reused=bool(result.get('reference_episode') is not None
                          and result.get('method') in ('reference_match', 'flow_with_reference', 'temporal_segmentation')),
                      semantic_grounding_requested=self.grounding_calls>calls_before)
        self.log('program_executed', step=row['step'], program_id=result['program_id'],
                 program_reused=result['program_reused'], origin=result['program_origin'],
                 source_episode=program_source, target_key=result['target_key'], operations=executed,
                 object_reference_reused=result['object_reference_reused'],
                 semantic_grounding_requested=result['semantic_grounding_requested'], valid=result['valid'],
                 relation_consistent=result['relation_consistent'], relation_evidence=result['relation_evidence'])
        if (self.program is None and result['valid'] and best is not None
                and any(best['item']['reference'] is r for r in self.references)):
            reference = best['item']['reference']
            sources = [{k: reference.get(k) for k in ('id','episode','source_step','instruction')}]
            self.program = self.make_program(sources)
            self.program['grounding_request'] = reference['grounding_request']
            self.program['validation_step'] = row['step']
            self.program['observed_operations'] = executed
            self.programs.append(self.program)
            self.program_origin = 'current_episode_registration'
            self.log('program_registered', program=self.program)
        self.last_step=row['step'];self.last_result=result
        self.log('observation',**result)
        return result

    def close(self):
        if getattr(self, 'temporal_id', None):
            from .segmentation_client import release_track
            try:
                release_track(self.temporal_id)
            except (OSError, RuntimeError, ValueError) as error:
                self.log('temporal_cleanup_error', error=str(error))
            self.temporal_seeded = False
        worker = self.worker
        if worker is None:
            return

        def report(stage, error):
            try:
                self.log('worker_cleanup_error', stage=stage, error=type(error).__name__+': '+str(error))
            except OSError:
                pass

        try:
            try:
                worker.stdin.close()
            except OSError as error:
                report('stdin', error)
            for action, timeout in (('eof', 20), ('terminate', 5), ('kill', 5)):
                if action != 'eof':
                    try:
                        getattr(worker, action)()
                    except OSError as error:
                        report(action, error)
                try:
                    worker.wait(timeout=timeout)
                    break
                except (subprocess.TimeoutExpired, OSError) as error:
                    report(action+'_wait', error)
        finally:
            for stream in (worker.stdout, self.error_file):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError as error:
                        report('stream_close', error)
            self.worker = None
            self.error_file = None
