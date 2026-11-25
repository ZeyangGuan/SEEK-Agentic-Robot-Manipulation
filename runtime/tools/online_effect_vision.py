import base64
from contextlib import redirect_stdout
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from seek.policy.segmented_geometry import encode_mask


def mask_result(mask, score, box=None):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('The segmentation worker must return a mask at the original image resolution')
    ys, xs = np.where(mask)
    center = [float(xs.mean()), float(ys.mean())] if len(xs) else (
        ((box[:2]+box[2:])/2).tolist() if box is not None else None)
    bounds = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if len(xs) else None
    valid = len(xs) >= 20 and np.isfinite(score)
    if box is not None:
        valid = valid and len(xs) < max(4000, 8*np.prod(box[2:]-box[:2]))
    return dict(center=center, bbox=bounds, area=int(len(xs)), valid=bool(valid),
                score=float(score), mask=encode_mask(mask), backend='sam3')


def as_numpy(value):
    if not hasattr(value, 'detach'):
        return np.asarray(value)
    tensor = value.detach().cpu()

    if tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.numpy()


class Sam3Service:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.image_id, self.state = None, None
        self.temporal = None
        self.image_tensor = None

    def process(self, request):
        if 'release_track' in request:
            released = self.temporal.release(request['release_track']) if self.temporal else False
            return dict(backend='sam3', released=released)
        raw = base64.b64decode(request['image'].split(',',1)[1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Invalid RGB image')
        h, w = frame.shape[:2]
        box = request.get('box')
        if box is not None:
            box = np.asarray(box, float)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError('Box must be original-image pixel xyxy coordinates')

            box = np.clip(box, [0,0,0,0], [w,h,w,h])
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('Box does not intersect the current image')
        prompt = request.get('text')
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256):
            raise ValueError('Text prompt must be a nonempty concept of at most 256 characters')
        mode = request.get('prompt_mode')
        if mode is not None and (mode not in ('concept', 'joint') or prompt is None):
            raise ValueError('prompt_mode requires text and must be concept or joint')
        if mode == 'joint' and box is None:
            raise ValueError('Joint prompting requires a selected box')
        temporal_request = request.get('track')
        if prompt is None and box is None and temporal_request is None:
            raise ValueError('A box or text prompt is required')
        image_id = hashlib.sha256(raw).hexdigest()
        if image_id != self.image_id:
            self.state = self.processor.set_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.image_id = image_id
            self.image_tensor = None
        if temporal_request is not None:
            import torch
            from seek.policy.temporal_segmentation import TemporalTracks
            if self.temporal is None:
                self.temporal = TemporalTracks(self.model.inst_interactive_predictor.model)
            if self.image_tensor is None:
                rgb = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2,0,1).to(self.processor.device)
                self.image_tensor = self.processor.transform(rgb).unsqueeze(0)
            tracked = dict(self.temporal.update(temporal_request['id'], temporal_request['frame_id'], image_id,
                self.image_tensor, self.state['backbone_out']['sam2_backbone_out'], (h,w), temporal_request.get('seed_mask')))
            mask = tracked.pop('mask_array', np.zeros((h,w), bool))
            logit = tracked.get('presence_logit', -100.)
            score = float(1/(1+np.exp(-np.clip(logit,-100,100))))
            return dict(mask_result(mask,score), **tracked)

        self.processor.reset_all_prompts(self.state)
        if prompt is None:
            masks, scores, _ = self.model.predict_inst(self.state, box=box, multimask_output=False)
            if as_numpy(masks)[0].shape != (h,w):
                raise ValueError('segmentation instance mask resolution differs from the original image')
            result = mask_result(as_numpy(masks)[0], float(as_numpy(scores)[0]), box)
        else:
            output = self.processor.set_text_prompt(prompt=prompt, state=self.state)
            if mode == 'joint':
                visual_box = [(box[0]+box[2])/(2*w), (box[1]+box[3])/(2*h),
                              (box[2]-box[0])/w, (box[3]-box[1])/h]
                output = self.processor.add_geometric_prompt(
                    box=visual_box, label=True, state=self.state)
            masks, scores = as_numpy(output['masks']), as_numpy(output['scores'])
            candidates = [mask_result(mask.reshape(h, w), float(score)) for mask, score in zip(masks, scores)]
            candidates = [c for c in candidates if c['valid']]
            candidates.sort(key=lambda c: c['score'], reverse=True)
            result = dict(backend='sam3', text=prompt, candidates=candidates, count=len(candidates),
                          prompt_mode=mode or 'concept')
            if box is not None:
                ranked = []
                for candidate in candidates:
                    b = np.asarray(candidate['bbox'], float)
                    intersection = np.maximum(0, np.minimum(b[2:], box[2:])-np.maximum(b[:2], box[:2])).prod()
                    union = np.prod(b[2:]-b[:2])+np.prod(box[2:]-box[:2])-intersection
                    ranked.append((float(intersection/union), candidate))
                ranked.sort(key=lambda item: item[0], reverse=True)
                ambiguous = len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .05
                if not ranked or ranked[0][0] < .2 or ambiguous:
                    result.update(mask_result(np.zeros((h, w), bool), 0., box), valid=False,
                                  reason='text_instance_ambiguous' if ambiguous else 'text_instance_not_in_box')
                else:
                    result.update(ranked[0][1], selection_iou=ranked[0][0])
                result.pop('candidates')
        if request.get('save') and result.get('mask'):
            from seek.policy.segmented_geometry import decode_mask
            mask = decode_mask(result['mask'])
            frame[mask] = (frame[mask]*.6+np.array([65, 200, 120])*.4).astype(np.uint8)
            if not cv2.imwrite(request['save'], frame):
                raise OSError('Could not save segmentation visualization')
        return result


def main():
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            torch.set_num_threads(2)
            checkpoint = Path(os.environ['SEEK_SAM3_CHECKPOINT'])
            if not checkpoint.is_file():
                raise FileNotFoundError('Authorized segmentation checkpoint is missing; set SEEK_SAM3_CHECKPOINT')
            model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False,
                                           enable_inst_interactivity=True, device='cuda')
            service = Sam3Service(model, Sam3Processor(model, device='cuda'))
        print(json.dumps(dict(ready=True, backend='sam3')), flush=True)
    except Exception as error:
        print(json.dumps(dict(ready=False, backend='sam3', error=type(error).__name__+': '+str(error))), flush=True)
        return 1
    for line in sys.stdin:
        started = time.perf_counter()
        try:
            with redirect_stdout(sys.stderr), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = service.process(json.loads(line))
            result['elapsed_s'] = time.perf_counter()-started
        except Exception as error:
            result = dict(backend='sam3', error=type(error).__name__+': '+str(error))
        print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
