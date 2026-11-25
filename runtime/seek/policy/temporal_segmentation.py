import numpy as np
from .segmented_geometry import decode_mask


class TemporalTracks:
    def __init__(self, predictor):
        self.predictor = predictor
        self.tracks = {}

    def release(self, track_id):
        return self.tracks.pop(track_id, None) is not None

    def update(self, track_id, step, image_id, image_tensor, features, shape, seed=None):
        import torch
        if not isinstance(track_id, str) or not track_id or len(track_id) > 256:
            raise ValueError('A unique instance track_id is required')
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('frame_id must be a nonnegative integer')
        entry = self.tracks.get(track_id)
        if seed is not None:
            mask = decode_mask(seed)
            if mask.shape != shape or not mask.any():
                raise ValueError('Seed mask must match the current RGB dimensions and be nonempty')
            state = self.predictor.init_state(video_height=shape[0], video_width=shape[1],
                num_frames=1, offload_state_to_cpu=True)
            entry = dict(state=state, first_step=step, step=step-1, image_id=None, shape=shape)
            self.tracks[track_id] = entry
        elif entry is None:
            return dict(valid=False, reason='track_uninitialized')
        elif step == entry['step'] and image_id == entry['image_id']:
            return entry['result']
        elif step != entry['step']+1 or shape != entry['shape']:
            self.release(track_id)
            return dict(valid=False, reason='track_frame_discontinuity')

        state = entry['state']
        index = step-entry['first_step']

        state['cached_features'] = {index: (image_tensor, features)}
        state['num_frames'] = index+1
        if seed is not None:
            self.predictor.add_new_mask(state, 0, 1, torch.from_numpy(mask))
        outputs = list(self.predictor.propagate_in_video(state, start_frame_idx=index,
            max_frame_num_to_track=0, reverse=False, tqdm_disable=True,
            propagate_preflight=seed is not None))
        if len(outputs) != 1 or outputs[0][0] != index:
            self.release(track_id)
            raise RuntimeError('segmentation temporal inference did not return exactly the current frame')
        _, ids, _, masks, scores = outputs[0]
        mask = (masks[0, 0] > 0).cpu().numpy()
        logit = float(scores.reshape(-1)[0].float().cpu())
        result = dict(mask_array=mask, presence_logit=logit,
            valid=bool(mask.sum() >= 20 and np.isfinite(logit) and logit > 0),
            reason='temporal_segmentation_instance' if logit > 0 else 'temporal_segmentation_absent',
            source='temporal_segmentation_mask', initial_frame_id=entry['first_step'], frame_id=step,
            tracked_frames=index+1, object_id=int(ids[0]),
            scope='Temporal association from the original visible seed; may drift. Not contact or retention proof.')
        entry.update(step=step, image_id=image_id, result=result)
        self._prune(state, index)
        result['memory_frames'] = {key:len(value) for key,value in state['output_dict'].items()}
        return result

    def _prune(self, state, index):
        if self.predictor.use_memory_selection:
            raise RuntimeError('Bounded streaming memory requires the image tracker configuration')
        oldest = index-self.predictor.num_maskmem
        stores = [state['output_dict'], *state['output_dict_per_obj'].values(),
                  *state['temp_output_dict_per_obj'].values()]
        for store in stores:
            for frame in list(store['non_cond_frame_outputs']):
                if frame < oldest:
                    del store['non_cond_frame_outputs'][frame]
        state['consolidated_frame_inds']['non_cond_frame_outputs'].intersection_update(range(max(0, oldest), index+1))
        for frame in list(state['frames_already_tracked']):
            if frame < oldest and frame not in state['output_dict']['cond_frame_outputs']:
                del state['frames_already_tracked'][frame]
