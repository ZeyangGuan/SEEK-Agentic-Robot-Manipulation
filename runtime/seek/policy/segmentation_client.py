import atexit
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading


class Sam3Client:
    def __init__(self):
        self.worker = None
        self.errors = None
        self.lock = threading.RLock()

    def close(self):
        with self.lock:
            worker, self.worker = self.worker, None
            if worker is not None:
                try:
                    if worker.poll() is None:
                        worker.terminate()
                        try:
                            worker.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            worker.kill(); worker.wait(timeout=5)
                finally:
                    for stream in (worker.stdin,worker.stdout):
                        if stream is not None:stream.close()
            if self.errors is not None:
                self.errors.close(); self.errors = None

    def request(self, image=None, box=None, text=None, track=None, release_track=None, prompt_mode=None):
        with self.lock:
            try:
                if self.worker is None or self.worker.poll() is not None:
                    self.close()
                    python = os.environ.get('SEEK_SAM_PYTHON')
                    if not python:
                        raise RuntimeError('Set SEEK_SAM_PYTHON to the segmentation environment interpreter')
                    script = Path(__file__).resolve().parents[2]/'tools/online_effect_vision.py'
                    self.errors = tempfile.TemporaryFile(mode='w+t',encoding='utf-8')
                    self.worker = subprocess.Popen([python,'-u',str(script)],stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,stderr=self.errors,text=True)
                    line = self.worker.stdout.readline()
                    if not line:
                        self.errors.seek(0)
                        raise RuntimeError('segmentation worker exited during startup: '+self.errors.read()[-2000:])
                    ready = json.loads(line)
                    if not ready.get('ready') or ready.get('backend') != 'sam3':
                        raise RuntimeError('Segmentation worker unavailable: '+ready.get('error','unexpected backend'))
                request = dict(image=image)
                if box is not None: request['box'] = [float(value) for value in box]
                if text is not None: request['text'] = text
                if prompt_mode is not None: request['prompt_mode'] = prompt_mode
                if track is not None: request['track'] = track
                if release_track is not None: request['release_track'] = release_track
                self.worker.stdin.write(json.dumps(request)+'\n'); self.worker.stdin.flush()
                line = self.worker.stdout.readline()
                if not line: raise RuntimeError('segmentation worker terminated during inference')
                result = json.loads(line)
                if result.get('backend') != 'sam3': raise RuntimeError('Unexpected segmentation backend')
                if result.get('error'): raise ValueError('segmentation inference failed: '+result['error'])
                return result
            except (OSError, RuntimeError, json.JSONDecodeError):
                self.close()
                raise


client = Sam3Client()
atexit.register(client.close)


def segment(image, box=None, text=None, prompt_mode=None):
    return client.request(image,box,text,prompt_mode=prompt_mode)


def track(image, track_id, frame_id, seed_mask=None):
    request = dict(id=track_id, frame_id=frame_id)
    if seed_mask is not None: request['seed_mask'] = seed_mask
    return client.request(image, track=request)


def release_track(track_id):
    with client.lock:
        if client.worker is not None and client.worker.poll() is None:
            return client.request(release_track=track_id)
    return None
