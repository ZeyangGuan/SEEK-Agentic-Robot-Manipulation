import numpy as np
from scipy.spatial.transform import Rotation

from .model_policy import STATE_KEYS, plain


SOURCE_NOTE = ('Live camera focal length/horizontal aperture/resolution with square-pixel pinhole projection; '
               'Camera.get_world_pose(camera_axes=usd); environment origin subtracted')


def finite(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    numbers = np.asarray(value, dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError('Camera calibration contains a non-finite value')
    return numbers


def _close(left, right, tolerance):
    return np.allclose(left, right, atol=tolerance, rtol=0)


def camera_arrays(camera):
    intrinsics, pose, shape = (finite(camera[key]) for key in
                               ('intrinsic_matrix', 'camera_to_environment', 'shape'))
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4) or shape.shape != (3,):
        raise ValueError('Camera calibration has the wrong array shapes')
    rotation = pose[:3, :3]
    pinhole = (min(intrinsics[0, 0], intrinsics[1, 1]) > 1 and _close(intrinsics[2], [0, 0, 1], 1e-9)
               and intrinsics[1, 0] == 0)
    image = min(shape[:2]) > 0 and shape[2] == 3 and np.equal(shape, shape.astype(int)).all()
    rigid = (_close(pose[3], [0, 0, 0, 1], 1e-7) and _close(rotation.T @ rotation, np.eye(3), 1e-5)
             and abs(np.linalg.det(rotation) - 1) <= 1e-5)
    if not (pinhole and image and rigid):
        raise ValueError('Camera intrinsics, pose or image shape are not valid')
    return intrinsics, pose, shape


def _pinhole(camera):
    width, height = camera.get_resolution()
    focal = width * camera.get_focal_length() / camera.get_horizontal_aperture()
    return np.array([[focal, 0., width / 2.], [0., focal, height / 2.], [0., 0., 1.]])


def capture_camera_reference(env, obs, name):
    if env.num_envs != 1:
        raise ValueError('Camera calibration needs a single environment')
    manager = env.camera_manager
    camera = manager.cameras[0][manager.camera_names[0].index(name)]
    position, quaternion = (finite(value) for value in camera.get_world_pose(camera_axes='usd'))
    if position.shape != (3,) or quaternion.shape != (4,) or abs(np.linalg.norm(quaternion) - 1) > .001:
        raise ValueError(f'{name} reports an invalid world pose')
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    origin = finite(env.robot_manager.scene.env_origins[0])
    if origin.shape != (3,):
        raise ValueError('The environment origin is not a 3-vector')
    pose[:3, 3] = position - origin
    intrinsics = _pinhole(camera)
    shape = list(obs['vision'][name]['color'].shape)
    if list(camera.get_resolution()) != [shape[1], shape[0]]:
        raise ValueError(f'{name} resolution differs from its RGB image')
    reference = dict(intrinsic_matrix=intrinsics.tolist(), camera_to_environment=pose.tolist(), shape=shape)
    camera_arrays(reference)
    return reference


def capture_head_calibration(env, obs):
    head = capture_camera_reference(env, obs, 'cam_head')
    state = {key: plain(obs['state'][key]) for key in STATE_KEYS}
    return dict(source='TEST_RUNTIME', head_reference=head,
                initial_sample=dict(step=0, state=state, head=head), camera_source=SOURCE_NOTE)
