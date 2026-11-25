import numpy as np
from scipy.spatial.transform import Rotation
from .model_policy import STATE_KEYS, plain


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite camera calibration')
    return result


def camera_arrays(camera):
    k, transform, shape = (array(camera[key]) for key in
        ('intrinsic_matrix', 'camera_to_environment', 'shape'))
    if k.shape != (3, 3) or transform.shape != (4, 4) or shape.shape != (3,):
        raise ValueError('Invalid head camera calibration shape')
    if (min(k[0, 0], k[1, 1]) <= 1 or not np.allclose(k[2], [0, 0, 1], atol=1e-9, rtol=0)
            or k[1, 0] != 0 or min(shape[:2]) <= 0 or shape[2] != 3
            or not np.equal(shape, shape.astype(int)).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(transform[:3, :3]) - 1) > 1e-5):
        raise ValueError('Invalid head camera intrinsics, pose, or image shape')
    return k, transform, shape


def rendered_pinhole_intrinsics(camera):
    width, height = camera.get_resolution()
    focal = width * camera.get_focal_length() / camera.get_horizontal_aperture()
    return np.array([[focal, 0., width / 2.], [0., focal, height / 2.], [0., 0., 1.]])


def capture_camera_reference(env, obs, name):
    if env.num_envs != 1:
        raise ValueError('Head calibration requires one live environment')
    camera_manager = env.camera_manager
    camera_id = camera_manager.camera_names[0].index(name)
    camera = camera_manager.cameras[0][camera_id]
    position, quaternion = camera.get_world_pose(camera_axes='usd')
    p, q = array(position), array(quaternion)
    if p.shape != (3,) or q.shape != (4,) or abs(np.linalg.norm(q) - 1) > .001:
        raise ValueError('Invalid head camera world pose')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    origin = array(env.robot_manager.scene.env_origins[0])
    if origin.shape != (3,):
        raise ValueError('Invalid environment origin')
    transform[:3, 3] = p - origin
    k = rendered_pinhole_intrinsics(camera)
    shape = list(obs['vision'][name]['color'].shape)
    if list(camera.get_resolution()) != [shape[1], shape[0]]:
        raise ValueError('Live RGB resolution differs from head calibration')
    head = {'intrinsic_matrix': k.tolist(), 'camera_to_environment': transform.tolist(), 'shape': shape}
    camera_arrays(head)
    return head


def capture_head_calibration(env, obs):
    head = capture_camera_reference(env, obs, 'cam_head')
    state = {key: plain(obs['state'][key]) for key in STATE_KEYS}
    return {'source': 'TEST_RUNTIME', 'head_reference': head,
            'initial_sample': {'step': 0, 'state': state, 'head': head},
            'camera_source': 'Live camera focal length/horizontal aperture/resolution with square-pixel pinhole projection; Camera.get_world_pose(camera_axes=usd); environment origin subtracted'}
