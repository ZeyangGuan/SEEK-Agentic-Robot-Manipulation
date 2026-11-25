import itertools
import unittest

import numpy as np

from seek.policy import rgb_perception as rgb
from seek.policy.calibrated_geometry import CalibratedCamera, MaskObservation, support_plane_points
from seek.policy.segmented_geometry import encode_mask
from test_rgb_perception import scene


class NativeGeometryTests(unittest.TestCase):
    def setUp(self):
        rgb._cache.clear()

    def camera(self):
        intrinsic = np.array([[120., 7., 50.], [0., 100., 40.], [0., 0., 1.]])
        pose = np.eye(4)
        pose[:3, 3] = [0., 0., 1.]
        return CalibratedCamera(intrinsic, pose, (80, 100))

    def test_calibrated_projection_against_hand_calculated_pixel_and_skew(self):
        camera = self.camera()

        row, col, depth = camera.project(np.array([.1, .2, .5]))
        self.assertAlmostEqual(row, 0.)
        self.assertAlmostEqual(col, 71.2)
        self.assertAlmostEqual(depth, .5)
        np.testing.assert_allclose(camera.rays(row, col), [.2, .4, -1.], atol=1e-12)

    def test_batched_projection_and_input_calibration_are_not_mutated(self):
        camera = self.camera()
        original = camera.intrinsic.copy()
        points = np.array([[[0., 0., 0.], [.1, 0., 0.]], [[0., .1, 0.], [0., 0., .5]]])
        row, col, depth = camera.project(points)
        self.assertEqual(row.shape, (2, 2))
        np.testing.assert_allclose(camera.pos + camera.rays(row, col) * depth[..., None], points, atol=1e-12)
        np.testing.assert_array_equal(camera.intrinsic, original)

    def test_camera_plane_projection_does_not_create_infinite_coordinates(self):
        rows, cols, depth = self.camera().project(np.array([[0., 0., 1.]]))
        self.assertTrue(np.isnan(rows[0]) and np.isnan(cols[0]))
        self.assertEqual(depth[0], 0.)

    def test_mask_validation(self):
        with self.assertRaises(ValueError):
            MaskObservation(self.camera(), np.zeros((2, 3)))

    def test_single_view_preserves_estimate_fields_and_height_ambiguity(self):
        row = scene()
        camera = 'cam_head'
        mask = np.zeros((240, 320), bool)
        mask[115:125, 155:165] = True
        segmenter = lambda *a, **k: dict(valid=True, mask=encode_mask(mask), score=1.)
        result = rgb.measure_rgb({0: row}, 0, [dict(camera=camera, bbox=[.4, .4, .6, .6])],
                                 0., segmenter=segmenter)
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['height_ambiguous'])
        estimate = result['estimate']
        self.assertEqual(estimate['views_used'], 1)
        self.assertGreater(estimate['height_m'], 0.)
        self.assertGreater(estimate['silhouette_match'], .8)
        self.assertEqual(len(estimate['point_environment_m']), 3)
        self.assertAlmostEqual(estimate['bounds_environment_m'][0][2], 0.)
        self.assertAlmostEqual(estimate['bounds_environment_m'][1][2], estimate['height_m'])

    def test_plane_projection_rejects_plane_behind_camera(self):
        camera = self.camera()
        observation = MaskObservation(camera, np.ones(camera.shape, bool))
        self.assertIsNone(support_plane_points(observation, 2.))

    def test_conservative_unknown_region_cannot_erase_known_foreground(self):
        camera = self.camera()
        mask = np.zeros(camera.shape, bool)
        mask[30:40, 40:50] = True
        observation = MaskObservation(camera, mask, np.ones(camera.shape, bool))
        self.assertEqual(support_plane_points(observation, 0.)['usable_pixels'], 100)

    def test_dlt_uses_all_views_and_is_permutation_invariant(self):
        row = scene()
        target = np.array([.08, .04, .07])
        samples = []
        for camera in row['images']:
            view = rgb.camera_view(row, camera)
            r, c, _ = view.project(target)
            samples.append(dict(camera=camera, uv=[(c+.5)/320, (r+.5)/240]))
        for order in itertools.permutations(samples):
            result = rgb.locate_rgb_point({0: row}, 0, list(order))
            self.assertTrue(result['valid'])
            self.assertEqual(result['solver'], 'calibrated_dlt')
            np.testing.assert_allclose(result['point_environment_m'], target, atol=1e-10)

    def test_parallel_correspondences_are_not_valid_3d_points(self):
        row = scene()
        for i, camera in enumerate(row['images']):
            pose = np.eye(4); pose[0, 3] = i*.1; pose[2, 3] = 1.
            row['camera_calibration'][camera]['camera_to_environment'] = pose.tolist()
        samples = [dict(camera=c, uv=[.5, .5]) for c in row['images']]
        result = rgb.locate_rgb_point({0: row}, 0, samples)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'degenerate_ray_geometry')

    def test_dlt_rejects_solution_behind_cameras(self):
        row = scene()
        samples = []
        for i, camera in enumerate(row['images']):
            pose = np.eye(4); pose[0, 3] = (i-1)*.1; pose[2, 3] = 1.
            row['camera_calibration'][camera]['camera_to_environment'] = pose.tolist()
            view = rgb.camera_view(row, camera)
            r, c, _ = view.project(np.array([0., 0., 2.]))
            samples.append(dict(camera=camera, uv=[(c+.5)/320, (r+.5)/240]))
        result = rgb.locate_rgb_point({0: row}, 0, samples)
        self.assertFalse(result['valid'])
        self.assertFalse(result['forward'])

    def test_tool_interval_and_projection_share_optical_depth_off_axis(self):
        camera = self.camera()
        shape = camera.shape
        geometry = dict(outer_bounds=dict(base=[[-1., -1., .1], [1., 1., .2]],
                                         fingers_at_zero_joint=[], finger_joint_axes=[]))
        state = dict(left_ee_pose=[0., 0., 0., 1., 0., 0., 0.], left_gripper_joint_positions=[])
        near = rgb.tool_occlusion_near(camera, state, geometry)
        self.assertAlmostEqual(near[60, 80], .8)
        point = camera.pos + .8 * camera.rays(60, 80)
        self.assertAlmostEqual(camera.project(point)[2], near[60, 80])


if __name__ == '__main__':
    unittest.main()
