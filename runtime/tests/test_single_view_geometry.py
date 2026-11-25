import json
import unittest

import numpy as np

from seek.policy import rgb_perception as rgb
from seek.policy.calibrated_geometry import MaskObservation
from seek.policy.single_view_geometry import measure
from test_joint_silhouette_geometry import box_observations, measure_box


class SingleViewGeometryTests(unittest.TestCase):
    def test_box_in_each_current_camera_keeps_the_metric_result_contract(self):
        row, masks, views = box_observations()
        for selection in views:
            with self.subTest(camera=selection['camera']):
                result = measure_box(row, masks, [selection])
                self.assertTrue(result['valid'], result)
                self.assertTrue(result['height_ambiguous'])
                estimate = result['estimate']
                self.assertGreater(estimate['silhouette_match'], .8)
                self.assertTrue(np.isfinite(estimate['bounds_environment_m']).all())
                self.assertEqual(estimate['views_used'], 1)
                self.assertGreater(estimate['height_m'], 0.)
                self.assertLessEqual(estimate['height_m'], .36)
                json.dumps(result, allow_nan=False)

    def test_empty_mask_and_impossible_support_plane_are_rejected(self):
        row, masks, views = box_observations()
        camera = rgb.camera_view(row, views[0]['camera'])
        empty = MaskObservation(camera, np.zeros(camera.shape, bool))
        self.assertIsNone(measure(empty, 0.))
        visible = MaskObservation(camera, masks[views[0]['camera']])
        self.assertIsNone(measure(visible, camera.pos[2]+1.))

    def test_single_view_result_is_deterministic(self):
        row, masks, views = box_observations()
        first = measure_box(row, masks, views[:1])
        second = measure_box(row, masks, views[:1])
        self.assertEqual(first['estimate'], second['estimate'])


if __name__ == '__main__':
    unittest.main()
