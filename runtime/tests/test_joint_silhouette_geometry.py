import itertools
import unittest

import cv2
import numpy as np

from seek.policy import rgb_perception as rgb
from seek.policy.segmented_geometry import encode_mask
from seek.policy.execution_policy import TOOL_GEOMETRY
from test_rgb_perception import scene, calibration


def box_observations(height=.06):
    row = scene()
    corners = np.array(list(itertools.product([.02,.10],[-.03,.03],[0,height])))
    masks, views = {}, []
    for camera in row['images']:
        view = rgb.camera_view(row,camera)
        r,c,_ = view.project(corners)
        mask = np.zeros(view.shape,np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(np.c_[c,r]).astype(np.int32)),1)
        masks[camera] = mask.astype(bool)
        views.append(dict(camera=camera,bbox=[max(0,(c.min()-2)/320),max(0,(r.min()-2)/240),
                                             min(1,(c.max()+3)/320),min(1,(r.max()+3)/240)]))
    return row, masks, views


def measure_box(row,masks,views,tool_geometry=None):
    def segment(image,box,**options):
        return dict(valid=True,score=1.,mask=encode_mask(masks[image]))
    rgb._cache.clear()
    return rgb.measure_rgb({0:row},0,views,0.,tool_geometry=tool_geometry,segmenter=segment)


class JointSilhouetteGeometryTests(unittest.TestCase):
    def setUp(self):
        rgb._cache.clear()

    def test_two_views_retain_partially_visible_volume_in_either_order(self):
        row, masks, views = box_observations()
        views = views[:2]
        x1, y1, x2, y2 = views[0]['bbox']
        views[0]['exclude_boxes'] = [[x1, y1, (x1 + x2) / 2, y2]]
        first = measure_box(row, masks, views)
        reversed_ = measure_box(row, masks, views[::-1])
        self.assertTrue(first['valid'])
        self.assertTrue(reversed_['valid'])
        self.assertEqual(first['estimate']['views_used'],2)
        self.assertEqual(reversed_['estimate']['views_used'],2)
        self.assertEqual(first['estimate'],reversed_['estimate'])
        self.assertGreater(first['estimate']['joint_constraints']['one_view_supported_possible_voxels'],0)

    def test_conservative_tool_envelope_does_not_erase_observed_foreground(self):
        row, masks, views = box_observations()
        views = views[:2]
        row['state'] = dict(left_ee_pose=[.04, 0, .06, 1, 0, 0, 0],
                            left_gripper_joint_positions=[.02, .02])
        with_tool = measure_box(row, masks, views, tool_geometry=TOOL_GEOMETRY)
        without_tool = measure_box(row, masks, views)
        self.assertTrue(with_tool['valid'], with_tool)
        self.assertTrue(without_tool['valid'], without_tool)
        self.assertTrue(any(v['robot_or_excluded_pixels'] for v in with_tool['views']))
        for view in with_tool['views']:
            self.assertEqual(view['usable_pixels'],view['pixels'])

        with_constraints = with_tool['estimate']['joint_constraints']
        without_constraints = without_tool['estimate']['joint_constraints']
        self.assertEqual(with_constraints['foreground_supported_voxels'],
                         without_constraints['foreground_supported_voxels'])
        self.assertEqual(with_constraints['outer_hull_height_m'],
                         without_constraints['outer_hull_height_m'])

    def test_known_box_dimensions_and_order(self):
        row,masks,views = box_observations()
        result = measure_box(row,masks,views)
        reversed_ = measure_box(row,masks,views[::-1])
        self.assertTrue(result['valid'],result)
        estimate = result['estimate']
        self.assertEqual(estimate,reversed_['estimate'])
        self.assertEqual(estimate['views_used'],3)
        self.assertAlmostEqual(estimate['height_m'],.06,delta=.006)
        bounds = np.array(estimate['bounds_environment_m'])
        np.testing.assert_allclose(bounds.mean(axis=0)[:2],[.06,0],atol=.006)

        self.assertLessEqual(bounds[0,0],.022)
        self.assertGreaterEqual(bounds[1,0],.098)

    def test_thin_object_does_not_acquire_a_tall_shape_prior(self):
        row,masks,views = box_observations(.006)
        result = measure_box(row,masks,views)
        self.assertTrue(result['valid'],result)
        self.assertAlmostEqual(result['estimate']['height_m'],.006,delta=.004)

    def test_partial_occlusion_is_not_negative_shape_evidence(self):
        row,masks,views = box_observations()
        x1,y1,x2,y2 = views[0]['bbox']
        views[0]['exclude_boxes'] = [[x1,y1,(x1+x2)/2,y2]]
        result = measure_box(row,masks,views)
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['estimate']['views_used'],3)
        self.assertAlmostEqual(result['estimate']['height_m'],.06,delta=.008)
        reversed_ = measure_box(row,masks,views[::-1])
        self.assertEqual(result['estimate'],reversed_['estimate'])

    def test_incompatible_view_is_not_dropped_to_make_a_valid_measurement(self):
        row,masks,views = box_observations()
        masks[views[-1]['camera']][:] = False
        masks[views[-1]['camera']][10:20,10:20] = True
        result = measure_box(row,masks,views)
        self.assertFalse(result['valid'],result)

    def test_known_robot_interval_uses_current_pose(self):
        row = dict(camera_calibration={'cam_head':calibration([0,0,1])})
        view = rgb.camera_view(row,'cam_head')
        geometry = dict(outer_bounds=dict(base=[[-.1,-.1,.1],[.1,.1,.2]],
                         fingers_at_zero_joint=[],finger_joint_axes=[]))
        state = dict(left_ee_pose=[0,0,0,1,0,0,0],left_gripper_joint_positions=[])
        near = rgb.tool_occlusion_near(view,state,geometry)
        self.assertAlmostEqual(near[120,160],.8,places=10)
        self.assertTrue(np.isinf(near[0,0]))
        state['left_ee_pose'][2] = .1
        moved = rgb.tool_occlusion_near(view,state,geometry)
        self.assertAlmostEqual(moved[120,160],.7,places=10)

    def test_occlusion_allows_behind_but_not_front_of_known_tool(self):
        from seek.policy.silhouette_hull import _allows
        from seek.policy.calibrated_geometry import MaskObservation as Sight
        view = rgb.camera_view(dict(camera_calibration={'cam_head':calibration([0,0,1])}),'cam_head')
        mask = np.zeros(view.shape,bool)
        unknown = np.ones(view.shape,bool)
        observation = Sight(view,mask,unknown)
        points = np.array([[0,0,.4],[0,0,0]])
        permits,_,_ = _allows(observation,points,np.full(view.shape,.8))
        np.testing.assert_array_equal(permits,[False,True])
        foreground = Sight(view,np.ones(view.shape,bool),unknown)
        permits,_,_ = _allows(foreground,points,np.full(view.shape,.8))
        np.testing.assert_array_equal(permits,[True,True])


if __name__ == '__main__':
    unittest.main()
