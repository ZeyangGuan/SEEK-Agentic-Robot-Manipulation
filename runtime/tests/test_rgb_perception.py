import base64
import itertools
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from seek.policy.model_policy import Model, CAMERAS
from seek.policy.rgb_perception import camera_view, locate_rgb_point, measure_rgb, point_statistics, tool_pixels, _cache
from seek.policy.segmented_geometry import encode_mask
from seek.policy.execution_policy import TOOL_GEOMETRY, empty_memory


def calibration(position):
    position = np.asarray(position, float)
    back = position / np.linalg.norm(position)
    right = np.cross([0,1,0], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    pose = np.eye(4); pose[:3,:3] = np.column_stack([right, up, back]); pose[:3,3] = position
    return dict(intrinsic_matrix=[[180,0,160],[0,180,120],[0,0,1]],
        camera_to_environment=pose.tolist(),shape=[240,320,3])


def scene():
    return dict(step=0,state={},images={c:c for c in CAMERAS},
        camera_calibration={c:calibration(p) for c,p in zip(CAMERAS,[[0,-.6,1],[-.5,.1,.8],[.5,.1,.8]])})


class RGBPerceptionTests(unittest.TestCase):
    def setUp(self):
        _cache.clear()

    def test_triangulation_reconstructs_visible_point_without_depth(self):
        row=scene(); point=np.array([.08,.04,.07]); samples=[]
        for c in CAMERAS:
            view=camera_view(row,c); r,col,_=view.project(point[None,:])
            samples.append(dict(camera=c,uv=[(col[0]+.5)/320,(r[0]+.5)/240]))
        result=locate_rgb_point({0:row},0,samples)
        self.assertTrue(result['valid'])
        np.testing.assert_allclose(result['point_environment_m'],point,atol=1e-10)
        self.assertLess(result['residual_m'],1e-10)
        self.assertGreater(result['ray_angle_deg'],20)
        self.assertNotIn('depth',row)

    def test_wrong_correspondence_exposes_residual(self):
        row=scene(); samples=[]
        for i,c in enumerate(CAMERAS):
            view=camera_view(row,c); r,col,_=view.project(np.array([[.05,.02,.08]]))
            samples.append(dict(camera=c,uv=[(col[0]+.5+(25 if i==1 else 0))/320,(r[0]+.5)/240]))
        result=locate_rgb_point({0:row},0,samples)
        self.assertGreater(result['residual_m'],.01)
        self.assertGreater(max(result['reprojection_error_px']),5)

    def test_single_view_requires_declared_plane(self):
        row=scene(); samples=[dict(camera=CAMERAS[0],uv=[.5,.5])]
        with self.assertRaises(ValueError): locate_rgb_point({0:row},0,samples)
        result=locate_rgb_point({0:row},0,samples,0.)
        self.assertEqual(result['source'],'rgb_plane_hypothesis')
        self.assertAlmostEqual(result['point_environment_m'][2],0.)

    def test_silhouette_shape_from_three_rgb_views(self):
        row=scene(); masks={}; selections=[]
        corners=np.array(list(itertools.product([.02,.10],[-.03,.03],[0,.06])))
        for c in CAMERAS:
            view=camera_view(row,c); r,col,_=view.project(corners)
            mask=np.zeros(view.shape,np.uint8)
            cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(np.c_[col,r]).astype(np.int32)),1)
            masks[c]=mask.astype(bool)
            selections.append(dict(camera=c,bbox=[max(0,(col.min()-2)/320),max(0,(r.min()-2)/240),
                min(1,(col.max()+3)/320),min(1,(r.max()+3)/240)]))
        segment=lambda image,box:dict(valid=True,score=1.,mask=encode_mask(masks[image]))
        pose=[.3,-.2,.4,2**-.5,0.,0.,2**-.5]
        result=measure_rgb({0:row},0,selections,0.,reference_pose=pose,segmenter=segment)
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['source'],'rgb_silhouette_hypothesis')
        self.assertGreaterEqual(result['estimate']['views_used'],2)
        self.assertAlmostEqual(result['estimate']['height_m'],.06,delta=.015)
        np.testing.assert_allclose(np.mean(result['estimate']['bounds_environment_m'],axis=0)[:2],[.06,0],atol=.015)
        self.assertNotIn('surface',result)
        reference=result['estimate']['reference_frame']
        self.assertEqual(reference['pose'],pose)
        point=result['estimate']['point_environment_m']
        np.testing.assert_allclose(reference['point_m'],[point[1]+.2,-(point[0]-.3),point[2]-.4],atol=1e-10)

    def test_reference_frame_preserves_metric_points_and_rotation(self):
        points=np.array([[1.,2.,3.],[2.,4.,6.],[3.,6.,9.]])
        pose=[1.,2.,3.,2**-.5,0.,0.,2**-.5]
        result=point_statistics(points,pose)
        reference=result['reference_frame']
        expected=np.array([[0.,0.,0.],[2.,-1.,3.],[4.,-2.,6.]])
        np.testing.assert_allclose(reference['point_m'],np.median(expected,axis=0),atol=1e-12)
        np.testing.assert_allclose(reference['bounds_m'],[expected.min(axis=0),expected.max(axis=0)],atol=1e-12)
        np.testing.assert_allclose(reference['p10_p90_m'],np.percentile(expected,[10,90],axis=0),atol=1e-12)
        np.testing.assert_array_equal(points,[[1.,2.,3.],[2.,4.,6.],[3.,6.,9.]])

    def test_tool_occlusion_uses_measured_joint_positions(self):
        row=scene(); view=camera_view(row,CAMERAS[0])
        state=dict(left_ee_pose=[0,0,.2,1,0,0,0],left_gripper_joint_positions=[.04,.04])
        mask=tool_pixels(view,state,TOOL_GEOMETRY)
        self.assertGreater(mask.sum(),10)
        state['left_gripper_joint_positions']=[0,0]
        self.assertTrue(np.any(mask != tool_pixels(view,state,TOOL_GEOMETRY)))

    def test_observation_boundary_discards_even_supplied_depth(self):
        model=object.__new__(Model); model.archive={}; model.head_calibration_context=None
        model._log=Mock(); model.ledger=[]
        obs=dict(model_meta=dict(step=0),state={},instruction='sort',vision={})
        for c in CAMERAS:
            obs['vision'][c]=dict(color=np.ones((240,320,3),np.uint8)*120,
                depth=object(),depth_calibration=object(),rgb_calibration=calibration([0,-.6,1]))
        model.update_obs(obs)
        self.assertNotIn('depth',model.archive[0])
        self.assertEqual(set(model.archive[0]['camera_calibration']),set(CAMERAS))
        self.assertEqual(model.archive[0]['perception_mode'],'rgb_only')

    def test_fresh_memory_has_no_incompatible_depth_seed(self):
        for task in ('classify_objects','classify_objects_by_language'):
            memory=empty_memory(task)
            self.assertEqual(memory['operation_knowledge']['selected'],{})
            self.assertFalse(memory['episodes'])


if __name__=='__main__': unittest.main()
