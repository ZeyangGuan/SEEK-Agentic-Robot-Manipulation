import unittest
import numpy as np
from seek.policy.calibrated_geometry import CalibratedCamera,HULL_RESOLUTION
from seek.policy.rgb_perception import locate_rgb_point
from test_rgb_perception import scene

class PixelContractTests(unittest.TestCase):
    def test_direct_intrinsics_match_configured_projection_and_rays(self):
        rng=np.random.default_rng(13)
        for angle in (0.,.31,-.82):
            c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=[.2,-.3,1.2]
            camera=CalibratedCamera([[397.,0,320.],[0,288.,240.],[0,0,1.]],pose,(480,640))
            local=rng.uniform([-.2,-.2,-2.],[.2,.2,-.4],size=(80,3))
            points=local @ rotation.T+pose[:3,3]
            row,col,depth=camera.project(points)
            np.testing.assert_allclose(row,240.-288.*local[:,1]/(-local[:,2]),atol=1e-11)
            np.testing.assert_allclose(col,320.+397.*local[:,0]/(-local[:,2]),atol=1e-11)
            np.testing.assert_allclose(depth,-local[:,2],atol=1e-12)
            pixels=rng.uniform([0,0],[479,639],size=(80,2))
            local_rays=np.column_stack(((pixels[:,1]-320.)/397.,-(pixels[:,0]-240.)/288.,-np.ones(80)))
            np.testing.assert_allclose(camera.rays(pixels[:,0],pixels[:,1]),local_rays @ rotation.T,atol=1e-12)
        self.assertEqual((HULL_RESOLUTION.cell_m,HULL_RESOLUTION.xy_samples,HULL_RESOLUTION.level_m,HULL_RESOLUTION.max_height_m),(.001,40000,.002,.36))

    def test_public_normalized_coordinates_preserve_existing_half_pixel_conversion(self):
        observation=scene();point=np.array([.08,.04,.07]);samples=[]
        for name,cal in observation['camera_calibration'].items():
            pose=np.array(cal['camera_to_environment']);k=np.array(cal['intrinsic_matrix'])
            local=pose[:3,:3].T@(point-pose[:3,3])
            optical=local*np.array([1.,-1.,-1.]);pixel=k@optical;pixel=pixel[:2]/pixel[2]
            samples.append(dict(camera=name,uv=[(pixel[0]+.5)/320,(pixel[1]+.5)/240]))
        result=locate_rgb_point({0:observation},0,samples)
        self.assertTrue(result['valid'])
        self.assertEqual(result['solver'],'calibrated_dlt')
        np.testing.assert_allclose(result['point_environment_m'],point,atol=1e-10)

if __name__=='__main__':unittest.main()
