import unittest
import numpy as np
from selfwatch_v2.ground_link import GroundCamera,GroundLinker,mutual_pairs


class GroundLinkTests(unittest.TestCase):
    def test_floor_projection_and_clipped_feet_abstain(self):
        camera=GroundCamera(np.array([[100,0,20],[0,100,20],[0,0,1]]))
        np.testing.assert_allclose(camera.foot((100,20,140,220)),[1,2])
        self.assertIsNone(camera.foot((0,20,140,220)))
        self.assertIsNone(camera.foot((100,20,140,1080)))
        with self.assertRaises(ValueError):GroundCamera(np.zeros((3,3)))

    def test_mutual_nearest_rejects_close_competitor(self):
        self.assertEqual(mutual_pairs({'a':(0,0)},{'b':(.1,0),'c':(.2,0)},.75,.25),[])

    def test_two_sightings_clean_appearance_and_gap_are_required(self):
        a,b=('A','e',1),('B','e',1)
        views={'A':{a:np.array([0.,0.])},'B':{b:np.array([.1,0.])}}
        linker=GroundLinker(.75,.25,min_cosine=.55)
        for t in (0,.5):
            ids=linker.update(t,views,{a:np.array([1.,0.]),b:np.array([0.,1.])})
            self.assertNotEqual(ids[a],ids[b])
        features={a:np.array([1.,0.]),b:np.array([1.,0.])}
        ids=linker.update(1.,views,features);self.assertNotEqual(ids[a],ids[b])
        ids=linker.update(2.,views,features);self.assertNotEqual(ids[a],ids[b])
        ids=linker.update(2.5,views,features);self.assertEqual(ids[a],ids[b])
        with self.assertRaises(ValueError):linker.update(2.5,views,features)

    def test_same_camera_component_collision_is_not_merged(self):
        a,b,c=('A','e',1),('B','e',1),('A','e',2)
        linker=GroundLinker(.75,.25)
        for t in (0,.5):ids=linker.update(t,{'A':{a:np.array([0.,0.])},'B':{b:np.array([.1,0.])}})
        self.assertEqual(ids[a],ids[b])
        for t in (1.,1.5):ids=linker.update(t,{'A':{c:np.array([0.,0.])},'B':{b:np.array([.1,0.])}})
        self.assertNotEqual(ids[c],ids[b])

    def test_transitive_chain_cannot_span_radius(self):
        keys=[(camera,'e',1) for camera in 'ABC']
        linker=GroundLinker(.75,.25)
        views={key[0]:{key:np.array([i*.6,0.])} for i,key in enumerate(keys)}
        for t in (0,.5):ids=linker.update(t,views)
        self.assertNotEqual(ids[keys[0]],ids[keys[2]])
