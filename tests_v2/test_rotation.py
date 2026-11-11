from dataclasses import replace
import unittest
import numpy as np
from selfwatch_v2.contracts import FrameRef,Instance,Observation
from selfwatch_v2.rotation import RotationPersonPerception,inverse_point,inverse_instance


def frame(t=0.,epoch='e'):
    return FrameRef('A',epoch,0,t,1.,200,100,'test')


class Base:
    provenance={'sha256':'fixture','task':'segment'}
    def __init__(self):self.calls=[]
    def predict(self,f,image):
        self.calls.append((f.width,f.height))
        if len(self.calls)%2:
            people=(Instance((5,5,15,15),.8,41,'cup'),)
        else:
            people=(Instance((40,10,80,40),.9,13,'person',((69,10),(39,39),(49,20)),((69,10,.8),)),)
        return Observation(f,people,'fixture','segment',0),None


class RotationTests(unittest.TestCase):
    def test_rectangular_geometry_box_edges_and_landmark_centers(self):
        p=Instance((40,10,80,40),.9,13,'person',((69,10),(39,39)),((69,10,.8),))
        restored=inverse_instance(p,frame(),90)
        self.assertEqual(restored.box_xyxy,(10.,20.,40.,60.))
        self.assertEqual(restored.mask_polygon_xy[0],(10.,30.))
        self.assertEqual(restored.keypoints_xyc[0],(10.,30.,.8))
        self.assertEqual(restored.class_id,13)

    def test_all_rotations_roundtrip_pixel_points(self):
        point=(17.,42.)
        for angle in (0,90,180,270):
            rotated=point if angle==0 else (99-point[1],point[0]) if angle==90 else (199-point[0],99-point[1]) if angle==180 else (point[1],199-point[0])
            self.assertEqual(inverse_point(rotated,200,100,angle),point)

    def test_extra_people_are_real_detections_and_original_objects_survive(self):
        base=Base();model=RotationPersonPerception(base,angles=(90,))
        f=frame();image=np.zeros((100,200,3),dtype=np.uint8)
        result,detail=model.predict(f,image)
        self.assertEqual(base.calls,[(200,100),(100,200)])
        self.assertEqual(result.frame,f)
        self.assertEqual([p.label for p in result.instances],['cup','person'])
        self.assertEqual(result.instances[1].box_xyxy,(10.,20.,40.,60.))
        self.assertEqual(detail['source_rotation_deg'],[0,90])
        self.assertNotEqual(result.model,'fixture')

    def test_probe_throttle_and_reconnect_scope(self):
        class Empty(Base):
            def predict(self,f,image):
                self.calls.append((f.width,f.height));return Observation(f,(),'fixture','segment',0),None
        base=Empty();model=RotationPersonPerception(base,angles=(90,),probe_interval_s=.5)
        image=np.zeros((100,200,3),dtype=np.uint8)
        model.predict(frame(),image)
        _,info=model.predict(frame(.1),image)
        self.assertTrue(info['throttled']);self.assertEqual(len(base.calls),3)
        model.predict(frame(.1,'reconnect'),image)
        self.assertEqual(len(base.calls),5)

    def test_strong_other_person_does_not_mask_a_memory_recovery_request(self):
        class Strong(Base):
            def predict(self,f,image):
                self.calls.append(1);return Observation(f,(Instance((5,5,25,45),.9,0,'person'),),'fixture','segment',0),None
        base=Strong();model=RotationPersonPerception(base,mode='memory',angles=(90,))
        image=np.zeros((100,200,3),dtype=np.uint8);f=frame()
        model.set_recovery_regions(f,[(100,20,150,90)])
        _,info=model.predict(f,image)
        self.assertTrue(info['unmatched_memory']);self.assertEqual(len(base.calls),2)

    def test_duplicates_fuse_without_promoting_scores(self):
        class Duplicate(Base):
            def predict(self,f,image):
                p=Instance((10,20,40,60),.7,0,'person') if f.width==200 else Instance((40,10,80,40),.6,0,'person')
                return Observation(f,(p,),'fixture','segment',0),None
        result,_=RotationPersonPerception(Duplicate(),mode='always',angles=(90,)).predict(frame(),np.zeros((100,200,3),dtype=np.uint8))
        self.assertEqual(len(result.instances),1);self.assertEqual(result.instances[0].score,.7)

    def test_recovery_region_from_another_frame_cannot_trigger_strong_view(self):
        class Strong(Base):
            def predict(self,f,image):
                self.calls.append(1);return Observation(f,(Instance((5,5,25,45),.9,0,'person'),),'fixture','segment',0),None
        base=Strong();model=RotationPersonPerception(base,mode='memory',angles=(90,))
        model.set_recovery_regions(frame(),[(100,20,150,90)])
        _,detail=model.predict(frame(.1),np.zeros((100,200,3),dtype=np.uint8))
        self.assertFalse(detail['triggered']);self.assertEqual(len(base.calls),1)

    def test_input_resolution_changes_policy_identity(self):
        base=Base();base.imgsz=640;base.confidence=.25
        first=RotationPersonPerception(base).provenance['sha256']
        base.imgsz=960
        self.assertNotEqual(first,RotationPersonPerception(base).provenance['sha256'])


if __name__=='__main__':unittest.main()
