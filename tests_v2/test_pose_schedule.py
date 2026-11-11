from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from selfwatch_v2.contracts import FrameRef,Instance,Observation
from selfwatch_v2.pose_schedule import ScheduledPose
from selfwatch_v2.appearance import AppearanceEncoder


def observation(t=0,camera='A',epoch='e',people=5):
    frame=FrameRef(camera,epoch,round(t*100),t,0,256,128,'fixture')
    return Observation(frame,tuple(Instance((i*35+2,2,i*35+25,110),.9,0,'person') for i in range(people)),'detector','detect',3)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def refine(obs,image):
            self.calls.append(obs)
            return replace(obs,instances=tuple(replace(p,keypoints_xyc=((1,2,.8),)) for p in obs.instances),keypoint_schema='fixture',inference_ms=7)
        self.pose=ScheduledPose(SimpleNamespace(provenance={'schema':'fixture'},refine=refine))

    def test_optional_pose_rotates_without_losing_other_objects(self):
        seen=[]
        for t in (0,.5,1.):
            original=observation(t);result=self.pose.refine(original,None)
            self.assertEqual([p.box_xyxy for p in result.instances],[p.box_xyxy for p in original.instances])
            seen.extend(i for i,p in enumerate(result.instances) if p.keypoints_xyc)
        self.assertEqual(seen,[0,1,2,3,0,4])

    def test_skipped_frame_does_not_inherit_stale_landmarks(self):
        self.pose.refine(observation(),None)
        original=observation(.1);result=self.pose.refine(original,None)
        self.assertIs(result,original);self.assertTrue(all(p.keypoints_xyc is None for p in result.instances))
        self.assertEqual(self.pose.last_diagnostics['posed_people'],0)

    def test_camera_and_epoch_scheduling_is_independent_and_bounded(self):
        for camera,epoch in [('A','e'),('B','e'),('A','restart')]:
            self.pose.refine(observation(camera=camera,epoch=epoch),None)
        self.assertEqual(len(self.calls),3)
        for n in range(20):self.pose.refine(observation(camera=f'C{n}'),None)
        self.assertEqual(len(self.pose.states),16)
        with self.assertRaisesRegex(ValueError,'backward'):
            self.pose.refine(observation(.1,camera='C19'),None)
            self.pose.refine(observation(0,camera='C19'),None)

    def test_no_people_does_not_spend_interval_budget(self):
        self.pose.refine(observation(people=0),None)
        self.pose.refine(observation(.1),None)
        self.assertEqual(len(self.calls),1)

    def test_padding_does_not_emit_extra_people_or_embeddings(self):
        encoder=AppearanceEncoder.__new__(AppearanceEncoder)
        encoder.states={};encoder.attention={};encoder.batch_buckets=True;encoder.refresh_s=.5
        calls=[]
        def extract(crops):calls.append(len(crops));return np.ones((len(crops),512))
        encoder.extractor=SimpleNamespace(extract_batch=extract)
        obs=observation(people=5)
        vectors,diagnostics,_=encoder.extract_all(obs,np.zeros((128,256,3),dtype=np.uint8))
        self.assertEqual(calls,[8]);self.assertEqual(set(vectors),set(range(5)))
        self.assertEqual(len(diagnostics),5)
