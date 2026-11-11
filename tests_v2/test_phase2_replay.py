from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.validate_phase2 import OrientedDetector, SampledHand
from selfwatch_v2.contracts import FrameRef, Instance, Observation


class Phase2ReplayTests(unittest.TestCase):
    def test_orientation_preserves_original_frame_and_restores_boxes(self):
        frame = FrameRef('A','clip',0,0,0,120,80,'test')

        def predict(rotated_frame, image):
            self.assertEqual(image.shape[:2], (120,80))
            person = Instance((30,20,70,60),.9,0,'person')
            return Observation(rotated_frame,(person,),'test','detect',1), None

        wrapped = OrientedDetector(SimpleNamespace(predict=predict), {'A':90})
        result, _ = wrapped.predict(frame, np.zeros((80,120,3),dtype=np.uint8))
        self.assertEqual(result.frame, frame)
        self.assertEqual(result.instances[0].box_xyxy, (20,10,60,50))

    def test_hand_budget_is_per_view_and_low_scores_keep_fresh_body_points(self):
        frame = FrameRef('A','clip',0,0,0,100,100,'test')
        points = tuple((30.,30.,.8) for _ in range(133))
        person = Instance((10,10,80,80),.9,0,'person',keypoints_xyc=points)
        observation = Observation(frame,(person,),'test','detect+wholebody',1)
        calls = []

        def predict(image, box):
            calls.append(box)
            result = np.tile([45.,45.,.2],(21,1))
            result[0,2] = .8
            return result

        mode = SampledHand(SimpleNamespace(refine=lambda obs,img:obs),
                           SimpleNamespace(predict=predict),.5,.35)
        image = np.zeros((100,100,3),dtype=np.uint8)
        first = mode.refine(observation,image)
        self.assertEqual(first.instances[0].keypoints_xyc[91],(45.,45.,.8))
        self.assertEqual(first.instances[0].keypoints_xyc[92],points[92])
        later = replace(observation, frame=replace(frame,frame_index=1,captured_at_s=.2))
        self.assertEqual(mode.refine(later,image).instances[0].keypoints_xyc,points)
        mode.refine(replace(later, frame=replace(later.frame,camera_id='B')),image)
        mode.refine(replace(later, frame=replace(later.frame,captured_at_s=.5)),image)
        self.assertEqual(len(calls),3)
        self.assertEqual(mode.calls,{'A':2,'B':1})
