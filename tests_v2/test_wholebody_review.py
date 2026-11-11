import importlib.util
import unittest
from types import SimpleNamespace
from selfwatch_v2.contracts import FrameRef, Instance, Observation
from selfwatch_v2.wholebody import WholeBody
from selfwatch_v2.evaluate_review import evaluate


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires numpy")
class WholeBodyTests(unittest.TestCase):
    def adapter(self):
        import numpy as np
        adapter = WholeBody.__new__(WholeBody)
        adapter.max_people = 1
        adapter.provenance = {"sha256": "pose"}
        self.calls = []
        def predict(image, bboxes):
            self.calls.append(bboxes)
            return np.ones((len(bboxes),133,2)), np.ones((len(bboxes),133))*.8
        adapter.model = predict
        return adapter

    def observation(self, people):
        f = FrameRef("A","r",0,0,1000,640,480,"test")
        return Observation(f, tuple(people), "det", "segment", 1)

    def test_empty_people_never_trigger_whole_image_fallback(self):
        adapter = self.adapter()
        adapter.refine(self.observation([]), None)
        self.assertEqual(self.calls, [])

    def test_only_person_boxes_and_explicit_budget(self):
        adapter = self.adapter()
        items = [Instance((0,0,10,10),.9,0,"head"),
                 Instance((10,20,30,40),.8,1,"person"),
                 Instance((40,50,60,70),.7,1,"person")]
        result = adapter.refine(self.observation(items), None)
        self.assertEqual(self.calls, [[(10,20,30,40)]])
        self.assertIsNone(result.instances[0].keypoints_xyc)
        self.assertEqual(len(result.instances[1].keypoints_xyc), 133)
        self.assertIsNone(result.instances[2].keypoints_xyc)

    def test_wrong_topology_rejected(self):
        import numpy as np
        adapter = self.adapter()
        adapter.model=lambda *a,**k:(np.zeros((1,17,2)),np.zeros((1,17)))
        with self.assertRaises(ValueError):
            adapter.refine(self.observation([Instance((0,0,10,10),.8,0,"person")]),None)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Requires scipy")
class ReviewTests(unittest.TestCase):
    def test_unreviewed_and_partial_frames_are_not_ground_truth(self):
        data={"run_id":"x","frames":[{"observation_id":"a","instances":[]}]}
        review={"run_id":"x","frames":[{"observation_id":"a","reviewed":True,"marks":[]}]}
        self.assertEqual(evaluate(data,review)['models']['YOLO + RTMW']['precision'],None)

    def test_duplicate_person_predictions_are_counted_as_false_positive(self):
        box=[0,0,20,20]
        data={"run_id":"x","frames":[{"observation_id":"a","instances":[{"label":"person","box_xyxy":box}]*2}]}
        review={"run_id":"x","frames":[{"observation_id":"a","reviewed":True,"exhaustive_labels":["person"],
                "marks":[{"label":"person","kind":"box","xyxy":box}]}]}
        result=evaluate(data,review)['models']['YOLO + RTMW']
        import json
        json.dumps(result)  # Real reports must contain JSON-native numbers.
        self.assertEqual(result['true_positive'],1)
        self.assertEqual(result['false_positive'],1)
        self.assertEqual(result['recall'],1)
