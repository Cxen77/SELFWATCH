import json
import unittest
import numpy as np
from selfwatch_v2.mot_failure_analysis import clear_trace,episodes,recovery


class FailureAnalysisTests(unittest.TestCase):
    def test_trace_localizes_both_sides_of_identity_swap(self):
        data=dict(num_gt_ids=2,gt_ids=[np.array([0,1])]*4,
            tracker_ids=[np.array([0,1])]*2+[np.array([1,0])]*2,
            similarity_scores=[np.eye(2)]*4)
        _,events=clear_trace(data)
        self.assertEqual(len(events),2);self.assertEqual([r['frame'] for r in events],[3,3])

    def test_clean_return_requires_two_same_id_observations(self):
        rows=np.array([[i+1,42,0,0,10,20,1,1,v] for i,v in enumerate([.9,.9,.2,.2,.2,.9,.9,.9])])
        events,visibility=episodes(rows,8);self.assertEqual(len(events),1)
        self.assertEqual(json.loads(json.dumps(events))[0]['original_subject'],42)
        matched=[{0:10} for _ in range(8)]
        result=recovery(events[0],matched,visibility,8)
        self.assertEqual(result['status'],'same_id_recovered');self.assertAlmostEqual(result['delay_s'],1/30)
        for i in range(5,8):matched[i]={0:11}
        self.assertEqual(recovery(events[0],matched,visibility,8)['status'],'not_recovered_within_1s')

    def test_missing_pre_identity_is_not_dropped_from_denominator(self):
        rows=np.array([[i+1,42,0,0,10,20,1,1,v] for i,v in enumerate([.9,.9,.2,.2,.2,.9,.9,.9])])
        events,visibility=episodes(rows,8);matched=[{},{}]+[{0:10} for _ in range(6)]
        self.assertEqual(recovery(events[0],matched,visibility,8)['status'],'no_pre_occlusion_identity')
