import io
import unittest
import zipfile

import numpy as np

from scripts.download_wildtrack_pilot import member
from selfwatch_v2.hand_calibration import choose_threshold
from selfwatch_v2.wildtrack_tracking import marks_for_camera, pair_scores


class TrackingPilotTests(unittest.TestCase):
    def test_archive_workaround_still_rejects_corruption(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('labels.json',b'["published reference"]')
        buffer.size=len(buffer.getvalue())
        with zipfile.ZipFile(buffer) as archive:
            info=archive.getinfo('labels.json')
            info.header_offset+=2**32
            self.assertEqual(member(buffer,info),b'["published reference"]')
            info.CRC^=1
            with self.assertRaisesRegex(ValueError,'CRC'):
                member(buffer,info)

    def test_public_labels_preserve_identity_and_clip_partial_boxes(self):
        people=[dict(personID=12,views=[dict(viewNum=0,xmin=-4,ymin=5,xmax=9,ymax=20),
            dict(viewNum=2,xmin=-1,ymin=-1,xmax=-1,ymax=-1)])]
        self.assertEqual(marks_for_camera(people,1,10,10)[0]['xyxy'],[0,5,9,10])
        self.assertEqual(marks_for_camera(people,3,10,10),[])

    def test_pair_score_penalizes_missing_people_and_reports_false_sharing(self):
        labels=[dict(observation_id=oid,frame={'captured_at_s':1.5},
            marks=[{'subject':s} for s in subjects]) for oid,subjects in [('a','xyz'),('b','xyz')]]
        traces=[dict(observation_id=oid,matches=[dict(subject=s,global_id=g) for s,g in pairs])
            for oid,pairs in [('a',[('x',1),('y',2)]),('b',[('x',1),('z',2)])]]
        scores=pair_scores(labels,traces)
        self.assertEqual((scores['same_person_possible'],scores['same_person_both_matched'],
            scores['same_person_shared_id'],scores['different_person_shared_id']),(3,1,1,1))

    def test_calibration_abstains_when_accuracy_floor_is_unreachable(self):
        scores=np.array([.15,.25,.45,.50])
        chosen,_=choose_threshold(scores,np.array([True,True,False,False]),8)
        self.assertIsNone(chosen['threshold'])
        self.assertEqual(chosen['all_joint_coverage'],0)
        chosen,_=choose_threshold(scores,np.array([False,False,True,True]),8)
        self.assertEqual(chosen['precision'],1)
        self.assertEqual(chosen['correct_yield'],.25)


if __name__=='__main__':
    unittest.main()
