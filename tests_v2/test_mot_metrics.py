from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from selfwatch_v2.mot_metrics import evaluate_mot


@unittest.skipUnless(Path('.runtime/TrackEval-numpy2/provenance.json').exists(),'Optional pinned TrackEval not installed')
class MotMetricsTests(unittest.TestCase):
    def fixture(self,root,swap=False,distractor=False):
        gt=root/'gt'/'fixture'/'gt';gt.mkdir(parents=True)
        tracker=root/'predictions'/'trial'/'data';tracker.mkdir(parents=True)
        truth=[];pred=[]
        for frame in range(1,5):
            for person in (1,2):
                x=person*100
                truth.append(f'{frame},{person},{x},10,40,80,1,1,1')
                predicted=3-person if swap and frame>2 else person
                pred.append(f'{frame},{predicted},{x},10,40,80,1,-1,-1,-1')
            if distractor:
                truth.append(f'{frame},3,400,10,40,80,1,8,1')
                pred.append(f'{frame},3,400,10,40,80,1,-1,-1,-1')
        (gt/'gt.txt').write_text('\n'.join(truth));(tracker/'fixture.txt').write_text('\n'.join(pred))
        return evaluate_mot(root/'gt',root/'predictions','fixture',4,'trial')['summary']

    def test_perfect_identity_and_standard_distractor_preprocessing(self):
        with TemporaryDirectory() as temp:r=self.fixture(Path(temp),distractor=True)
        self.assertEqual(r['IDF1'],1);self.assertEqual(r['HOTA'],1);self.assertEqual(r['IDSW'],0)
        self.assertEqual(r['predicted_detections'],8)

    def test_crossing_identity_swap_is_counted_even_with_perfect_boxes(self):
        with TemporaryDirectory() as temp:r=self.fixture(Path(temp),swap=True)
        self.assertEqual(r['IDSW'],2);self.assertAlmostEqual(r['IDF1'],.5)
        self.assertLess(r['HOTA'],.8);self.assertEqual(r['CLR_FN'],0)
