import copy
import unittest
from selfwatch_v2.person_error_analysis import size_bucket,detected_targets,summarize_pair


class PersonErrorTests(unittest.TestCase):
    def fixture(self):
        result=dict(predictions=[{'box_xyxy':[0,0,10,10]}],matches=[dict(gt=0,prediction=0,iou=.9)],
            true_positive=1,false_positive=0,false_negative=1)
        frame=dict(image_id='a',partition='validation',source_group='scene',processed_pixel_sha256='same',
            width=100,height=100,ground_truth=[dict(box_xyxy=[0,0,2,2]),dict(box_xyxy=[10,10,40,40])],cases={'native_640':result})
        return dict(split_lock={},protocol={},frames=[frame],metrics={r:{'native_640':dict(F1=.5)} for r in ('development','validation')})

    def test_relative_area_bins_use_original_dimensions(self):
        self.assertEqual(size_bucket((0,0,2,2),100,100),'below_0.1pct')
        self.assertEqual(size_bucket((0,0,10,10),200,100),'0.1_to_1pct')
        self.assertEqual(size_bucket((0,0,10,10),100,100),'at_least_1pct')

    def test_rescue_regression_and_bucket_counts_reconcile(self):
        before=self.fixture();after=copy.deepcopy(before)
        after['frames'][0]['cases']['native_640']['matches'][0]['gt']=1
        report=summarize_pair(before,after)
        row=report['images'][0]
        self.assertEqual((row['rescued'],row['lost'],row['persistent_misses']),(1,1,0))
        self.assertEqual(sum(r['models']['adapted']['targets'] for r in report['sizes']),2)
        self.assertEqual(sum(r['models']['adapted']['recalled'] for r in report['sizes']),1)

    def test_malformed_saved_matches_are_rejected(self):
        frame=self.fixture()['frames'][0]
        frame['cases']['native_640']['matches'][0]['gt']=99
        with self.assertRaises(ValueError):detected_targets(frame,'native_640')

    def test_final_test_cannot_enter_diagnostics(self):
        before=self.fixture();before['frames'][0]['partition']='sealed_test'
        with self.assertRaises(ValueError):summarize_pair(before,copy.deepcopy(before))


if __name__=='__main__':unittest.main()
