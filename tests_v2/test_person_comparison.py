import copy
import unittest
from scripts.finish_person_pilot import compare


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        return dict(split_lock={'selection_sha256':'locked'},protocol={'iou':.5},
            frames=[dict(image_id='image1',partition='validation',processed_pixel_sha256='pixels',ground_truth=[])],
            metrics={role:{'native_640':dict(F1=.5,recall=.4)} for role in ('development','validation')})

    def test_identical_input_comparison_reports_delta(self):
        baseline=self.fixture();adapted=copy.deepcopy(baseline)
        adapted['metrics']['validation']['native_640']['F1']=.6
        self.assertAlmostEqual(compare(baseline,adapted)['validation']['native_640']['F1_delta'],.1)

    def test_changed_pixels_targets_or_protocol_cannot_be_compared(self):
        baseline=self.fixture()
        for field,value in (('processed_pixel_sha256','different'),('ground_truth',[{'box':[1,2,3,4]}])):
            adapted=copy.deepcopy(baseline);adapted['frames'][0][field]=value
            with self.assertRaises(ValueError):compare(baseline,adapted)
        adapted=copy.deepcopy(baseline);adapted['protocol']['iou']=.7
        with self.assertRaises(ValueError):compare(baseline,adapted)


if __name__=='__main__':unittest.main()
