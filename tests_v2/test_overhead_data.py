from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
from selfwatch_v2.overhead_data import contained,ground_truth_polygon,in_roi,match_boxes,load_selection
from selfwatch_v2.overhead_review import build
from selfwatch_v2.prepare_overhead_training import label_line
from scripts.download_wepdtof_subset import safe_target


class OverheadDataTests(unittest.TestCase):
    def test_rotated_box_geometry_and_clipping(self):
        points=ground_truth_polygon((50,50,20,40,90),100,100)
        self.assertAlmostEqual(min(x for x,y in points),30)
        self.assertAlmostEqual(max(x for x,y in points),70)
        self.assertAlmostEqual(min(y for x,y in points),40)
        clipped=ground_truth_polygon((5,5,20,20,45),100,100)
        self.assertTrue(all(0<=x<=100 and 0<=y<=100 for x,y in clipped))

    def test_source_roi_center_rule(self):
        mask=np.zeros((100,200),dtype=np.uint8);mask[:,100:]=255
        self.assertFalse(in_roi((0,0,40,40),mask))
        self.assertTrue(in_roi((110,20,150,80),mask))
        self.assertFalse(in_roi((-10,20,10,20,0),mask,center=True))

    def test_one_to_one_matches_and_missing_people(self):
        result=match_boxes([(10,10,40,90),(60,10,90,90)],[(10,10,40,90),(10,10,40,90)])
        self.assertEqual((result['true_positive'],result['false_positive'],result['false_negative']),(1,1,1))
        self.assertEqual(match_boxes([],[(0,0,10,10)])['false_positive'],1)

    def test_untrusted_dataset_paths_do_not_escape_or_alias_windows_devices(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('../escape.jpg','C:/escape.jpg','a\\b.jpg','a/CON.txt','a/.. /b.jpg'):
                with self.assertRaises(ValueError):safe_target(root,name)
            with self.assertRaises(ValueError):contained(root,'../escape.jpg')
            self.assertTrue(safe_target(root,'frames/a.jpg').is_relative_to(root))

    def test_scene_split_preserves_sealed_groups_and_renderer_rejects_leakage(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);groups=[f'scene{i}' for i in range(14)]
            (root/'selection.json').write_text(json.dumps(dict(status='complete',group_order=groups)))
            (root/'download_manifest.json').write_text(json.dumps(dict(status='complete',files=[])))
            _,roles=load_selection(root)
            self.assertEqual([roles[g] for g in groups],['development']*7+['validation']*3+['sealed_test']*4)
            (root/'data.json').write_text(json.dumps(dict(frames=[dict(partition='sealed_test')])))
            with self.assertRaisesRegex(ValueError,'final-test'):build(root)

    def test_exported_target_uses_original_rectangular_image_dimensions(self):
        values=list(map(float,label_line((20,10,100,90),200,100).split()))
        self.assertEqual(values,[0.,.3,.5,.4,.8])
        with self.assertRaises(ValueError):label_line((0,0,300,100),200,100)


if __name__=='__main__':unittest.main()
