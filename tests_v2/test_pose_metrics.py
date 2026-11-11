import unittest
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from selfwatch_v2.pose_metrics import score_person,finalize,person_box
from selfwatch_v2.wholebody_benchmark import refine_indices,load_dataset
from selfwatch_v2.contracts import FrameRef,Instance,Observation
from selfwatch_v2.hand_refinement import hand_roi
from scripts.download_coco_wholebody import choose


def annotation():
    record=dict(bbox=[0,0,100,100],keypoints=[0,0,0]*17,foot_kpts=[0,0,0]*6,
        face_kpts=[0,0,0]*68,lefthand_kpts=[0,0,0]*21,righthand_kpts=[0,0,0]*21,
        foot_valid=False,face_valid=False,lefthand_valid=True,righthand_valid=False,
        lefthand_box=[10,10,10,10],righthand_box=[0,0,0,0],face_box=[0,0,0,0])
    record['keypoints'][:6]=[10,10,2,20,20,1]
    record['lefthand_kpts'][:3]=[10,10,.5]
    return record


class PoseMetricTests(unittest.TestCase):
    def test_absent_detection_keeps_the_joint_denominator(self):
        metrics,_=score_person(annotation(),None);summary=finalize(metrics)
        self.assertEqual(summary['body_visible']['eligible'],1)
        self.assertEqual(summary['body_visible']['PCK'],0.)
        self.assertEqual(summary['hands']['pose_availability'],0.)
        self.assertIsNone(summary['hands']['PCK_given_confident'])

    def test_confident_errors_are_not_reported_as_accuracy(self):
        predicted=np.zeros((133,3));predicted[0]=[10,10,.2];predicted[1]=[20,20,.9]
        predicted[91]=[12,10,.9] # 2px exceeds 10% of the 10px hand scale.
        metrics,_=score_person(annotation(),predicted);summary=finalize(metrics)
        self.assertEqual(summary['body_visible']['PCK'],1.)
        self.assertEqual(summary['body_visible']['gated_PCK'],0.)
        self.assertEqual(summary['body_occluded']['gated_PCK'],1.)
        self.assertEqual(summary['hands']['PCK'],0.)

    def test_invalid_parts_are_excluded_even_with_nonzero_points(self):
        record=annotation();record['lefthand_valid']=False
        metrics,_=score_person(record,np.zeros((133,3)))
        self.assertNotIn('hands',metrics)

    def test_float_hand_reliability_is_not_called_body_visibility(self):
        metrics,_=score_person(annotation(),np.zeros((133,3)))
        self.assertEqual(metrics['hands']['eligible'],1)
        self.assertEqual(metrics['body_labeled']['eligible'],2)

    def test_layout_and_nonfinite_outputs_rejected(self):
        for predicted in (np.zeros((17,3)),np.full((133,3),np.nan)):
            with self.assertRaises(ValueError):score_person(annotation(),predicted)
        self.assertEqual(person_box(dict(bbox=[-10,-10,30,40]),100,100),(0,0,20,30))

    def test_selection_is_independent_of_model_and_keeps_all_selected_image_people(self):
        data=dict(images=[dict(id=i) for i in range(16)],categories=[],annotations=[])
        for i in range(16):data['annotations'].append(dict(id=i,image_id=i,category_id=1,iscrowd=False,keypoints=[1,2,2]))
        data['annotations'].append(dict(id=100,image_id=1,category_id=1,iscrowd=True,keypoints=[0,0,0]))
        selection=choose(data,8)
        self.assertEqual([i['id'] for i in selection['images']],[1,3,5,7,9,11,13,15])
        self.assertIn(100,[a['id'] for a in selection['annotations']])

    def test_bounded_crop_batches_preserve_indices_and_unselected_objects(self):
        class FakePose:
            sizes=[]
            def refine(self,observation,image):
                self.sizes.append(len(observation.instances))
                return replace(observation,instances=tuple(replace(p,keypoints_xyc=((p.box_xyxy[0],0,.8),)*133) for p in observation.instances))
        frame=FrameRef('camera','epoch',0,0,0,100,100,'source')
        instances=tuple(Instance((i,0,i+10,20),.9,0,'person') for i in range(8))+(Instance((0,0,5,5),.8,1,'item'),)
        observation=Observation(frame,instances,'test','detect',0)
        pose=FakePose();result=refine_indices(pose,observation,None,list(range(7)))
        self.assertEqual(pose.sizes,[3,3,1])
        self.assertEqual([p.keypoints_xyc[0][0] for p in result.instances[:7]],list(range(7)))
        self.assertIsNone(result.instances[7].keypoints_xyc)
        self.assertEqual(result.instances[8],instances[8])

    def test_corrupted_dataset_cannot_reach_model_initialization(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);selection=root/'selection.json'
            selection.write_text(json.dumps(dict(role='public_validation_diagnostic_only',images=[dict(id=1)])))
            (root/'image.jpg').write_bytes(b'original')
            manifest=dict(status='complete',selection_sha256=sha256(selection.read_bytes()).hexdigest(),
                files=[dict(path='image.jpg',bytes=8,sha256=sha256(b'original').hexdigest())])
            (root/'download_manifest.json').write_text(json.dumps(manifest))
            load_dataset(root)
            (root/'image.jpg').write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError,'checksum'):load_dataset(root)

    def test_hand_roi_requires_supported_finite_landmarks(self):
        points=np.zeros((21,3));points[:7]=[10,10,.9]
        self.assertIsNone(hand_roi(points,100,100))
        points[:8]=[10,10,.9]
        roi=hand_roi(points,100,100)
        self.assertIsNotNone(roi)
        x1,y1,x2,y2=roi['box']
        self.assertTrue(0<=x1<x2<=100 and 0<=y1<y2<=100)
        points[0,0]=np.nan
        self.assertIsNone(hand_roi(points,100,100))

    def test_out_of_frame_predictions_do_not_create_a_fake_hand_crop(self):
        points=np.tile([500,500,.9],(21,1))
        self.assertIsNone(hand_roi(points,100,100))


if __name__=='__main__':unittest.main()
