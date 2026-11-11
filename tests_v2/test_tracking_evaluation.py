from copy import deepcopy
import unittest
from selfwatch_v2.evaluate_tracking import evaluate


def fixtures(ids=(1,1),cameras=('A','B')):
    frames=[];annotations=[]
    for i,(gid,camera) in enumerate(zip(ids,cameras)):
        f=dict(camera_id=camera,stream_epoch='s',frame_index=i,captured_at_s=float(i),width=100,height=100,source='synthetic')
        oid=f'{camera}/{i}'
        frames.append(dict(frame=f,observation_id=oid,instances=[dict(label='person',box_xyxy=[10,10,50,90])],
            tracks=[dict(instance_index=0,global_id=gid,local_id=1,status='observed')]))
        annotations.append(dict(frame=f,observation_id=oid,reviewed=True,exhaustive_labels=['person'],
            marks=[dict(label='person',kind='box',xyxy=[10,10,50,90],subject='shopper A',source='manual')]))
    return dict(run_id='test',frames=frames,identity={'aliases':{}}),dict(run_id='test',frames=annotations)


class TrackingEvaluationTests(unittest.TestCase):
    def test_cross_camera_subject_uses_one_global_assignment(self):
        data,labels=fixtures()
        result=evaluate(data,labels)
        self.assertEqual((result['IDTP'],result['IDFP'],result['IDFN']),(2,0,0))
        self.assertEqual(result['IDF1'],1)

    def test_split_identity_is_penalized_even_when_boxes_are_perfect(self):
        data,labels=fixtures(ids=(1,2))
        result=evaluate(data,labels)
        self.assertEqual(result['IDF1'],.5)
        self.assertEqual((result['IDTP'],result['IDFP'],result['IDFN']),(1,1,1))

    def test_online_does_not_leak_final_aliases(self):
        data,labels=fixtures(ids=(1,2))
        data['identity']['aliases']={'2':1}
        self.assertEqual(evaluate(data,labels)['IDF1'],.5)
        self.assertEqual(evaluate(data,labels,identity_view='retrospective')['IDF1'],1)

    def test_missing_track_counts_as_false_negative(self):
        data,labels=fixtures()
        data['frames'][1]['tracks']=[]
        result=evaluate(data,labels)
        self.assertEqual((result['IDTP'],result['IDFN']),(1,1))

    def test_extra_visible_prediction_counts_as_false_positive(self):
        data,labels=fixtures()
        data['frames'][0]['instances'].append(dict(label='person',box_xyxy=[60,10,90,90]))
        data['frames'][0]['tracks'].append(dict(instance_index=1,global_id=8,local_id=8,status='observed'))
        result=evaluate(data,labels)
        self.assertEqual(result['IDFP'],1)

    def test_refuse_partial_unidentified_assisted_or_wrong_run(self):
        data,labels=fixtures()
        bad=deepcopy(labels)
        for a in bad['frames']:a['exhaustive_labels']=[]
        with self.assertRaisesRegex(ValueError,'No explicitly complete'):
            evaluate(data,bad)
        bad=deepcopy(labels);bad['frames'][0]['marks'][0]['subject']=''
        with self.assertRaisesRegex(ValueError,'subject reference'):evaluate(data,bad)
        bad=deepcopy(labels);bad['frames'][0]['marks'][0]['source']='assistant_seed'
        with self.assertRaisesRegex(ValueError,'Assisted'):evaluate(data,bad)
        bad=deepcopy(labels);bad['run_id']='another'
        with self.assertRaisesRegex(ValueError,'run_id'):evaluate(data,bad)

    def test_duplicate_ids_and_source_mismatch_are_rejected(self):
        data,labels=fixtures()
        data['frames'][0]['tracks']*=2
        with self.assertRaisesRegex(ValueError,'Two visible'):evaluate(data,labels)
        data,labels=fixtures();labels=deepcopy(labels)
        labels['frames'][0]['frame']['captured_at_s']=.3
        with self.assertRaisesRegex(ValueError,'source frame'):evaluate(data,labels)

    def test_camera_local_changes_are_diagnostics_not_hota(self):
        data,labels=fixtures(ids=(1,2),cameras=('A','A'))
        result=evaluate(data,labels)
        self.assertEqual(result['sampled_global_id_changes'],1)
        self.assertNotIn('HOTA',result)

    def test_all_compatible_pairs_enter_global_matching_before_frame_greediness(self):
        data,labels=fixtures(ids=(1,1),cameras=('A','A'))
        # A duplicate on one frame is a separate false-positive identity, not a
        # reason to destroy the longer correct identity's global assignment.
        data['frames'][0]['instances'].append(dict(label='person',box_xyxy=[10,10,50,90]))
        data['frames'][0]['tracks'].insert(0,dict(instance_index=1,global_id=2,local_id=2))
        result=evaluate(data,labels)
        self.assertEqual(result['IDTP'],2)
        self.assertEqual(result['IDFP'],1)

    def test_same_annotations_compare_cases_without_mixing_aliases(self):
        data,labels=fixtures()
        data['tracking_cases']={'legacy':'original','baseline':'motion'}
        data['identity_by_case']={'legacy':{'aliases':{'2':1}},'baseline':{'aliases':{}}}
        for i,f in enumerate(data['frames']):
            f['tracks_by_case']={'legacy':deepcopy(f['tracks']),'baseline':deepcopy(f['tracks'])}
            f['tracks_by_case']['baseline'][0]['global_id']=i+1
        self.assertEqual(evaluate(data,labels,case='legacy')['IDF1'],1)
        self.assertEqual(evaluate(data,labels,case='baseline')['IDF1'],.5)
        self.assertEqual(evaluate(data,labels,case='baseline',identity_view='retrospective')['IDF1'],.5)


if __name__=='__main__':unittest.main()
