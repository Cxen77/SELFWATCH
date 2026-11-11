from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
from selfwatch_v2.appearance import AppearanceEncoder
from selfwatch_v2.contracts import FrameRef,Instance,Observation
from selfwatch_v2.identity_study import run_sample


class FakeExtractor:
    def __init__(self):self.calls=[]
    def extract_batch(self,crops):
        self.calls.append(len(crops))
        return np.tile(np.eye(1,512,dtype=np.float32),(len(crops),1))


def encoder():
    value=AppearanceEncoder.__new__(AppearanceEncoder)
    value.extractor=FakeExtractor()
    value.states={};value.attention={};value.refresh_s=.5;value.batch_buckets=False;value.provenance={'model':'synthetic'}
    return value


class FixedFeatureTests(unittest.TestCase):
    def test_all_crop_policy_is_track_independent_and_keeps_quality_gates(self):
        e=encoder();image=np.zeros((480,640,3),dtype=np.uint8)
        f=FrameRef('A','s',0,0.,1.,640,480,'fixture')
        p=Instance((30,20,130,300),.9,0,'person')
        obs=Observation(f,(p,),'fixture','segment',0)
        self.assertEqual(len(e.extract_all(obs,image)[0]),1)
        self.assertEqual(len(e.extract_all(obs,image)[0]),1)
        self.assertEqual(e.states,{})
        self.assertEqual(e.attention,{})
        self.assertEqual(e.extract_all(replace(obs,instances=(p,p)),image)[0],{})
        self.assertEqual(e.extract_all(replace(obs,instances=(replace(p,score=.3),)),image)[0],{})

    def test_default_schedule_remains_enabled(self):
        e=encoder();image=np.zeros((480,640,3),dtype=np.uint8)
        f=FrameRef('A','s',0,0.,1.,640,480,'fixture')
        obs=Observation(f,(Instance((30,20,130,300),.9,0,'person'),),'fixture','segment',0)
        tracks=[dict(instance_index=0,local_id=1,hits=1)]
        self.assertTrue(e.extract(obs,image,tracks)[0])
        again=replace(obs,frame=replace(f,frame_index=1,captured_at_s=.05))
        self.assertFalse(e.extract(again,image,tracks)[0])
        self.assertEqual(len(e.extractor.calls),1)


class CacheReplayTests(unittest.TestCase):
    def test_cpu_only_replay_reproduces_online_assignments_and_rejects_corruption(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);sources=[]
            for camera in ('A','B'):
                path=root/f'{camera}.avi'
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,320))
                if not writer.isOpened():self.skipTest('MJPG writer unavailable')
                for _ in range(4):writer.write(np.zeros((320,320,3),dtype=np.uint8))
                writer.release()
                sources.append(dict(camera_id=camera,stream_epoch='fixture',path=str(path),fps=10.))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,sources=sources)))
            sample=dict(sample='test',replay_manifest=str(manifest))
            class Detector:
                provenance={'model':'synthetic','class_names':{0:'person',2:'ear',10:'hand'}}
                def predict(self,frame,image):
                    return Observation(frame,(Instance((40,20,140,280),.95,0,'person'),),'fixture','segment',0),None
            fresh=root/'fresh';fresh.mkdir()
            reused=root/'reused';reused.mkdir()
            a=run_sample(sample,fresh,Detector(),encoder())
            b=run_sample(sample,reused,None,None,cache_root=fresh)
            self.assertEqual(a['input_cache_sha256'],b['input_cache_sha256'])
            left=json.loads((fresh/'test/data.json').read_text())
            right=json.loads((reused/'test/data.json').read_text())
            def signature(data):
                return [(f['observation_id'],{k:[(t['local_id'],t['global_id']) for t in v]
                    for k,v in f['tracks_by_case'].items()}) for f in data['frames']]
            self.assertEqual(signature(left),signature(right))
            with (fresh/'test/features.npz').open('ab') as handle:handle.write(b'changed')
            corrupt=root/'corrupt';corrupt.mkdir()
            with self.assertRaisesRegex(ValueError,'Corrupt perception cache'):
                run_sample(sample,corrupt,None,None,cache_root=fresh)


if __name__=='__main__':unittest.main()
