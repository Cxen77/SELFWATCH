from threading import Event
from types import SimpleNamespace
import unittest
from selfwatch_v2.contracts import FrameRef, Observation
from selfwatch_v2.pipeline import process


class PipelineTests(unittest.TestCase):
    def test_cpu_work_overlaps_next_detection_and_results_remain_ordered(self):
        second_detection=Event()
        consumed=[]
        def frames():
            for i in range(4):
                consumed.append(i)
                yield FrameRef('A','r',i,i,1000,64,64,'test'),None
        class Detector:
            def predict(self,frame,image):
                if frame.frame_index==1:
                    second_detection.set()
                return Observation(frame,(),'fixture','detect',0),None
        class Pose:
            def refine(self,observation,image):
                if observation.frame.frame_index==0 and not second_detection.wait(timeout=3):
                    raise AssertionError('GPU preparation did not overlap blocked CPU pose')
                return observation
        tracker=SimpleNamespace(update=lambda *a:[])
        encoder=SimpleNamespace(extract=lambda *a:({},[],0))
        identity=SimpleNamespace(update=lambda frame,tracks,*a:tracks)
        iterator=process(frames(),Detector(),Pose(),{'A':tracker},encoder,identity)
        first=next(iterator)
        self.assertEqual(consumed,[0,1])
        all_results=[first,*iterator]
        self.assertEqual([r[0].frame.frame_index for r in all_results],[0,1,2,3])

    def test_worker_failure_is_propagated(self):
        frame=FrameRef('A','r',0,0,1000,64,64,'test')
        detector=SimpleNamespace(predict=lambda *a:(Observation(frame,(),'test','detect',0),None))
        def fail(*a):
            raise ValueError('pose failed')
        with self.assertRaisesRegex(ValueError,'pose failed'):
            list(process([(frame,None)],detector,SimpleNamespace(refine=fail),
                {'A':SimpleNamespace(update=lambda *a:[])},
                SimpleNamespace(extract=lambda *a:({},[],0)),SimpleNamespace(update=lambda *a:[])))

    def test_detector_attention_is_camera_scoped_before_inference(self):
        frames=[FrameRef(camera,'r',0,0,1000,64,64,'test') for camera in ('A','B')]
        seen=[]
        class Detector:
            def set_recovery_regions(self,frame,regions):seen.append((frame.camera_id,regions))
            def predict(self,frame,image):
                self_scope=seen[-1]
                if self_scope[0]!=frame.camera_id:raise AssertionError('Cross-camera attention')
                return Observation(frame,(),'fixture','detect',0),{'passes':[{'rotation_deg':0}]}
        trackers={camera:SimpleNamespace(update=lambda *a:[],recovery_regions=lambda f:[(0,0,10,10)] if f.camera_id=='A' else []) for camera in ('A','B')}
        result=list(process([(f,None) for f in frames],Detector(),SimpleNamespace(refine=lambda o,i:o),
            trackers,SimpleNamespace(extract=lambda *a:({},[],0)),SimpleNamespace(update=lambda *a:[])))
        self.assertEqual(seen,[('A',[(0,0,10,10)]),('B',[])])
        self.assertTrue(all(r[-1]['perception_diagnostics']['passes'] for r in result))
