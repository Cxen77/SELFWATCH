from contextlib import redirect_stdout
import io
import unittest
import numpy as np
from selfwatch_v2.contracts import FrameRef,Instance
from selfwatch_v2.continuity_guard import ContinuityGuard


class CleanMemoryTests(unittest.TestCase):
    def setUp(self):
        self.guard=ContinuityGuard();self.image=np.zeros((480,640,3),dtype=np.uint8)
        self.vector=np.eye(1,512,dtype=np.float32)[0]
        self.sink=redirect_stdout(io.StringIO());self.sink.__enter__()

    def tearDown(self):self.guard.close();self.sink.__exit__(None,None,None)

    def step(self,index,score=.95,features=True):
        f=FrameRef('A','e',index,index/30,0,640,480,'fixture')
        people=(Instance((50,30,150,350),score,0,'person'),)
        self.guard.preview(f,people)
        return self.guard.commit(f,people,self.image,{0:self.vector} if features else {})

    def test_cached_skip_preserves_id_and_does_not_relearn_old_feature(self):
        for index in range(6):tracks=self.step(index)
        token=tracks[0]['local_id'];track=self.guard.engine.tracker.tracks[0]
        before=track.embedding.copy();history=len(track._emb_history)
        result=self.step(6,features=False)
        self.assertEqual(result[0]['local_id'],token)
        np.testing.assert_array_equal(before,track.embedding)
        self.assertEqual(len(track._emb_history),history)
        self.assertFalse(track.memory_update_allowed)
        self.assertEqual(self.guard.continuity_audit['local_gallery_held'],1)

    def test_first_weak_observation_does_not_update_clean_gallery(self):
        for index in range(6):self.step(index)
        track=self.guard.engine.tracker.tracks[0];before=len(track._emb_history)
        self.step(6,score=.6)
        self.assertEqual(len(track._emb_history),before)
        self.assertFalse(track.memory_update_allowed)

    def test_birth_without_fresh_embedding_has_no_fake_gallery(self):
        self.step(0,features=False)
        track=self.guard.engine.tracker.tracks[0]
        self.assertIsNone(track.embedding);self.assertEqual(len(track._emb_history),0)
        self.assertTrue(self.guard.continuity_audit['untrusted_birth_gallery_cleared'])
