"""CPU-only integration regressions against the real original cognitive engine."""
from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
from selfwatch_v2.contracts import FrameRef, Instance
from selfwatch_v2.legacy_guard import LegacyIdentityGuard
from selfwatch_v2.identity import SharedIdentity


def person(x=40, score=.95):
    return Instance((x,40,x+100,300),score,0,"person")


class LegacyGuardTests(unittest.TestCase):
    def setUp(self):
        self.sink=StringIO()
        self.output=redirect_stdout(self.sink)
        self.output.__enter__()
        self.guard=LegacyIdentityGuard()
        self.identity=SharedIdentity()
        self.image=np.zeros((480,640,3),dtype=np.uint8)
        self.vector=np.eye(1,512,dtype=np.float32)[0]
        self.index=0

    def tearDown(self):
        self.guard.close()
        self.output.__exit__(None,None,None)

    def step(self,t,people,epoch="visit",features=None):
        f=FrameRef('A',epoch,self.index,t,1000,640,480,'fixture')
        self.index+=1
        self.guard.preview(f,people)
        vectors=features if features is not None else {i:self.vector for i in range(len(people))}
        tracks=self.guard.commit(f,people,self.image,vectors)
        self.identity.synchronize_memory(f,self.guard.memory)
        result=self.identity.update(f,tracks,vectors,people)
        return result,self.identity.memory_snapshot(f,self.guard.memory)

    def establish(self):
        tracks=None
        for t in (0.,.1,.2,.3,.4,.5):
            tracks,_=self.step(t,[person()])
        self.assertTrue(self.guard.engine.tracker.tracks[0].is_confirmed)
        return tracks[0]['global_id']

    def test_real_phantom_warm_path_reserves_person_id_without_rendering_ghost(self):
        gid=self.establish()
        states=set()
        for t in (.6,1.,1.5,2.,3.,4.,5.,6.,7.,8.):
            tracks,memory=self.step(t,[])
            self.assertEqual(tracks,[])
            states.update(r['state'] for r in memory)
            self.assertTrue(all(not r['observed'] for r in memory))
        self.assertIn('THINKING',states)
        self.assertIn('PHANTOM',states)
        self.assertIn('WARM',states)
        tracks,_=self.step(8.1,[person()])
        self.assertEqual(tracks[0]['global_id'],gid)
        self.assertTrue(self.identity.lifecycle_events)

    def test_overlap_freezes_original_and_shared_memory(self):
        self.establish()
        for t in (.6,.7,.8,.9,1.):
            self.step(t,[person(),person(220)],features={0:self.vector,1:np.roll(self.vector,1)})
        tracks,memory=self.step(1.1,[person(80),person(100)],
            features={0:self.vector,1:np.roll(self.vector,1)})
        self.assertTrue(self.guard.audit['frozen_local_ids'])
        self.assertTrue(any(t['protection_state']=='FROZEN' for t in tracks))
        self.assertTrue(all(not t['memory_update_allowed'] for t in tracks if t['protection_state']=='FROZEN'))
        self.assertTrue(all(not t['cross_view_update_allowed'] for t in tracks))
        self.assertTrue(self.guard.audit['ownership'].get('VISIBLE_FROZEN',0))
        self.assertEqual(len({t['local_id'] for t in tracks}),len(tracks))

    def test_new_lookalike_cannot_claim_reserved_owner(self):
        gid=self.establish()
        self.step(.6,[])
        self.assertEqual(self.identity.snapshot()['registry']['active_global_ids'],0)
        # Far-away birth is a different protected local token. Appearance alone
        # cannot bypass the original engine's held-owner/motion checks.
        tracks,_=self.step(.7,[person(480)])
        self.assertNotEqual(tracks[0]['global_id'],gid)

    def test_original_crossing_locks_preserve_two_separating_people(self):
        other=np.roll(self.vector,1)
        expected=None
        for step in range(23):
            x=40+step*15
            people=[person(x),person(400-step*15)]
            features={0:self.vector,1:other}
            if step%2:
                people.reverse()
                features={0:other,1:self.vector}
            tracks,_=self.step(step*.1,people,features=features)
            mapped={float(people[t['instance_index']].box_xyxy[0]):t['global_id'] for t in tracks}
            if step==0:
                expected=(mapped[40.],mapped[400.])
            if step==22:
                self.assertEqual(mapped[x],expected[0])
                self.assertEqual(mapped[400-step*15],expected[1])

    def test_capture_clock_epoch_and_original_image_geometry(self):
        self.establish()
        self.assertEqual(self.guard.tick,16)
        self.assertAlmostEqual(self.guard.engine.event_clock(),.5)
        self.assertEqual(self.image.shape,(480,640,3))
        old=self.guard.engine
        self.step(.6,[person()],epoch='reconnected')
        self.assertIsNot(old,self.guard.engine)
        self.assertFalse(self.identity.held)
        with self.assertRaisesRegex(ValueError,'strictly increase'):
            self.step(.5,[],'reconnected')

    def test_no_label_index_assumptions_and_no_hidden_gpu_fallback(self):
        head=Instance((20,20,100,100),.9,0,'head')
        tracks,_=self.step(0.,[head])
        self.assertEqual(tracks,[])
        p=Instance((40,40,140,300),.95,13,'person')
        tracks,_=self.step(.1,[p],features={0:self.vector})
        self.assertEqual(len(tracks),1)
        # ReID skips use cached appearance, not a second GPU call.
        tracks,_=self.step(.2,[p],features={})
        self.assertEqual(len(tracks),1)

    def test_gait_abstains_on_irregular_sparse_video(self):
        self.establish()
        for t in (1.1,1.8,2.2):
            self.step(t,[person()])
        self.assertFalse(self.guard.audit['gait']['sampling_valid'])
        self.assertEqual(self.guard.audit['gait']['signatures_available'],0)
        self.assertFalse(self.guard.audit['gait']['used_for_identity'])

    def test_full_loss_beyond_tracker_survival_recovers_from_warm_memory(self):
        gid=self.establish()
        for t in (.7,1.5,2.5,4.,6.,8.,10.,12.):
            self.step(t,[])
        self.assertEqual(self.guard.engine.tracker.tracks,[])
        self.assertTrue(self.guard.engine.warm_memory.count)
        tracks,_=self.step(12.1,[person()])
        self.assertEqual(tracks[0]['global_id'],gid)

    def test_experimental_weak_birth_requires_repetition_and_preserves_raw_score(self):
        self.guard=LegacyIdentityGuard(weak_births=True)
        weak=person(score=.35)
        first,_=self.step(0.,[weak],features={})
        self.assertEqual(first,[])
        self.assertEqual(self.identity.snapshot()['registry']['total_global_ids_created'],0)
        second,_=self.step(.1,[weak],features={})
        self.assertEqual(len(second),1)
        self.assertAlmostEqual(self.guard.engine.tracker.tracks[0].score,.35)
        self.assertEqual(second[0]['global_id'],1)

    def test_isolated_weak_birth_dies_without_public_id(self):
        self.guard=LegacyIdentityGuard(weak_births=True)
        self.step(0.,[person(score=.35)],features={})
        self.step(.1,[],features={})
        self.assertEqual(self.guard.engine.tracker.tracks,[])
        self.assertEqual(self.guard.last_seen,{})
        self.assertEqual(self.identity.snapshot()['registry']['total_global_ids_created'],0)
        again,_=self.step(.2,[person(score=.35)],features={})
        self.assertEqual(again,[])

    def test_missing_owner_attention_is_scoped_recent_and_never_a_detection(self):
        self.establish()
        self.step(.6,[])
        query=FrameRef('A','visit',8,.7,1000,640,480,'fixture')
        before=(self.guard.tick,self.guard.previous_time,len(self.guard.engine.tracker.tracks))
        regions=self.guard.recovery_regions(query)
        self.assertTrue(regions)
        self.assertEqual(before,(self.guard.tick,self.guard.previous_time,len(self.guard.engine.tracker.tracks)))
        self.assertEqual(self.guard.recovery_regions(FrameRef('B','visit',8,.7,1000,640,480,'fixture')),[])
        self.assertEqual(self.guard.recovery_regions(FrameRef('A','reconnect',8,.7,1000,640,480,'fixture')),[])
        self.assertEqual(self.guard.recovery_regions(FrameRef('A','visit',9,3.,1000,640,480,'fixture')),[])

    def test_visible_person_does_not_request_missing_owner_attention(self):
        self.establish()
        self.assertEqual(self.guard.recovery_regions(FrameRef('A','visit',8,.6,1000,640,480,'fixture')),[])


if __name__=='__main__':
    unittest.main()
