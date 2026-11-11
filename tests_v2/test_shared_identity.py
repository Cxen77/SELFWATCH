import unittest
import numpy as np
from selfwatch_v2.contracts import FrameRef, Instance
from selfwatch_v2.identity import SharedIdentity


def frame(camera,t,epoch="r"):
    return FrameRef(camera,epoch,int(t*10),t,1000,640,480,"fixture")


def tracks(local=1):
    return [dict(instance_index=0,local_id=local,hits=3)]


PEOPLE=[Instance((1,2,30,80),.9,0,"person")]


class SharedIdentityTests(unittest.TestCase):
    def observe(self, manager,camera,t,vector,local=1,epoch="r"):
        vectors={} if vector is None else {0:np.array(vector,dtype=np.float32)}
        return manager.update(frame(camera,t,epoch),tracks(local),vectors,PEOPLE)[0]

    def test_active_overlapping_views_can_share_identity_after_confirmation(self):
        memory=SharedIdentity(allow_overlap=True)
        a=self.observe(memory,"A",0,[1,0])
        first=self.observe(memory,"B",0,[1,0])
        self.assertNotEqual(first['global_id'],a['global_id'])
        self.assertEqual(first['identity_status'],'candidate_link')
        updated_a=self.observe(memory,"A",.3,[1,0])
        b=self.observe(memory,"B",.3,[1,0])
        self.assertEqual(b['global_id'],updated_a['global_id'])
        self.assertEqual(b['identity_status'],'existing_link')
        self.assertEqual(len(memory.registry.get_local_tracks(b['global_id'])),2)
        self.assertEqual(memory.registry.get_stats()['active_global_ids'],1)
        self.assertEqual(memory.registry.get_stats()['active_view_assignments'],2)
        self.assertNotEqual(memory.events[0]['old_global_id'],memory.events[0]['global_id'])
        self.assertEqual(memory.registry.get_memory_entry(b['global_id']).last_seen_time,.3)

    def test_ambiguous_lookalikes_abstain(self):
        memory=SharedIdentity(allow_overlap=True)
        self.observe(memory,"A",0,[1,0],1)
        self.observe(memory,"A",0,[1,.01],2)
        b=self.observe(memory,"B",.3,[1,0])
        self.assertEqual(b['identity_status'],'ambiguous')
        self.assertEqual(len(memory.events),0)

    def test_same_camera_exclusivity(self):
        memory=SharedIdentity(allow_overlap=True,confirmations=1)
        a=self.observe(memory,"A",0,[1,0],1)
        b=self.observe(memory,"A",0,[1,0],2)
        self.assertNotEqual(a['global_id'],b['global_id'])

    def test_disjoint_handoff_recovers_dormant_identity(self):
        memory=SharedIdentity(allow_overlap=False)
        a=self.observe(memory,"A",0,[1,0])
        self.observe(memory,"B",3,[1,0])
        b=self.observe(memory,"B",3.3,[1,0])
        self.assertEqual(a['global_id'],b['global_id'])

    def test_no_embedding_does_not_force_match(self):
        memory=SharedIdentity(allow_overlap=True)
        a=self.observe(memory,"A",0,[1,0])
        b=self.observe(memory,"B",0,None)
        self.assertNotEqual(a['global_id'],b['global_id'])
        self.assertFalse(b['appearance_fresh'])

    def test_reconnect_scopes_local_id_and_preserves_other_camera(self):
        memory=SharedIdentity(allow_overlap=True,confirmations=1)
        a=self.observe(memory,"A",0,[1,0])
        self.observe(memory,"B",0,[1,0])
        new=self.observe(memory,"A",.3,[-1,0],epoch="new")
        self.assertNotEqual(a['global_id'],new['global_id'])
        self.assertEqual(memory.registry.get_global_id("B",("r",1)),a['global_id'])

    def test_prototype_budget_evicts_inactive_without_reusing_ids(self):
        memory=SharedIdentity(max_identities=1)
        a=self.observe(memory,"A",0,[1,0])
        b=self.observe(memory,"B",3,[-1,0])
        self.assertGreater(b['global_id'],a['global_id'])
        self.assertEqual(len(memory.identities),1)

    def test_reciprocal_views_need_distinct_times_and_preserve_threshold(self):
        memory=SharedIdentity(allow_overlap=True,similarity=.85)
        a=self.observe(memory,'A',0,[1,0])
        self.observe(memory,'B',0,[.8,.6])
        self.observe(memory,'A',.3,[.9,.43])
        b=self.observe(memory,'B',.6,[1,0])
        self.assertEqual(a['global_id'],b['global_id'])
        self.assertEqual(memory.events[0]['evidence_type'],'reciprocal_views')

    def test_same_camera_recovery_is_not_counted_as_cross_camera(self):
        memory=SharedIdentity(confirmations=1)
        self.observe(memory,'A',0,[1,0])
        self.observe(memory,'A',3,[1,0],local=2)
        self.assertEqual(memory.events[0]['association_kind'],'same_camera_recovery')
        self.assertEqual(memory.registry.get_stats()['total_cross_camera_matches'],0)

    def test_dormant_age_uses_capture_clock(self):
        memory=SharedIdentity()
        a=self.observe(memory,'A',0,[1,0])
        memory.update(frame('B',3),[],{},[])
        dormant=memory.registry.get_dormant_identities()[a['global_id']]
        self.assertEqual(dormant.age_seconds,0)
        memory.update(frame('B',4),[],{},[])
        self.assertEqual(dormant.age_seconds,1)
