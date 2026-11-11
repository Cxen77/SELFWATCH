import unittest
from selfwatch_v2.mot_policy import select


def score(switches=20,idf1=.8,hota=.7,recall=.9):
    return dict(summary=dict(IDSW=switches,IDF1=idf1,HOTA=hota,CLR_Re=recall))


class SelectionTests(unittest.TestCase):
    def test_lower_switches_cannot_hide_lost_people_or_identity_quality(self):
        cases=dict(baseline=score(),clean_memory=score(10,recall=.7),strict_margin=score(11,idf1=.7),clean_margin=score(12,hota=.6))
        self.assertEqual(select(cases),'baseline')

    def test_ties_resolve_by_identity_quality_without_future_data(self):
        cases=dict(baseline=score(),clean_memory=score(15,idf1=.81),strict_margin=score(15,idf1=.82))
        self.assertEqual(select(cases),'strict_margin')

    def test_equal_switch_count_is_not_a_reduction(self):
        self.assertEqual(select(dict(baseline=score(),candidate=score(idf1=.9))),'baseline')
