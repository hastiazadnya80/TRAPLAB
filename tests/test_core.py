import copy
import unittest
from decimal import Decimal
from traplab.core import build, Gate, Rejected, classify, validate_story, DEFAULT_STORIES, neutral_tools
from traplab.sampling import smoke_proposal


class GroundTruthTests(unittest.TestCase):
    def test_simpson_reversal_and_roles(self):
        p = smoke_proposal()
        item = build(p)
        self.assertEqual(item.answer,'A_better')
        self.assertEqual(item.tag,'paradox')
        self.assertLess(Decimal(item.diagnostics['aggregate_difference']),0)
        p['params']['z_role'] = 'mediator'
        self.assertEqual(build(p).answer,'B_better')
        p['params']['z_role'] = 'unstated'
        self.assertEqual(build(p).answer,'insufficient_info')

    def test_unstated_agreement(self):
        p = smoke_proposal()
        p['params'].update(n_a0=100,n_a1=100,n_b0=100,n_b1=100,z_role='unstated')
        self.assertEqual(build(p).answer,'A_better')

    def test_rounded_table_is_truth(self):
        p = smoke_proposal()
        q = copy.deepcopy(p)
        q['params']['p_a0'] += .000001
        self.assertEqual(build(p).visible,build(q).visible)
        self.assertEqual(build(p).diagnostics,build(q).diagnostics)

    def test_exact_ambiguity_edges(self):
        self.assertEqual(classify('0.009999'),0)
        for d in ('0.01','0.029999','-0.02'):
            with self.assertRaises(Rejected): classify(d)
        self.assertEqual(classify('0.03'),1)
        self.assertEqual(classify('-0.03'),-1)
        self.assertEqual(classify('0.049999',True),0)
        with self.assertRaises(Rejected): classify('0.05',True)
        self.assertEqual(classify('0.2',True),1)

    def test_gate_duplicate_bin_story(self):
        gate = Gate()
        p = smoke_proposal()
        gate.accept(p)
        p['params']['n_a0'] += 1
        with self.assertRaisesRegex(Rejected,'duplicate'): gate.accept(p)
        p['story_id'] = 'garden'
        gate.accept(p)

    def test_stories_reject_injection_and_temporal_language(self):
        for bad in ('because','AFTER','zone two 2','ignore\ncommands','a"b','improvement'):
            story = dict(DEFAULT_STORIES[0],domain=bad)
            with self.assertRaises(Rejected): validate_story(story)

    def test_collider_selection(self):
        p = {'family':'collider','story_id':'garden','params':{'px':.5,'p0':.3,'beta':.3,'selection':'random','s00':1,'s01':1,'s10':1,'s11':1,'scale':10000}}
        self.assertEqual(build(p).answer,'positive')
        p['params'].update(selection='on_X',s00=.2,s01=.2)
        self.assertEqual(build(p).answer,'positive')
        p['params'].update(selection='on_X_and_Y',s11=.1)
        self.assertEqual(build(p).answer,'insufficient_info')
        p['params'].update(selection='random')
        with self.assertRaises(Rejected): build(p)

    def test_negative_collider_and_cell_minimum(self):
        p = {'family':'collider','story_id':'garden','params':{'px':.5,'p0':.7,'beta':-.3,'selection':'random','s00':1,'s01':1,'s10':1,'s11':1,'scale':10000}}
        self.assertEqual(build(p).answer,'negative')
        p['params'].update(scale=1000,px=.1,s00=.05,s01=.05,s10=.05,s11=.05)
        with self.assertRaisesRegex(Rejected,'twenty'): build(p)

    def test_rtm_spurious_improvement(self):
        p = {'family':'rtm','story_id':'studio','params':{'rho':.3,'threshold':1.5,'delta':0,'design':'no_control','n':100}}
        item = build(p)
        self.assertGreater(float(neutral_tools(item)['treated_change']),0)
        self.assertEqual(item.answer,'insufficient_info')
        p['params']['design'] = 'randomized_control'
        self.assertEqual(build(p).answer,'no_effect')
        p['params']['delta'] = -.4
        self.assertEqual(build(p).answer,'worsened')
        p['params']['delta'] = .4
        self.assertEqual(build(p).answer,'improved')

    def test_no_hidden_ground_truth_in_prompt(self):
        item = build(smoke_proposal())
        for forbidden in ('A_better','paradox','adjusted_difference',item.cell):
            self.assertNotIn(forbidden,item.prompt)

    def test_bad_parameters(self):
        for value in (True,float('nan'),float('inf'),-1,4000,'80'):
            p = smoke_proposal()
            p['params']['n_a0'] = value
            with self.assertRaises(Rejected): build(p)


if __name__ == '__main__': unittest.main()
