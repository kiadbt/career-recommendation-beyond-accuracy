"""Synthetic tests contain no study participants or source records."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from evaluate_fusion import metrics,analyse

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.q={'pid':np.arange(30),'history_blocks':np.zeros((30,1),int),'length':np.ones(30,int),'target_block':np.ones(30,int),'bank_members':np.array([[0,-1],[0,2]]),'bank_size':np.array([1,2])}
        self.groups=['A','A','B','C']
        self.p={'pid':self.q['pid'],'rec':np.tile([0,1],(30,1)),'neu':np.tile([2,0],(30,1)),'fus':np.tile([0,1],(30,1))}
    def test_complete_targets_and_breadth(self):
        m,u=metrics(self.q,self.p['neu'],self.groups)
        np.testing.assert_allclose(m,np.ones((30,3)));self.assertTrue(u.all())
        m,_=metrics(self.q,self.p['rec'],self.groups)
        self.assertAlmostEqual(m[0,0],1/(1+1/np.log2(3)))
        np.testing.assert_allclose(m[:,1],1);np.testing.assert_allclose(m[:,2],0)
    def test_history_groups_not_last_occupation_only(self):
        q=dict(self.q); q['bank_members']=np.array([[0,3],[2,-1]]);q['bank_size']=np.array([2,1])
        m,_=metrics(q,self.p['neu'],['A','A','C','C'])
        self.assertTrue(np.isnan(m[:,2]).all())
    def test_paired_contrasts(self):
        r=analyse(self.q,self.p,self.groups,['neu'],['fus'],'rec',100,1)
        self.assertEqual(r['contrasts']['fusion_minus_neural_broader_recall'],-1)
        self.assertEqual(r['contrasts']['change_in_model_gap'],0)
        for k,v in r['contrasts'].items():np.testing.assert_allclose(r['simultaneous_95_intervals'][k],[v,v],atol=1e-12)
    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):metrics(self.q,np.tile([0,0],(30,1)),self.groups)
    def test_misaligned_ids_rejected(self):
        self.p['pid']=self.p['pid'][::-1]
        with self.assertRaises(ValueError):analyse(self.q,self.p,self.groups,['neu'],['fus'],'rec',0)
    def test_empty_eligible_subset_rejected(self):
        with self.assertRaises(ValueError):analyse(self.q,self.p,['A']*4,['neu'],['fus'],'rec',0)

if __name__=='__main__':unittest.main()
