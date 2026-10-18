"""Small fixtures test person partitioning and calendar boundaries, without training."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
import pandas as pd
from prepare_training import pack,records,partitions

class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame({'resume_id':[1,1,1,1,1,2,2], 'q':[8058,8059,8060,8063,8064,8059,8064], 'matched_code':['a','b','a','c','b','b','c']})
        self.ci={'a':0,'b':1,'c':2};self.emb=np.eye(3,384,dtype=np.float32)
    def test_chronology_uses_first_eligible_target(self):
        fit=records(self.d,[1,2],self.ci,'fit');tune=records(self.d,[1,2],self.ci,'tune');test=records(self.d,[1,2],self.ci,'test')
        self.assertEqual([r[2][0] for r in fit],[8059]);self.assertEqual([r[2][0] for r in tune],[8060]);self.assertEqual([r[2][0] for r in test],[8064,8064])
        for r in fit+tune+test:self.assertLess(max(x[0] for x in r[1]),r[2][0])
    def test_same_quarter_targets_preserved_and_person_weights_sum_one(self):
        d=pd.concat([self.d,pd.DataFrame([dict(resume_id=1,q=8059,matched_code='c')])],ignore_index=True)
        rs=records(d,[1,2],self.ci);q=pack(rs,self.emb)
        self.assertEqual(rs[0][2][1],(1,2))
        for pid in [1,2]:self.assertAlmostEqual(float(q['prefix_weight'][q['pid']==pid].sum()),1.)
    def test_partitions_invariant_to_row_order(self):
        self.assertEqual(partitions(self.d),partitions(self.d.iloc[::-1]))
        p=partitions(self.d);self.assertFalse(set(p['fit'])&set(p['tune']))

if __name__=='__main__':unittest.main()
