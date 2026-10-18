"""Portable fixed-target cohort metrics. Inputs remain local; output is aggregate.

No model selection, training, network access, or occupation-candidate filtering.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def measurements(q,top):
    n=len(q['pid']); assert top.shape==(n,10)
    assert (top>=0).all() and (top<3007).all()
    assert np.all(np.diff(np.sort(top,axis=1),axis=1)!=0)
    out=np.zeros((n,2)); unseen=np.zeros(n,bool)
    discount=1/np.log2(np.arange(2,12))
    for i,row in enumerate(top):
        b=q['target_block'][i];t=set(map(int,q['bank_members'][b,:q['bank_size'][b]]));assert t
        history={int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]}
        hits=np.array([int(c) in t for c in row]);out[i]=[(hits*discount).sum()/discount[:min(10,len(t))].sum(),hits.sum()/len(t)]
        unseen[i]=bool(t-history)
    return out,unseen

def analyse(q,predictions,recency,draws=5000,seed=20261018,indices=None):
    assert np.array_equal(q['pid'],predictions['pid']) and np.array_equal(q['pid'],recency['pid'])
    assert len(np.unique(q['pid']))==len(q['pid'])
    values=[]
    for key in ['top_attention_42','top_attention_43']:
        v,u=measurements(q,predictions[key]);values.append(v)
    m=sum(values)/len(values);r,u2=measurements(q,recency['top_PersonalRecency']);assert np.array_equal(u,u2)
    ix=np.arange(len(u)) if indices is None else indices;assert len(np.unique(ix))==len(ix)
    n=len(ix);uu=u[ix];assert uu.any();gap=m[ix,0]-r[ix,0]
    point=np.array([gap.mean(),gap[uu].mean(),gap[uu].mean()-gap.mean()])
    result={'resumes':n,'unseen_positive':int(uu.sum()),'attention_NDCG':float(m[ix,0].mean()),'attention_unseen_NDCG':float(m[ix[uu],0].mean()),'attention_Recall':float(m[ix,1].mean()),'estimates':point.tolist(),'seed_means':[v[ix].mean(0).tolist() for v in values],'draws':draws,'seed':seed}
    if draws:
        assert draws>=100; rng=np.random.default_rng(seed);dev=[]
        for start in range(0,draws,100):
            w=rng.multinomial(n,np.full(n,1/n),size=min(100,draws-start)).astype(float)
            den=w@uu.astype(float);assert (den>0).all(),'Bootstrap cohort empty'
            ga=w@gap/n;gu=w@(gap*uu)/den;sample=np.c_[ga,gu,gu-ga];dev.extend(abs(sample-point).max(1))
        radius=float(np.quantile(dev,.95));result['simultaneous_95_intervals']=np.c_[point-radius,point+radius].tolist()
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--queries',required=True,type=Path);ap.add_argument('--predictions',required=True,type=Path);ap.add_argument('--recency-predictions',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--draws',type=int,default=5000);ap.add_argument('--seed',type=int,default=20261018);ap.add_argument('--unique-nonmatching',action='store_true');args=ap.parse_args()
    assert args.draws>=0
    q=dict(np.load(args.queries,allow_pickle=False));z=dict(np.load(args.predictions,allow_pickle=False));r=dict(np.load(args.recency_predictions,allow_pickle=False))
    ix=q['sensitivity_indices'] if args.unique_nonmatching else None
    result=analyse(q,z,r,args.draws,args.seed,ix)
    result['input_sha256']={label:sha(path) for label,path in [('queries',args.queries),('attention_predictions',args.predictions),('recency_predictions',args.recency_predictions)]}
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
