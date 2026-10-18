"""Evaluate saved neural, recency and fusion rankings on complete target sets.

No training or downloads. Bootstrap intervals condition on the supplied models.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def metrics(q, top, groups):
    n=len(q['pid']); top=np.asarray(top); groups=np.asarray(groups)
    if top.ndim!=2 or top.shape[0]!=n or top.shape[1]<1:
        raise ValueError('Rankings must have shape (histories, k).')
    if not np.issubdtype(top.dtype,np.integer) or np.any(top<0) or np.any(top>=len(groups)):
        raise ValueError('Rankings contain invalid catalogue indices.')
    if np.any(np.diff(np.sort(top,axis=1),axis=1)==0):
        raise ValueError('Each ranking must contain distinct occupations.')
    values=np.full((n,3),np.nan); unseen=np.zeros(n,bool)
    discount=1/np.log2(np.arange(2,top.shape[1]+2))
    for i,row in enumerate(top):
        def members(b):
            codes=set(map(int,q['bank_members'][b,:q['bank_size'][b]]))
            if not codes or min(codes)<0 or max(codes)>=len(groups):
                raise ValueError('Target/history block is empty or outside the catalogue.')
            return codes
        target=members(q['target_block'][i])
        if q['length'][i]<1: raise ValueError('History must contain at least one block.')
        history=set().union(*(members(b) for b in q['history_blocks'][i,:q['length'][i]]))
        repeated=target & history
        historical_groups={groups[c] for c in history}
        broader={c for c in target if groups[c] not in historical_groups}
        hits=set(map(int,row))
        values[i,0]=sum(discount[j] for j,c in enumerate(row) if int(c) in target)/discount[:min(len(target),len(row))].sum()
        if repeated: values[i,1]=len(hits & repeated)/len(repeated)
        if broader: values[i,2]=len(hits & broader)/len(broader)
        unseen[i]=bool(target-history)
    return values,unseen

def analyse(q,p,groups,neural_keys,fusion_keys,recency_key='top_PersonalRecency',draws=5000,seed=20261021):
    if not np.array_equal(q['pid'],p['pid']) or len(np.unique(q['pid']))!=len(q['pid']):
        raise ValueError('Prediction IDs must uniquely match query IDs in order.')
    if not neural_keys or len(neural_keys)!=len(fusion_keys):
        raise ValueError('Provide paired neural and fusion keys for each seed.')
    if any(p[k].shape!=p[recency_key].shape for k in neural_keys+fusion_keys):
        raise ValueError('All component and fusion rankings must use the same output length.')
    rec,u=metrics(q,p[recency_key],groups)
    neural=np.mean([metrics(q,p[k],groups)[0] for k in neural_keys],axis=0)
    fusion=np.mean([metrics(q,p[k],groups)[0] for k in fusion_keys],axis=0)
    rep=np.isfinite(rec[:,1]); broad=np.isfinite(rec[:,2])
    if not u.any() or not rep.any() or not broad.any():
        raise ValueError('The seven-contrast family requires nonempty unseen, repeated and broader subsets.')
    if draws<0 or 0<draws<100: raise ValueError('Use zero or at least 100 bootstrap draws.')
    n=len(u); gap=neural[:,0]-rec[:,0]
    numer=np.column_stack([gap,gap*u,fusion[:,0]-rec[:,0],fusion[:,0]-neural[:,0],np.nan_to_num(fusion[:,1]-rec[:,1]),np.nan_to_num(fusion[:,2]-neural[:,2])])
    den=np.column_stack([np.ones(n),u,np.ones(n),np.ones(n),rep,broad])
    def contrast(w):
        counts=w@den
        if np.any(counts==0): raise ValueError('Bootstrap draw has an empty eligible subset.')
        v=(w@numer)/counts
        return np.concatenate([v[:,:2],(v[:,1]-v[:,0])[:,None],v[:,2:]],axis=1)
    point=contrast(np.ones((1,n)))[0]
    labels=['neural_minus_recency_all','neural_minus_recency_unseen','change_in_model_gap','fusion_minus_recency_NDCG','fusion_minus_neural_NDCG','fusion_minus_recency_repeated_recall','fusion_minus_neural_broader_recall']
    result={'histories':n,'unseen_positive':int(u.sum()),'repeated_eligible':int(rep.sum()),'broader_eligible':int(broad.sum()),'contrasts':dict(zip(labels,point.tolist())),'draws':draws,'seed':seed,'training_seeds_averaged':len(neural_keys),'scope':'Paired-history uncertainty conditional on supplied models; no training-seed resampling or adjustment for adaptive research history.'}
    if draws:
        rng=np.random.default_rng(seed); deviations=[]
        for start in range(0,draws,100):
            w=rng.multinomial(n,np.full(n,1/n),size=min(100,draws-start))
            deviations.extend(np.max(np.abs(contrast(w)-point),axis=1))
        radius=float(np.quantile(deviations,.95))
        result['simultaneous_95_intervals']={k:[float(v-radius),float(v+radius)] for k,v in zip(labels,point)}
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['queries','predictions','groups','out']: ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--neural-keys',nargs='+',required=True);ap.add_argument('--fusion-keys',nargs='+',required=True)
    ap.add_argument('--recency-key',default='top_PersonalRecency');ap.add_argument('--draws',type=int,default=5000);ap.add_argument('--seed',type=int,default=20261021)
    args=ap.parse_args()
    q=dict(np.load(args.queries,allow_pickle=False));p=dict(np.load(args.predictions,allow_pickle=False))
    groups=json.loads(args.groups.read_text(encoding='utf8'))
    result=analyse(q,p,groups,args.neural_keys,args.fusion_keys,args.recency_key,args.draws,args.seed)
    result['input_sha256']={name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in ['queries','predictions','groups']}
    args.out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))
