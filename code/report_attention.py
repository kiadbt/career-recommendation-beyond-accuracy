"""Recompute the reported self-attention architecture sensitivity analyses."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json,math
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    for k in ['queries','predictions','output']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--source',choices=['jobhop','decorte','karrierewege'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    q=dict(np.load(a.queries));z=dict(np.load(a.predictions));assert np.array_equal(q['pid'],z['pid'])
    targets=[];unseen=[]
    for i in range(len(q['pid'])):
        b=q['target_block'][i];t=set(map(int,q['bank_members'][b,:q['bank_size'][b]]));targets.append(t)
        h={int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]};unseen.append(bool(t-h))
    unseen=np.array(unseen)
    def measure(top):
        if a.source=='karrierewege':
            hits=top==q['target_block'][:,None];return np.c_[(hits*(1/np.log2(np.arange(2,12)))).sum(1),hits.any(1).astype(float)]
        return np.array([[sum((c in t)/math.log2(j+2) for j,c in enumerate(row))/sum(1/math.log2(j+2) for j in range(min(10,len(t)))),sum(c in t for c in row)/len(t)] for row,t in zip(top,targets)])
    seeds=[measure(z[f'top_attention_{s}']) for s in [42,43]];att=sum(seeds)/2;rec=measure(z['top_PersonalRecency']);neu=sum(measure(z[f'top_BiLSTM_{s}']) for s in [42,43])/2
    subsets=[('all_histories',np.arange(len(targets)),20261018 if a.source=='karrierewege' else 20261015)]
    if a.source=='karrierewege':subsets.append(('unique_nonmatching_sequences',q['sensitivity_indices'],20261019))
    out={}
    for name,ix,seed in subsets:
        n=len(ix);u=unseen[ix];gap=att[ix,0]-rec[ix,0];point=np.array([gap.mean(),gap[u].mean(),gap[u].mean()-gap.mean()]);rng=np.random.default_rng(seed);dev=[]
        for _ in range(50):
            w=rng.multinomial(n,np.full(n,1/n),size=100).astype(float);den=w@u.astype(float);assert (den>0).all();ga=w@gap/n;gu=w@(gap*u)/den;b=np.c_[ga,gu,gu-ga];dev.extend(abs(b-point).max(1))
        radius=float(np.quantile(dev,.95))
        out[name]=dict(histories=n,unseen_positive=int(u.sum()),attention_minus_recency=dict(estimates=point.tolist(),simultaneous_95_intervals=np.c_[point-radius,point+radius].tolist()),models=[dict(model=label,NDCG_all=float(v[ix,0].mean()),NDCG_unseen_positive=float(v[ix[u],0].mean()),Recall_all=float(v[ix,1].mean())) for label,v in [('PersonalRecency',rec),('BiLSTM',neu),('QuarterSelfAttention',att)]])
    a.output.write_text(json.dumps(out,indent=2));print('Wrote',a.output)

if __name__=='__main__':main()
