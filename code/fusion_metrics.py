"""CPU-only frozen metrics; importing never reads data or scores a split."""
import numpy as np

def fuse(recency, neural):
    output=[]
    for a,b in zip(recency,neural):
        scores={}
        for ranking in [a,b]:
            assert len(ranking)==10 and len(set(ranking))==10
            for rank,code in enumerate(ranking,1):
                code=int(code);scores[code]=scores.get(code,0)+.5/(60+rank)
        output.append(sorted(scores,key=lambda c:(-scores[c],c))[:10])
    return np.asarray(output,dtype=np.int64)

def measures(top, targets, histories, group_by_code):
    """Full NDCG, repeated-target recall, new-group-target recall; NaN if undefined."""
    out=np.full((len(top),3),np.nan);discount=1/np.log2(np.arange(2,12))
    for i,(ranked,target,history) in enumerate(zip(top,targets,histories)):
        t=set(map(int,target));h=set(map(int,history));assert t and len(set(ranked))==10
        hits=np.asarray([int(c) in t for c in ranked])
        out[i,0]=hits@discount/discount[:min(10,len(t))].sum()
        repeats=t&h
        if repeats:out[i,1]=len(set(ranked)&repeats)/len(repeats)
        if all(c in group_by_code for c in t|h):
            seen={group_by_code[c] for c in h}
            broad={c for c in t if group_by_code[c] not in seen}
            if broad:out[i,2]=len(set(ranked)&broad)/len(broad)
    return out

def contrasts(fusion,recency,neural):
    return np.array([(fusion[:,0]-recency[:,0]).mean(),(fusion[:,0]-neural[:,0]).mean(),
      np.nanmean(fusion[:,1]-recency[:,1]),np.nanmean(fusion[:,2]-neural[:,2])])

def intervals(fusion,recency,neural):
    for a in [fusion,recency,neural]:
        assert a.shape==fusion.shape and np.isfinite(a[:,0]).all()
        for col in [1,2]:assert np.array_equal(np.isfinite(a[:,col]),np.isfinite(fusion[:,col]))
    assert np.isfinite(fusion[:,1]).any() and np.isfinite(fusion[:,2]).any()
    point=contrasts(fusion,recency,neural);rng=np.random.default_rng(20261021);draws=[]
    for _ in range(5000):
        ix=rng.integers(len(fusion),size=len(fusion))
        assert np.isfinite(fusion[ix,1]).any() and np.isfinite(fusion[ix,2]).any()
        draws.append(contrasts(fusion[ix],recency[ix],neural[ix]))
    radius=float(np.quantile(np.max(np.abs(np.array(draws)-point),axis=1),.95))
    return point,np.c_[point-radius,point+radius]
