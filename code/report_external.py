"""External fusion contrasts with source-specific original bootstrap schemes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json
from pathlib import Path
import numpy as np
from fusion_metrics import measures,contrasts

def main():
    p=argparse.ArgumentParser()
    for k in ['queries','predictions','catalogue','output']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--source',choices=['decorte','karrierewege'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    q=dict(np.load(a.queries));z=dict(np.load(a.predictions));assert np.array_equal(q['pid'],z['pid'])
    cat=json.loads(a.catalogue.read_text(encoding='utf8'));gm={i:r['code'][:2] for i,r in enumerate(cat)};truth=[];history=[]
    for i in range(len(q['pid'])):
        b=q['target_block'][i];truth.append(set(map(int,q['bank_members'][b,:q['bank_size'][b]])))
        history.append({int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]})
    unseen=np.array([bool(t-h) for t,h in zip(truth,history)])
    rec=measures(z['top_PersonalRecency'],truth,history,gm)
    neu=sum(measures(z[f'top_BiLSTM_{s}'],truth,history,gm) for s in [42,43])/2
    fus=sum(measures(z[f'top_Fusion_BiLSTM_{s}'],truth,history,gm) for s in [42,43])/2
    names=['NDCG_vs_recency','NDCG_vs_BiLSTM','repeat_Recall_vs_recency','new_group_Recall_vs_BiLSTM','neural_minus_recency_all','neural_minus_recency_unseen','cohort_shift']
    def seven(f,r,n,u):
        g=n[:,0]-r[:,0];return np.r_[contrasts(f,r,n),g.mean(),g[u].mean(),g[u].mean()-g.mean()]
    output={};samples=[('all_histories',np.arange(len(rec)),20261022 if a.source=='decorte' else 20261016)]
    if a.source=='karrierewege':samples.append(('unique_nonmatching_sequences',q['sensitivity_indices'],20261017))
    for label,ix,seed in samples:
        r,n,f,u=rec[ix],neu[ix],fus[ix],unseen[ix];N=len(ix);rng=np.random.default_rng(seed);deviations=[]
        if a.source=='decorte':
            point=seven(f,r,n,u)
            for _ in range(5000):
                j=rng.integers(N,size=N);deviations.append(abs(seven(f[j],r[j],n[j],u[j])-point).max())
        else:
            g=n[:,0]-r[:,0];mask=np.c_[np.ones(N),np.ones(N),np.isfinite(r[:,1]),np.isfinite(r[:,2]),np.ones(N),u].astype(float)
            num=np.nan_to_num(np.c_[f[:,0]-r[:,0],f[:,0]-n[:,0],f[:,1]-r[:,1],f[:,2]-n[:,2],g,g])*mask
            point=num.sum(0)/mask.sum(0);point=np.r_[point,point[5]-point[4]]
            for _ in range(50):
                w=rng.multinomial(N,np.full(N,1/N),size=100).astype(float);den=w@mask
                if not (den>0).all():raise ValueError('Empty bootstrap subgroup')
                b=w@num/den;b=np.c_[b,b[:,5]-b[:,4]];deviations.extend(abs(b-point).max(1))
        radius=float(np.quantile(deviations,.95));output[label]=dict(histories=N,bootstrap_seed=seed,bootstrap_draws=5000,simultaneous_radius=radius,contrasts={k:dict(estimate=float(v),simultaneous_95_interval=[float(v-radius),float(v+radius)]) for k,v in zip(names,point)})
    a.output.write_text(json.dumps(output,indent=2));print('Wrote',a.output)

if __name__=='__main__':main()
