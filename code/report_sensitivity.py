import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import numpy as np,json,time,csv,hashlib
import argparse
ap=argparse.ArgumentParser()
for name in ['prepared-dir','predictions-dir','output-dir']:ap.add_argument('--'+name,type=Path,required=True)
ap.add_argument('--taxonomy',type=Path)
args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
if (args.output_dir/'results.json').exists():raise FileExistsError('Output already exists')
HERE=args.output_dir;P=args.prepared_dir;ROOT=Path(__file__).resolve().parents[1]
start=time.monotonic();manifest={'source':'portable saved-ranking rebuild'}
cat=json.loads((P/'catalogue_texts.json').read_text(encoding='utf-8'));groups=np.array([r['code'][:2] for r in cat])
assert [r['code'] for r in cat]==sorted(r['code'] for r in cat)
configs=[(k,L,w) for k in [5,10,20] for L in [10,20,50] if L>=k for w in [.25,.5,.75]]
assert len(configs)==24
names=['NDCG_vs_recency','NDCG_vs_BiLSTM','repeat_Recall_vs_recency','new_group_Recall_vs_BiLSTM']
results={};csvrows=[]
for split in ['fusion_validation','decorte']:
 q=dict(np.load(P/('fusion_queries.npz' if split=='fusion_validation' else 'decorte_queries.npz')));z=dict(np.load(args.predictions_dir/('fusion.npz' if split=='fusion_validation' else 'decorte.npz')));z={('top_'+k[8:] if k.startswith('depth50_') else k):v for k,v in z.items() if k=='pid' or k.startswith('depth50_')};N=len(q['pid']);assert np.array_equal(q['pid'],z['pid'])
 truth=np.zeros((N,len(cat)),bool);repeat=np.zeros_like(truth);broad=np.zeros_like(truth)
 for i in range(N):
  h={int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]}
  b=q['target_block'][i];t=set(map(int,q['bank_members'][b,:q['bank_size'][b]]));hg=set(groups[list(h)])
  truth[i,list(t)]=True;repeat[i,list(t&h)]=True;broad[i,[c for c in t if groups[c] not in hg]]=True
 den=[x.sum(1) for x in [truth,repeat,broad]];ix=np.arange(N)[:,None]
 def metrics(top,k):
  top=top[:,:k];hit=truth[ix,top];discount=1/np.log2(np.arange(2,k+2));ideal=np.cumsum(discount)[np.minimum(den[0],k)-1]
  out=np.empty((N,4));out[:,0]=(hit*discount).sum(1)/ideal;out[:,1]=hit.sum(1)/den[0]
  for col,mask,n in [(2,repeat,den[1]),(3,broad,den[2])]:
   out[:,col]=np.divide(mask[ix,top].sum(1),n,out=np.full(N,np.nan),where=n>0)
  return out
 rec={k:metrics(z['top_PersonalRecency'],k) for k in [5,10,20]}
 neu={k:sum(metrics(z[f'top_BiLSTM_{s}'],k) for s in [42,43])/2 for k in [5,10,20]}
 fused={}
 for L in [10,20,50]:
  for seed in [42,43]:
   codes=np.concatenate([z['top_PersonalRecency'][:,:L],z[f'top_BiLSTM_{seed}'][:,:L]],axis=1)
   order=np.argsort(codes,axis=1,kind='stable');scodes=np.take_along_axis(codes,order,axis=1)
   eq=scodes[:,1:]==scodes[:,:-1];base=1/(60+np.arange(1,L+1))
   for w in [.25,.5,.75]:
    values=np.broadcast_to(np.r_[base*(1-w),base*w],codes.shape)
    scores=np.take_along_axis(values,order,axis=1).copy()
    scores[:,1:]+=np.where(eq,scores[:,:-1],0)
    scores[:,:-1][eq]=-np.inf
    ranked=np.take_along_axis(scodes,np.argsort(-scores,axis=1,kind='stable')[:,:L],axis=1)
    assert np.all(np.diff(np.sort(ranked,axis=1),axis=1)!=0)
    if L==10 and w==.5:
     old=np.load(args.predictions_dir/('fusion.npz' if split=='fusion_validation' else 'decorte.npz'));assert np.array_equal(ranked,old[f'top_Fusion_BiLSTM_{seed}'])
    for k in [5,10,20]:
     if k<=L:fused.setdefault((k,L,w),[]).append(metrics(ranked,k))
 contrasts=[];rows=[]
 for k,L,w in configs:
  f=sum(fused[k,L,w])/2;r=rec[k];n=neu[k]
  diff=np.stack([f[:,0]-r[:,0],f[:,0]-n[:,0],f[:,2]-r[:,2],f[:,3]-n[:,3]],axis=1)
  contrasts.append(diff)
  row={'k':k,'input_depth':L,'neural_weight':w,'fusion_metrics':np.nanmean(f,axis=0).tolist(),'estimates':np.nanmean(diff,axis=0).tolist()}
  rows.append(row)
 diff=np.concatenate(contrasts,axis=1);point=np.nanmean(diff,axis=0);valid=np.isfinite(diff[:,0:4]).astype(float);numer=np.nan_to_num(diff)
 rng=np.random.default_rng(20261014);maxdev=[];bootstart=time.monotonic()
 for a in range(0,5000,100):
  weights=rng.multinomial(N,np.full(N,1/N),size=100).astype(float)
  denominator=np.tile(weights@valid,(1,24));assert (denominator>0).all()
  draws=(weights@numer)/denominator
  maxdev.extend(np.max(np.abs(draws-point),axis=1))
  if (a+100)%1000==0:print(json.dumps({'split':split,'bootstrap_draws':a+100,'seconds':time.monotonic()-bootstart}),flush=True)
 radius=float(np.quantile(maxdev,.95))
 for row in rows:
  est=np.array(row['estimates']);bounds=np.c_[est-radius,est+radius];row['simultaneous_95_intervals']=bounds.tolist()
  row['aggregate_gain_both_point']=bool((est[:2]>0).all())
  row['broader_loss_point']=bool(est[3]<0)
  row['all_reference_thresholds_pass']=bool((bounds[:2,0]>.005).all() and (bounds[2:,0]>-.01).all())
  csvrow={'dataset':split,**{k:row[k] for k in ['k','input_depth','neural_weight']}}
  for j,name in enumerate(names):csvrow[name]=est[j];csvrow[name+'_lower']=bounds[j,0];csvrow[name+'_upper']=bounds[j,1]
  csvrows.append(csvrow)
 baseline=next(r for r in rows if (r['k'],r['input_depth'],r['neural_weight'])==(10,10,.5))
 original=json.loads((ROOT/'aggregate_results'/('fusion.json' if split=='fusion_validation' else 'external.json')).read_text())
 assert np.allclose(baseline['estimates'],[original['contrasts'][name]['estimate'] for name in names],atol=1e-12,rtol=0)
 results[split]={'resumes':N,'repeat_support':int((den[1]>0).sum()),'broader_support':int((den[2]>0).sum()),'simultaneous_radius_96_contrasts':radius,'rows':rows,'components':{str(k):{'recency':np.nanmean(rec[k],axis=0).tolist(),'neural':np.nanmean(neu[k],axis=0).tolist()} for k in rec},'primary_points_reproduced':True}
 print(json.dumps({'split':split,'complete':True,'radius':radius}),flush=True)
report={'status':'post hoc exploratory robustness','configurations_per_dataset':24,'bootstrap_draws':5000,'metric_order':['NDCG','full_Recall','repeat_Recall','new_group_Recall'],'contrast_order':names,'inference':manifest,'CPU_seconds':time.monotonic()-start,'datasets':results}
(HERE/'results.json').write_text(json.dumps(report,indent=2))
with (HERE/'all_contrasts.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
print(json.dumps({'complete':True,'CPU_seconds':report['CPU_seconds'],'path':str(HERE/'results.json')}))
