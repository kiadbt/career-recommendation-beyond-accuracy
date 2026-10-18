import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import argparse
from fusion_metrics import measures,fuse
ap=argparse.ArgumentParser()
for name in ['prepared-dir','predictions-dir','trained-dir','output']:ap.add_argument('--'+name,type=Path,required=True)
args=ap.parse_args()
if args.output.exists():raise FileExistsError(args.output)
start=time.monotonic()
cat=json.loads((args.prepared_dir/'catalogue_texts.json').read_text(encoding='utf-8'));group={i:x['code'][:2] for i,x in enumerate(cat)}
seeds=[42,43,44,45,46];names=['neural_minus_recency_all','neural_minus_recency_unseen','cohort_shift','NDCG_vs_recency','NDCG_vs_BiLSTM','repeat_Recall_vs_recency','new_group_Recall_vs_BiLSTM']
analyses={}
for si,split in enumerate(['confirmation','fusion']):
 q=dict(np.load(args.prepared_dir/('chronological_'+split+'.npz')));p=dict(np.load(args.predictions_dir/('chronological_'+split+'.npz')));assert np.array_equal(q['pid'],p['pid']);n=len(q['pid'])
 targets=[];histories=[]
 for i in range(n):
  b=q['target_block'][i];targets.append(set(map(int,q['bank_members'][b,:q['bank_size'][b]])))
  histories.append({int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]})
 unseen=np.array([bool(t-h) for t,h in zip(targets,histories)]);repeat=np.array([bool(t&h) for t,h in zip(targets,histories)])
 rec=measures(p['top_PersonalRecency'],targets,histories,group)
 neu=np.array([measures(p[f'top_BiLSTM_{s}'],targets,histories,group) for s in seeds]);fus=np.array([measures(p[f'top_Fusion_BiLSTM_{s}'],targets,histories,group) for s in seeds])
 for s in seeds:assert np.array_equal(fuse(p['top_PersonalRecency'],p[f'top_BiLSTM_{s}']),p[f'top_Fusion_BiLSTM_{s}'])
 broad=np.isfinite(rec[:,2]);assert np.array_equal(np.isfinite(rec[:,1]),repeat)
 masks=np.c_[np.ones(n),unseen,np.ones(n),np.ones(n),repeat,broad].astype(float)
 num=[]
 for k in range(5):
  g=neu[k,:,0]-rec[:,0];raw=np.c_[g,g,fus[k,:,0]-rec[:,0],fus[k,:,0]-neu[k,:,0],fus[k,:,1]-rec[:,1],fus[k,:,2]-neu[k,:,2]]
  num.append(np.nan_to_num(raw)*masks)
 num=np.array(num)
 def complete(v):return np.concatenate([v[...,:2],(v[...,1]-v[...,0])[...,None],v[...,2:]],axis=-1)
 seedpoints=complete(num.sum(1)/masks.sum(0));point=seedpoints.mean(0)
 rng=np.random.default_rng(20261101+si);hdev=[];jdev=[];flat=num.transpose(1,0,2).reshape(n,30)
 for batch in range(50):
  w=rng.multinomial(n,np.full(n,1/n),size=100).astype(float);den=w@masks;assert np.all(den>0)
  b=(w@flat).reshape(100,5,6)/den[:,None,:]
  hb=complete(b.mean(1));sw=rng.multinomial(5,np.full(5,.2),size=100)/5
  jb=complete((b*sw[:,:,None]).sum(1));hdev.extend(abs(hb-point).max(1));jdev.extend(abs(jb-point).max(1))
 hr=float(np.quantile(hdev,.95));jr=float(np.quantile(jdev,.95))
 models=[]
 for name,arr in [('Popularity',measures(p['top_Popularity'],targets,histories,group)[None]),('PersonalRecency',rec[None]),('QuarterTransition',measures(p['top_QuarterTransition'],targets,histories,group)[None]),('BiLSTM',neu),('Fusion',fus)]:
  per_seed=np.array([[a[:,0].mean(),a[unseen,0].mean(),np.nanmean(a[:,1]),np.nanmean(a[:,2])] for a in arr])
  models.append({'model':name,'NDCG':float(per_seed[:,0].mean()),'unseen_NDCG':float(per_seed[:,1].mean()),'repeat_Recall':float(per_seed[:,2].mean()),'new_group_Recall':float(per_seed[:,3].mean()),'seed_SD':per_seed.std(0,ddof=1).tolist() if len(arr)>1 else [0.]*4})
 contrasts={name:{'estimate':float(point[j]),'seed_SD':float(seedpoints[:,j].std(ddof=1)),'seed_range':[float(seedpoints[:,j].min()),float(seedpoints[:,j].max())],'history_simultaneous_95_interval':[float(point[j]-hr),float(point[j]+hr)],'seed_history_simultaneous_95_interval':[float(point[j]-jr),float(point[j]+jr)]} for j,name in enumerate(names)}
 analyses[split]={'histories':n,'unseen_positive':int(unseen.sum()),'repeat_support':int(repeat.sum()),'broader_support':int(broad.sum()),'models':models,'contrasts':contrasts,'seed_contrasts':{str(s):dict(zip(names,seedpoints[k].tolist())) for k,s in enumerate(seeds)},'bootstrap_draws':5000,'bootstrap_seed':20261101+si,'history_radius':hr,'seed_history_radius':jr,'joint_aggregate_superiority':all(contrasts[k]['seed_history_simultaneous_95_interval'][0]>.005 for k in ['NDCG_vs_recency','NDCG_vs_BiLSTM']),'joint_specialised_retention':all(contrasts[k]['seed_history_simultaneous_95_interval'][0]>-.01 for k in ['repeat_Recall_vs_recency','new_group_Recall_vs_BiLSTM'])}
 # No person-level metric files are exported by this report.
 print(split,'complete',flush=True)
result={'exploratory_chronological_sensitivity':True,'analyses':analyses,'training':{str(s):json.loads((args.trained_dir/f'BiLSTM_{s}_training.json').read_text()) for s in seeds},'CPU_analysis_seconds':time.monotonic()-start,'bootstrap_caution':'Five fitted seeds do not capture architecture or hyperparameter selection uncertainty.'}
args.output.write_text(json.dumps(result,indent=2))
print('Wrote chronological point estimates and both interval families.')
