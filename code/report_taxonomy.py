import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import sys,json,time,hashlib,csv,io,zipfile
import numpy as np
import argparse
ap=argparse.ArgumentParser()
for name in ['prepared-dir','predictions-dir','output-dir']:ap.add_argument('--'+name,type=Path,required=True)
ap.add_argument('--taxonomy',type=Path)
args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
if (args.output_dir/'results.json').exists():raise FileExistsError('Output already exists')
HERE=args.output_dir;P=args.prepared_dir;ROOT=Path(__file__).resolve().parents[1]
W=HERE
from fusion_metrics import fuse
start=time.monotonic();cat=json.loads((P/'catalogue_texts.json').read_text(encoding='utf8'))
codes=[c['code'].split('.')[0] for c in cat]
assert all(len(c)==4 and c.isdigit() for c in codes)
archive=args.taxonomy
if archive is None:raise ValueError('--taxonomy is required')
with zipfile.ZipFile(archive) as z:
 def rows(n):return list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig'))))
 occ={r['CODE']:r for r in rows('occupations.csv') if r['OCCUPATIONTYPE']=='escooccupation'}
 groups={r['ID']:r for r in rows('occupation_groups.csv')};parents={}
 for r in rows('occupation_hierarchy.csv'):parents.setdefault(r['CHILDID'],set()).add(r['PARENTID'])
 for c in cat:
  pending=list(parents.get(occ[c['code']]['ID'],[]));seen=set();anc=set()
  while pending:
   node=pending.pop()
   if node in seen:continue
   seen.add(node)
   if node in groups and groups[node]['GROUPTYPE']=='iscogroup':anc.add(groups[node]['CODE'])
   pending.extend(parents.get(node,[]))
  assert all(c['code'][:depth] in anc for depth in range(1,5)),c['code']
mapping={depth:np.array([c[:depth] for c in codes]) for depth in range(1,5)}
sources=[('JobHop',P/'fusion_queries.npz',args.predictions_dir/'fusion.npz'),('Decorte',P/'decorte_queries.npz',args.predictions_dir/'decorte.npz'),('KARRIEREWEGE',P/'karrierewege/queries.npz',args.predictions_dir/'karrierewege.npz')]
all_results={};metrics={};files=[archive,P/'catalogue_texts.json',Path(__file__)]
for source,qpath,ppath in sources:
 q=dict(np.load(qpath));p=dict(np.load(ppath));assert np.array_equal(q['pid'],p['pid']);files.extend([qpath,ppath])
 n=len(q['pid']);r=p['top_PersonalRecency'];ns=[p[f'top_BiLSTM_{seed}'] for seed in (42,43)];fs=[p[f'top_Fusion_BiLSTM_{seed}'] for seed in (42,43)]
 for neu,fus in zip(ns,fs):assert np.array_equal(fuse(r,neu),fus)
 targets=[];histories=[]
 for i in range(n):
  b=q['target_block'][i];targets.append(set(map(int,q['bank_members'][b,:q['bank_size'][b]])))
  histories.append({int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]})
 assert all(max(t|h)<len(cat) and min(t|h)>=0 for t,h in zip(targets,histories))
 discount=1/np.log2(np.arange(2,12))
 def ndcg(rankings):return np.array([np.array([int(c) in t for c in row])@discount/discount[:min(10,len(t))].sum() for row,t in zip(rankings,targets)])
 rr=ndcg(r);nn=np.array([ndcg(x) for x in ns]);ff=np.array([ndcg(x) for x in fs]);dn=np.c_[ff.mean(0)-rr,ff.mean(0)-nn.mean(0)]
 overlap=np.array([[len(set(a)&set(b)) for a,b in zip(r,neu)] for neu in ns]).mean(0)
 repeat=np.array([len(t&h)/len(t) for t,h in zip(targets,histories)])
 cohort=np.where(repeat==1,0,np.where(repeat==0,2,1));sizes=np.array(list(map(len,targets)))
 broad=np.zeros((n,4),bool);dr=np.zeros((n,4));raw_recall={}
 for depth,group in mapping.items():
  vals=np.full((n,3),np.nan)
  for i,(t,h) in enumerate(zip(targets,histories)):
   hg=set(group[list(h)]);bt={c for c in t if group[c] not in hg}
   if not bt:continue
   broad[i,depth-1]=True
   vals[i]=[len(set(r[i])&bt)/len(bt),sum(len(set(neu[i])&bt)/len(bt) for neu in ns)/2,sum(len(set(fus[i])&bt)/len(bt) for fus in fs)/2]
   dr[i,depth-1]=vals[i,2]-vals[i,1]
  raw_recall[depth]=vals
 samples=[(source,np.arange(n))]
 if source=='KARRIEREWEGE':samples.append((source+' unique',q['sensitivity_indices']))
 for label,ix in samples:
  count=len(ix);eligible=broad[ix].astype(float);num=dr[ix];point=num.sum(0)/eligible.sum(0)
  rng=np.random.default_rng(20261030+len(all_results));dev=[]
  for batch in range(50):
   w=rng.multinomial(count,np.full(count,1/count),size=100).astype(float)
   den=w@eligible;assert np.all(den>0)
   boot=w@num/den;dev.extend(abs(boot-point).max(1))
  radius=float(np.quantile(dev,.95))
  rows=[]
  for depth in range(1,5):
   vals=raw_recall[depth][ix];avg=np.nanmean(vals,axis=0)
   rows.append(dict(depth=depth,eligible=int(eligible[:,depth-1].sum()),recency=float(avg[0]),neural=float(avg[1]),fusion=float(avg[2]),difference=float(point[depth-1]),simultaneous_95_interval=[float(point[depth-1]-radius),float(point[depth-1]+radius)]))
  strata=[]
  for variable,values in [('cohort',cohort),('history_blocks',np.where(q['length']==1,0,np.where(q['length']<=4,1,2))),('target_size',(sizes>1).astype(int)),('agreement',(overlap>=5).astype(int))]:
   names={'cohort':['repeat_only','mixed','unseen_only'],'history_blocks':['1','2-4','5+'],'target_size':['single','multiple'],'agreement':['below_5','at_least_5']}[variable]
   for value,name in enumerate(names):
    j=ix[values[ix]==value]
    if len(j):strata.append(dict(variable=variable,stratum=name,n=len(j),fusion_minus_recency=float(dn[j,0].mean()),fusion_minus_neural=float(dn[j,1].mean())))
  all_results[label]=dict(n=count,mean_history_blocks=float(q['length'][ix].mean()),median_history_blocks=float(np.median(q['length'][ix])),mean_target_size=float(sizes[ix].mean()),multi_target_percent=float((sizes[ix]>1).mean()*100),mean_repeated_target_fraction=float(repeat[ix].mean()),mean_top10_intersection=float(overlap[ix].mean()),cohort_counts=np.bincount(cohort[ix],minlength=3).tolist(),fusion_minus_recency=float(dn[ix,0].mean()),fusion_minus_neural=float(dn[ix,1].mean()),taxonomy=rows,strata=strata,seed_contrasts=[dict(seed=seed,fusion_minus_recency=float((ff[k,ix]-rr[ix]).mean()),fusion_minus_neural=float((ff[k,ix]-nn[k,ix]).mean())) for k,seed in enumerate((42,43))])
  # Exact decomposition into outcome-cohort shares times conditional means.
  cs=[x for x in strata if x['variable']=='cohort']
  for key in ['fusion_minus_recency','fusion_minus_neural']:assert abs(sum(x['n']/count*x[key] for x in cs)-all_results[label][key])<1e-12
 print(label, 'computed',flush=True)
 metrics[source]={'ndcg_difference':dn,'broad_difference':dr,'eligible':broad,'cohort':cohort,'length':q['length'],'overlap':overlap}
 # Person-level intermediates stay private, outside the release package.
 np.savez_compressed(W/(source+'_private_metrics.npz'),**metrics[source])
# Standardise repeat-only versus unseen-positive composition: common support in all sources.
counts=np.array([all_results[s]['cohort_counts'] for s,_,_ in sources]);weights=np.array([counts[:,0].sum(),counts[:,1:].sum()]);weights=weights/weights.sum()
for s,_,_ in sources:
 m=metrics[s];v=m['ndcg_difference'];g=m['cohort']!=0
 all_results[s]['pooled_binary_cohort_weights']=weights.tolist()
 all_results[s]['standardised_fusion_minus_recency']=float(weights@np.array([v[~g,0].mean(),v[g,0].mean()]))
 all_results[s]['standardised_fusion_minus_neural']=float(weights@np.array([v[~g,1].mean(),v[g,1].mean()]))
# Reconcile the two-digit results with already published local aggregates.
for s,path in [('JobHop',ROOT/'aggregate_results/fusion.json'),('Decorte',ROOT/'aggregate_results/external.json')]:
 old=json.loads(path.read_text());models={x['model']:x for x in old['models']}
 assert abs(all_results[s]['taxonomy'][1]['difference']-(models['Fusion']['new_group_Recall']-models['BiLSTM']['new_group_Recall']))<1e-7
out=dict(exploratory=True,bootstrap_draws=5000,interval_family='four taxonomy depths per sample',catalogue_verified=3007,results=all_results,cpu_seconds=time.monotonic()-start,gpu_seconds=0)
(W/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
