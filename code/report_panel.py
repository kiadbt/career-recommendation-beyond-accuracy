"""Recalculate primary model comparisons and the original four-contrast fusion family."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse,json
from pathlib import Path
import numpy as np
from training_metrics import target_sets,metrics,cohort_groups,cohort_contrast
from fusion_metrics import measures,intervals

def main():
    p=argparse.ArgumentParser()
    for k in ['queries','predictions','catalogue','output']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--analysis',choices=['primary','fusion'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    q=dict(np.load(a.queries));z=dict(np.load(a.predictions));assert np.array_equal(q['pid'],z['pid'])
    ix=np.arange(len(q['pid']));truth=target_sets(q,ix);groups=cohort_groups(q,ix);unseen=groups!='repeat_only'
    values={k[4:]:metrics(v,truth) for k,v in z.items() if k.startswith('top_')}
    panel={k:v for k,v in values.items() if k in ['Popularity','PersonalRecency','QuarterTransition','HistoryNeighbour']}
    for name in ['MLP','BiLSTM','attention']:
        if name+'_42' in values:panel[name]=(values[name+'_42']+values[name+'_43'])/2
    rows={k:dict(NDCG_all=float(v[:,0].mean()),NDCG_unseen_positive=float(v[unseen,0].mean()),Recall_all=float(v[:,1].mean()),Recall_unseen_positive=float(v[unseen,1].mean())) for k,v in panel.items()}
    out=dict(histories=len(ix),subgroup_support={g:int((groups==g).sum()) for g in ['repeat_only','unseen_only','mixed']},models=rows)
    if a.analysis=='primary':
        out['primary_NDCG_contrast']=cohort_contrast(panel['BiLSTM'][:,0],panel['PersonalRecency'][:,0],groups)
        out['secondary_Recall_contrast']=cohort_contrast(panel['BiLSTM'][:,1],panel['PersonalRecency'][:,1],groups)
    else:
        cat=json.loads(a.catalogue.read_text(encoding='utf8'));gm={i:r['code'][:2] for i,r in enumerate(cat)}
        histories=[{int(c) for b in q['history_blocks'][i,:q['length'][i]] for c in q['bank_members'][b,:q['bank_size'][b]]} for i in ix]
        rec=measures(z['top_PersonalRecency'],truth,histories,gm)
        neu=sum(measures(z[f'top_BiLSTM_{s}'],truth,histories,gm) for s in [42,43])/2
        fus=sum(measures(z[f'top_Fusion_BiLSTM_{s}'],truth,histories,gm) for s in [42,43])/2
        point,ci=intervals(fus,rec,neu)
        out['fusion_contrasts']={k:dict(estimate=float(v),simultaneous_95_interval=c.tolist()) for k,v,c in zip(['NDCG_vs_recency','NDCG_vs_BiLSTM','repeat_Recall_vs_recency','new_group_Recall_vs_BiLSTM'],point,ci)}
    a.output.write_text(json.dumps(out,indent=2));print('Wrote',a.output)

if __name__=='__main__':main()
