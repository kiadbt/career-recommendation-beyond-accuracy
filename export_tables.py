"""Integrity checks and aggregate table exports; Python standard library only."""
from pathlib import Path
import json,csv,hashlib,math
B=Path(__file__).resolve().parent
manifest=json.loads((B/'MANIFEST.json').read_text())
for name,digest in manifest['sha256'].items():
    assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
def load(name):return json.loads((B/'aggregate_results'/f'{name}.json').read_text())
def close(a,b):assert math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(a,b)
def interval(est,ci):
    assert ci[0]<=est<=ci[1]
    close(est-ci[0],ci[1]-est)
out=B/'generated_tables';out.mkdir(exist_ok=True)
def table(name,headers,rows):
    with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)
primary=load('primary');fusion=load('fusion');external=load('external');groups=load('groups');attention=load('attention');sensitivity=load('sensitivity')
ak=load('attention_karrierewege')
attention['datasets']['karrierewege']=ak['analyses']['all_histories']
cols=['model','NDCG_all','NDCG_unseen_positive','Recall_all']
table('Table5_original_models',cols,[[r[k] for k in cols] for r in primary['all_model_panel']])
table('Table6_occupational_groups',['group','resumes','gap','lower','upper'],[[r['group'],r['resumes'],r['gap'],*r['simultaneous_95_interval']] for r in groups['groups']])
table('Table7_fusion',['dataset','model','NDCG','repeat_Recall','new_group_Recall'],[[name]+[r[k] for k in ['model','NDCG','repeat_Recall','new_group_Recall']] for name,z in [('JobHop',fusion),('Decorte',external)] for r in z['models']])
table('Table8_attention',['dataset']+cols,[[name]+[r[k] for k in cols] for name,z in attention['datasets'].items() for r in z['models']])
contrasts=[]
c=primary['primary_NDCG_contrast']
close(c['estimates']['Delta_cohort'],c['estimates']['G_unseen_positive']-c['estimates']['G_all'])
for key,est in c['estimates'].items():
    ci=c['simultaneous_95_intervals'][key];interval(est,ci);contrasts.append(['primary',key,est,*ci])
for name,z in [('fusion',fusion),('external',external)]:
    for key,r in z['contrasts'].items():
        interval(r['estimate'],r['simultaneous_95_interval']);contrasts.append([name,key,r['estimate'],*r['simultaneous_95_interval']])
for name,z in attention['datasets'].items():
    c=z['attention_minus_recency'];close(c['estimates'][2],c['estimates'][1]-c['estimates'][0])
    for key,est,ci in zip(['all','unseen_positive','cohort_shift'],c['estimates'],c['simultaneous_95_intervals']):
        interval(est,ci);contrasts.append(['attention_'+name,key,est,*ci])
table('Contrast_intervals',['family','contrast','estimate','lower','upper'],contrasts)
rows=[]
for name,z in sensitivity['datasets'].items():
    assert len(z['rows'])==24
    for r in z['rows']:
        for key,est,ci in zip(sensitivity['contrast_order'],r['estimates'],r['simultaneous_95_intervals']):
            interval(est,ci);rows.append([name,r['k'],r['input_depth'],r['neural_weight'],key,est,*ci])
        if (r['k'],r['input_depth'],r['neural_weight'])==(10,10,.5):
            original=fusion if name=='fusion_validation' else external
            for key,est in zip(sensitivity['contrast_order'],r['estimates']):close(est,original['contrasts'][key]['estimate'])
table('TablesS7_S8_sensitivity',['dataset','k','input_depth','neural_weight','contrast','estimate','lower','upper'],rows)
table('TableS9_training',['seed','epochs_run','selected_epoch','best_tuning_NDCG'],[[seed,len(z['scores']),z['best_epoch'],z['best']] for seed,z in attention['training'].items()])
k=load('karrierewege')
table('Table9_KARRIEREWEGE',['analysis','model','NDCG','repeat_Recall','new_group_Recall'],[[name]+[r[key] for key in ['model','NDCG','repeat_Recall','new_group_Recall']] for name,analysis in k['analyses'].items() for r in analysis['models']])
extra=[]
for name,analysis in k['analyses'].items():
    c=analysis['contrasts'];close(c['cohort_shift']['estimate'],c['neural_minus_recency_unseen']['estimate']-c['neural_minus_recency_all']['estimate'])
    for key,r in c.items():
        interval(r['estimate'],r['simultaneous_95_interval']);extra.append([name,key,r['estimate'],*r['simultaneous_95_interval']])
table('TablesS11_S12_KARRIEREWEGE',['analysis','contrast','estimate','lower','upper'],extra)
table('TableS10_KARRIEREWEGE_attention',['analysis','contrast','estimate','lower','upper'],[[name,label,value,*interval] for name,r in ak['analyses'].items() for label,value,interval in zip(['all','unseen_positive','cohort_shift'],r['attention_minus_recency']['estimates'],r['attention_minus_recency']['simultaneous_95_intervals'])])
record={'integrity_passed':True,'aggregate_arithmetic_passed':True,'sensitivity_contrasts_exported':len(rows),'larger_external_contrasts_exported':len(extra),'table_files_exported':10,'scope':'Exports stored aggregates; does not regenerate bootstrap intervals or train models'}
(B/'EXPORT_VERIFICATION.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
