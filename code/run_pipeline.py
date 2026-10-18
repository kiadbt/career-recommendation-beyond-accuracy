"""Explicit staged rebuild; expensive training requires its own stage selection."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['embeddings','prepare','panel','chronological','score','report'])
    p.add_argument('--assets',required=True,type=Path);p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-seconds',type=int)
    a=p.parse_args();c=Path(__file__).parent;r=a.run_dir;data=r/'prepared';assets=a.assets
    if a.max_seconds is not None and a.max_seconds<=0:raise ValueError('The time cap must be positive')
    def run(script,*args):subprocess.run([sys.executable,str(c/script),*map(str,args)],check=True)
    cat=data/'catalogue_texts.json';emb=data/'catalogue_embeddings.npz'
    if a.stage=='embeddings':run('build_embeddings.py','--taxonomy',assets/'taxonomy.zip','--train',assets/'JobHop_v2_train.parquet','--encoder-dir',assets/'encoder','--tokenizer',assets/'encoder/tokenizer.json','--output-dir',data,'--device',a.device,'--max-seconds',a.max_seconds or 300)
    elif a.stage=='prepare':
        run('prepare_training.py','--train',assets/'JobHop_v2_train.parquet','--test',assets/'JobHop_v2_test.parquet','--validation',assets/'JobHop_v2_val.parquet','--catalogue',cat,'--embeddings',emb,'--output-dir',data)
        run('prepare_decorte.py','--csv',assets/'decorte_test.csv','--catalogue',cat,'--embeddings',emb,'--output',data/'decorte_queries.npz')
        run('prepare_karrierewege.py','--archive',assets/'karrierewege_test.zip','--catalogue',cat,'--embeddings',emb,'--jobhop-fitting-queries',data/'quarter_queries.npz','--decorte-queries',data/'decorte_queries.npz','--output-dir',data/'karrierewege')
    elif a.stage in ['panel','chronological']:
        chrono=a.stage=='chronological'
        run('train_models.py','--queries',data/('chronological_training.npz' if chrono else 'quarter_queries.npz'),'--embeddings',emb,'--output-dir',r/a.stage,'--models',*(['BiLSTM'] if chrono else ['MLP','BiLSTM','attention']),'--seeds',*([42,43,44,45,46] if chrono else [42,43]),'--device',a.device,'--max-seconds',a.max_seconds or 1800)
    elif a.stage=='score':
        for name in ['confirmation','fusion','decorte','karrierewege']:
            query=data/('karrierewege/queries.npz' if name=='karrierewege' else name+'_queries.npz')
            extra=['--neighbour-training',data/'quarter_queries.npz'] if name=='confirmation' else []
            run('score_models.py','--queries',query,'--embeddings',emb,'--trained-dir',r/'panel','--output',r/'predictions'/(name+'.npz'),'--device',a.device,'--max-seconds',a.max_seconds or 900,*extra)
        for name in ['confirmation','fusion']:
            run('score_models.py','--queries',data/('chronological_'+name+'.npz'),'--embeddings',emb,'--trained-dir',r/'chronological','--output',r/'predictions'/('chronological_'+name+'.npz'),'--device',a.device,'--max-seconds',a.max_seconds or 900)
    else:
        for name,analysis in [('confirmation','primary'),('fusion','fusion')]:
            run('report_panel.py','--analysis',analysis,'--queries',data/(name+'_queries.npz'),'--predictions',r/'predictions'/(name+'.npz'),'--catalogue',cat,'--output',r/(analysis+'_results.json'))
        for name in ['decorte','karrierewege']:
            run('report_external.py','--source',name,'--queries',data/('karrierewege/queries.npz' if name=='karrierewege' else 'decorte_queries.npz'),'--predictions',r/'predictions'/(name+'.npz'),'--catalogue',cat,'--output',r/(name+'_results.json'))
        run('report_chronological.py','--prepared-dir',data,'--predictions-dir',r/'predictions','--trained-dir',r/'chronological','--output',r/'chronological_results.json')

if __name__=='__main__':main()
