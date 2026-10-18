"""Score separately prepared histories with frozen trained checkpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from train_models import Inputs,configure,make,predict,pool_history
from training_metrics import ranking,neighbour_votes
from quarter_constructor import count_scores
from fusion_metrics import fuse

def main():
    p=argparse.ArgumentParser()
    for k in ['queries','embeddings','trained-dir','output']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--neighbour-training',type=Path)
    p.add_argument('--max-seconds',type=float,default=900)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    run=json.loads((a.trained_dir/'run.json').read_text())
    if not run['complete']:raise RuntimeError('Training is incomplete')
    for record in a.trained_dir.glob('*_training.json'):
        if not json.loads(record.read_text())['adequate']:raise RuntimeError('Training adequacy not established: '+record.name)
    configure(a.device);start=time.monotonic()
    def check():
        if time.monotonic()-start>a.max_seconds:raise TimeoutError('Scoring time cap reached')
    data=Inputs(a.queries,a.embeddings,a.device);q=data.q;ix=np.arange(len(q['pid']));out={'pid':q['pid']}
    count=np.load(a.trained_dir/'count_models.npz');pop=count['popularity'];tr=count['transition']
    for name in ['Popularity','PersonalRecency','QuarterTransition']:
        check()
        full=np.concatenate([ranking(count_scores(q,ix[j:j+128],name,pop,tr),50) for j in range(0,len(ix),128)])
        out['top_'+name]=full[:,:10];out['depth50_'+name]=full
    print('Count baselines scored for',len(ix),'histories',flush=True)
    for path in sorted(a.trained_dir.glob('*_best.pt')):
        name,seed,_=path.stem.rsplit('_',2);model,opt=make(name,int(seed),a.device);del opt
        model.load_state_dict(torch.load(path,map_location=a.device,weights_only=True));full=predict(model,name,data,ix,check,k=50)
        out['top_'+name+'_'+seed]=full[:,:10];out['depth50_'+name+'_'+seed]=full
        if name in ['BiLSTM','attention']:out['top_Fusion_'+name+'_'+seed]=fuse(out['top_PersonalRecency'],out['top_'+name+'_'+seed])
        print('Scored',name,seed,'elapsed seconds',round(time.monotonic()-start,1),flush=True)
        del model
    if a.neighbour_training:
        print('Scoring the history-neighbour control',flush=True)
        fitdata=Inputs(a.neighbour_training,a.embeddings,a.device);fit=fitdata.q['fit']
        def pool(d,ids):
            chunks=[]
            with torch.inference_mode():
                for j in range(0,len(ids),1024):x,l=d.sequence(ids[j:j+1024]);chunks.append(pool_history(x,l.to(a.device)))
            return torch.cat(chunks)
        trainx=pool(fitdata,fit);queryx=pool(data,ix);parts=[]
        for j in range(0,len(ix),32):
            check()
            scores=queryx[j:j+32]@trainx.T;order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:50]
            parts.append(ranking(neighbour_votes(fit[order.cpu().numpy()],scores.gather(1,order).cpu().numpy(),fitdata.q,pop)))
        out['top_HistoryNeighbour']=np.concatenate(parts)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out)
    print(json.dumps({'histories':len(ix),'rankings':list(out)[1:]}))

if __name__=='__main__':main()
