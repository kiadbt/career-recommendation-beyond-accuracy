"""Train the reported predictors from random initialization, with bounded resources."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from baseline_models import MLPTransformation,CareerBiLSTM
from attention_model import QuarterSelfAttention
from training_helpers import pool_history,multi_target_loss
from training_metrics import ranking,metrics,target_sets,fit_counts

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Inputs:
    def __init__(self,path,embeddings,device):
        self.q=dict(np.load(path));q=self.q;self.device=device
        self.bank=torch.from_numpy(q['bank_history_embeddings']).to(device)
        self.hist=torch.from_numpy(q['history_blocks'].astype(np.int64)).to(device)
        self.lengths=torch.from_numpy(q['length'].astype(np.int64))
        self.target=torch.from_numpy(np.load(embeddings)['target']).to(device)
        self.members=torch.from_numpy(q['bank_members'].astype(np.int64)).to(device)
        self.sizes=torch.from_numpy(q['bank_size'].astype(np.int64)).to(device)
        self.tb=torch.from_numpy(q['target_block'].astype(np.int64)).to(device)
        if 'fit' in q:
            self.weights=torch.from_numpy(q['prefix_weight'].astype(np.float32)).to(device)
            self.scale=len(q['fit'])/len(np.unique(q['pid'][q['fit']]))
        self.pooled=None
    def sequence(self,ix):
        l=self.lengths[ix];return self.bank[self.hist[ix,:int(l.max())]],l
    def logits(self,model,name,ix):
        if name=='MLP':
            if self.pooled is None:
                self.pooled=torch.empty((len(self.hist),384),device=self.device)
                with torch.inference_mode():
                    for a in range(0,len(self.hist),1024):
                        j=np.arange(a,min(a+1024,len(self.hist)));x,l=self.sequence(j);self.pooled[j]=pool_history(x,l.to(self.device))
            return model.compute_similarity(model(self.pooled[ix]),self.target)
        x,l=self.sequence(ix);return model(x,l)
    def loss(self,logits,ix):
        b=self.tb[ix];return multi_target_loss(logits,self.members[b],self.sizes[b],self.weights[ix],self.scale)

def configure(device):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
        torch.cuda.set_per_process_memory_fraction(min(.9,4.5*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.cuda.reset_peak_memory_stats()

def make(name,seed,device):
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    model=(MLPTransformation(384,512,384,.1) if name=='MLP' else CareerBiLSTM(384,16,3007,.1) if name=='BiLSTM' else QuarterSelfAttention()).to(device)
    opt=torch.optim.RMSprop(model.parameters(),lr=.001) if name=='BiLSTM' else torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    return model,opt

def predict(model,name,data,indices,check=lambda:None,k=10):
    model.eval();parts=[]
    with torch.inference_mode():
        for a in range(0,len(indices),128):check();parts.append(ranking(data.logits(model,name,indices[a:a+128]).cpu().numpy(),k))
    return np.concatenate(parts)

def main():
    p=argparse.ArgumentParser()
    for k in ['queries','embeddings','output-dir']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--models',nargs='+',choices=['MLP','BiLSTM','attention'],default=['MLP','BiLSTM'])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43]);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--max-seconds',type=float,default=1800);p.add_argument('--max-epochs',type=int,default=45)
    a=p.parse_args()
    if a.max_seconds<=0 or not 1<=a.max_epochs<=45:raise ValueError('Positive time cap and 1..45 epochs required')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if any(a.output_dir.iterdir()):raise FileExistsError('Training output must be empty; no implicit overwrite or budget reset')
    configure(a.device);start=time.monotonic()
    def check():
        if time.monotonic()-start>a.max_seconds:raise TimeoutError('Training resource cap reached; partial records retained')
    data=Inputs(a.queries,a.embeddings,a.device);q=data.q
    assert not set(q['pid'][q['fit']])&set(q['pid'][q['tune']])
    pop,tr=fit_counts(q,3007);np.savez_compressed(a.output_dir/'count_models.npz',popularity=pop,transition=tr)
    manifest=dict(queries_sha256=sha(a.queries),embeddings_sha256=sha(a.embeddings),code_sha256={n:sha(Path(__file__).parent/n) for n in ['train_models.py','baseline_models.py','attention_model.py','training_helpers.py','training_metrics.py']},torch=torch.__version__,numpy=np.__version__,device=a.device,models=a.models,seeds=a.seeds,max_seconds=a.max_seconds,max_epochs=a.max_epochs,complete=False)
    (a.output_dir/'run.json').write_text(json.dumps(manifest,indent=2))
    adequacy=[]
    try:
        for name in a.models:
            for seed in a.seeds:
                model,opt=make(name,seed,a.device);rng=np.random.default_rng(seed);scores=[];losses=[];best=-1.;stale=0;best_epoch=0
                for epoch in range(1,a.max_epochs+1):
                    check();model.train();order=rng.permutation(q['fit']);total=0.
                    for j in range(0,len(order),128):
                        check();ix=order[j:j+128];opt.zero_grad(set_to_none=True);loss=data.loss(data.logits(model,name,ix),ix)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix)
                    scores.append(float(metrics(predict(model,name,data,q['tune'],check),target_sets(q,q['tune']))[:,0].mean()));losses.append(total/len(order))
                    if scores[-1]>best:
                        best=scores[-1];best_epoch=epoch;stale=0;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},a.output_dir/f'{name}_{seed}_best.pt')
                    else:stale+=1
                    done=epoch>=10 and stale>=5
                    adequate=done or (epoch==45 and max(scores)-max(scores[:40])<.002)
                    record=dict(model=name,seed=seed,scores=scores,losses=losses,best=best,best_epoch=best_epoch,stale=stale,adequate=adequate,done=done or epoch==a.max_epochs)
                    (a.output_dir/f'{name}_{seed}_training.json').write_text(json.dumps(record,indent=2))
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if a.device=='cuda' else None,shuffle_state=rng.bit_generator.state),a.output_dir/f'{name}_{seed}_last.pt')
                    print(json.dumps(dict(model=name,seed=seed,epoch=epoch,tune_ndcg=scores[-1],seconds=round(time.monotonic()-start,1))),flush=True)
                    if done:break
                adequacy.append(bool(adequate));del model,opt
        manifest['complete']=True
    finally:
        manifest.update(all_stopping_adequacy_checks_pass=len(adequacy)==len(a.models)*len(a.seeds) and all(adequacy),seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else 0)
        (a.output_dir/'run.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
