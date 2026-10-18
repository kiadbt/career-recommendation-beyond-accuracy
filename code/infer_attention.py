"""Frozen inference from local assets; defaults to CPU, CUDA is explicit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from attention_model import QuarterSelfAttention

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['queries','checkpoint-42','checkpoint-43','selection','out']:ap.add_argument('--'+name,required=True,type=Path)
 ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--seconds',type=int,default=180);args=ap.parse_args()
 assert not args.out.exists(),'Do not overwrite predictions'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 selection=json.loads(args.selection.read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 if args.device=='cuda':
  assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  torch.cuda.set_per_process_memory_fraction(min(1,4.5*1024**3/torch.cuda.get_device_properties(0).total_memory));torch.cuda.reset_peak_memory_stats()
 q=dict(np.load(args.queries,allow_pickle=False));bank=torch.from_numpy(q['bank_history_embeddings']).to(args.device);hist=torch.from_numpy(q['history_blocks'].astype(np.int64)).to(args.device);lengths=torch.from_numpy(q['length'].astype(np.int64));out={'pid':q['pid']};start=time.monotonic()
 assert int(lengths.max())<=256
 for seed,path in [(42,args.checkpoint_42),(43,args.checkpoint_43)]:
  assert sha(path)==selection['checkpoint_hashes'][str(seed)]
  model=QuarterSelfAttention().to(args.device);model.load_state_dict(torch.load(path,map_location=args.device,weights_only=True));model.eval();parts=[]
  with torch.inference_mode():
   for a in range(0,len(lengths),128):
    if time.monotonic()-start>=args.seconds:raise TimeoutError('Inference time cap exceeded')
    l=lengths[a:a+128];scores=model(bank[hist[a:a+128,:int(l.max())]],l).cpu().numpy();parts.append(np.argsort(-scores,axis=1,kind='stable')[:,:10])
  out[f'top_attention_{seed}']=np.concatenate(parts);del model
 np.savez_compressed(args.out,**out)
 args.out.with_suffix('.receipt.json').write_text(json.dumps({'optimizer_steps':0,'seconds':time.monotonic()-start,'device':args.device,'queries_sha256':sha(args.queries),'predictions_sha256':sha(args.out)},indent=2))
if __name__=='__main__':main()
