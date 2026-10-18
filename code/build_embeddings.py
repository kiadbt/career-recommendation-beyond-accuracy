"""Rebuild catalogue and frozen MiniLM vectors; no fitted predictor is loaded."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import argparse,csv,io,zipfile,json,time,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import AutoModel
from train_models import configure
from quarter_constructor import quarter

def main():
    p=argparse.ArgumentParser()
    for k in ['taxonomy','train','encoder-dir','tokenizer','output-dir']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-seconds',type=float,default=300)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'catalogue_embeddings.npz').exists():raise FileExistsError('Embeddings exist; use a new directory')
    digest=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    assert digest(a.encoder_dir/'model.safetensors')=='eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b'
    assert digest(a.tokenizer)=='2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8'
    with zipfile.ZipFile(a.taxonomy) as z:rows=list(csv.DictReader(io.StringIO(z.read('occupations.csv').decode('utf-8-sig'))))
    rows=sorted([r for r in rows if r['OCCUPATIONTYPE']=='escooccupation'],key=lambda r:r['CODE'])
    cat=[dict(code=r['CODE'],uri=r['ORIGINURI'],history=f"role: {r['PREFERREDLABEL']} \n description: {r['DESCRIPTION']}",target=f"esco role: {r['PREFERREDLABEL']} \n esco description: {r['DESCRIPTION']}") for r in rows]
    assert len(cat)==3007
    training_rows=pq.read_table(a.train,columns=['resume_id','matched_code','start_date','end_date']).to_pylist()
    invalid={r['resume_id'] for r in training_rows if quarter(r['start_date']) is None and quarter(r['end_date']) is None}
    known={r['matched_code'] for r in training_rows if r['resume_id'] not in invalid}-{'unknown'}
    del training_rows
    old=[i for i,r in enumerate(cat) if r['code'] in known];pending=[i for i,r in enumerate(cat) if r['code'] not in known]
    assert len(old)==2975 and len(pending)==32
    configure(a.device);start=time.monotonic();tok=Tokenizer.from_file(str(a.tokenizer));tok.no_padding();tok.no_truncation();pad=tok.token_to_id('<pad>')
    model=AutoModel.from_pretrained(str(a.encoder_dir),local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager').to(a.device).eval()
    output={k:np.empty((3007,384),np.float32) for k in ['history','target']}
    # Preserve the original batch composition: 2*2975 texts followed by 2*32 texts.
    for indices in [old,pending]:
        texts=[cat[i][k] for k in ['history','target'] for i in indices];vectors=[]
        for j in range(0,len(texts),32):
            if time.monotonic()-start>a.max_seconds:raise TimeoutError('Embedding time cap reached')
            batch=[]
            for e in tok.encode_batch(texts[j:j+32],add_special_tokens=False):e.truncate(126,direction='right');batch.append(tok.post_process(e,add_special_tokens=True))
            width=max(len(e.ids) for e in batch);ids=np.full((len(batch),width),pad,np.int64);mask=np.zeros_like(ids)
            for i,e in enumerate(batch):ids[i,:len(e.ids)]=e.ids;mask[i,:len(e.ids)]=1
            ids=torch.from_numpy(ids).to(a.device);mask=torch.from_numpy(mask).to(a.device)
            with torch.inference_mode():h=model(input_ids=ids,attention_mask=mask).last_hidden_state;m=mask.unsqueeze(-1);vectors.append(F.normalize((h*m).sum(1)/m.sum(1),dim=1).cpu().numpy())
        v=np.concatenate(vectors);output['history'][indices]=v[:len(indices)];output['target'][indices]=v[len(indices):]
    (a.output_dir/'catalogue_texts.json').write_text(json.dumps(cat,ensure_ascii=False),encoding='utf8')
    np.savez(a.output_dir/'catalogue_embeddings.npz',**output)
    (a.output_dir/'embedding_run.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,torch=torch.__version__,numpy=np.__version__,device=a.device,taxonomy_sha256=digest(a.taxonomy),train_sha256=digest(a.train),encoder_sha256=digest(a.encoder_dir/'model.safetensors'),tokenizer_sha256=digest(a.tokenizer)),indent=2))
    print('All 6014 catalogue texts encoded.',flush=True)

if __name__=='__main__':main()
