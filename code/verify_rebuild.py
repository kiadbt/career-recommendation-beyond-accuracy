"""Compare local rebuilt arrays and fitted tensors against aggregate reference hashes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def array_hash(a):
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def tensor_hash(state):
    h=hashlib.sha256()
    for k in sorted(state):h.update(k.encode());h.update(array_hash(state[k].detach().cpu().numpy()).encode())
    return h.hexdigest()

def json_hash(path):
    value=json.loads(path.read_text(encoding='utf8'))
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs-only',action='store_true');a=p.parse_args()
    ref=json.loads((Path(__file__).resolve().parents[1]/'REBUILD_REFERENCE.json').read_text());checks={}
    for file,digest in ref.get('json',{}).items():
        path=a.run_dir/'prepared'/file;checks[file]={'canonical_contents':path.exists() and json_hash(path)==digest}
    for file,arrays in ref['arrays'].items():
        path=a.run_dir/'prepared'/file
        if not path.exists():checks[file]={'missing':True};continue
        with np.load(path) as q:checks[file]={k:k in q and array_hash(q[k])==v for k,v in arrays.items()}
    if not a.inputs_only:
        import torch
        for file,row in ref['training'].items():
            path=a.run_dir/file;record=path.with_name(path.name.replace('_best.pt','_training.json'))
            if not path.exists() or not record.exists():checks[file]={'missing':True};continue
            r=json.loads(record.read_text());checks[file]={'checkpoint_tensors':tensor_hash(torch.load(path,map_location='cpu',weights_only=True))==row['tensor_sha256'],'training_trace':r['scores']==row['scores'] and r['losses']==row['losses'],'best_epoch':r['best_epoch']==row['best_epoch'],'adequate':bool(r['adequate'])}
    passed=all(not v.get('missing',False) and all(v.values()) for v in checks.values())
    a.output.write_text(json.dumps(dict(passed=passed,inputs_only=a.inputs_only,checks=checks),indent=2));print('Exact reference verification:',passed)
    if not passed:raise SystemExit('Inspect the report. Missing assets or numerical differences must not be reported as exact reproduction.')

if __name__=='__main__':main()
