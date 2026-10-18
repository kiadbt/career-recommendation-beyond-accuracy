"""Rebuild original person partitions and all-prefix queries from JobHop train."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from quarter_constructor import quarter,build

def clean(path):
    d=pd.read_parquet(path,columns=['resume_id','matched_code','start_date','end_date'])
    d['q']=d.start_date.map(quarter).fillna(d.end_date.map(quarter))
    invalid=set(d.loc[d.q.isna(),'resume_id'])
    return d.loc[~d.resume_id.isin(invalid)&d.matched_code.ne('unknown')].copy()

def partitions(d):
    eligible=d.groupby('resume_id').q.nunique(); eligible=eligible[eligible>=2].index
    dig={int(p):hashlib.sha256(('JobHop-common-quarter-v1:'+str(int(p))).encode()).hexdigest() for p in eligible}
    fit=sorted(p for p in eligible if int(dig[int(p)][:8],16)%5!=0)
    dev=sorted(p for p in eligible if int(dig[int(p)][:8],16)%5==0)
    fit=sorted(sorted(fit,key=lambda p:dig[int(p)])[:30000])
    dev=sorted(dev,key=lambda p:dig[int(p)])[:6000]
    dev.sort(key=lambda p:hashlib.sha256(('step-temporal-dev-v1|'+str(int(p))).encode()).hexdigest())
    return {k:list(map(int,v)) for k,v in dict(fit=fit,tune=sorted(dev[:3000]),screen=sorted(dev[3000:])).items()}

def pack(records,emb,chronological=False):
    bank=[()];lookup={():0};hist=[];targets=[];finals=[]
    def index(b):
        if b not in lookup:lookup[b]=len(bank);bank.append(b)
        return lookup[b]
    for pid,h,t,w in records:
        hist.append([index(b[1]) for b in h]);targets.append(index(t[1]))
    dtype=np.int64 if chronological else np.int32
    lens=np.array(list(map(len,hist)),dtype); hb=np.zeros((len(hist),int(lens.max())),dtype)
    dates=np.full(hb.shape,-1,np.int64)
    for i,(h,r) in enumerate(zip(hist,records)):hb[i,:len(h)]=h;dates[i,:len(h)]=[b[0] for b in r[1]]
    sizes=np.array(list(map(len,bank)),dtype);members=np.full((len(bank),max(sizes)),-1,dtype);vec=np.zeros((len(bank),384),np.float32)
    for i,b in enumerate(bank[1:],1):
        members[i,:len(b)]=b;v=emb[list(b)].mean(0);vec[i]=v/max(float(np.linalg.norm(v)),1e-12)
    q=dict(pid=np.array([r[0] for r in records],np.int64),history_blocks=hb,length=lens,target_block=np.array(targets,dtype),bank_size=sizes,bank_members=members,bank_history_embeddings=vec,prefix_weight=np.array([r[3] for r in records],np.float32 if chronological else np.float64))
    if chronological:q.update(history_dates=dates,target_date=np.array([r[2][0] for r in records],np.int64))
    return q

def records(d,ids,ci,mode='all'):
    d=d[d.resume_id.isin(ids)].copy()
    if mode=='fit':d=d[d.q<=8059]
    d['code']=d.matched_code.map(ci);assert d.code.notna().all()
    out=[]
    for pid,g in d.groupby('resume_id',sort=True):
        blocks=[(int(t),tuple(sorted(set(map(int,b.code))))) for t,b in g.groupby('q',sort=True)]
        if len(blocks)<2:continue
        if mode in ['all','fit']:chosen=list(range(1,len(blocks)))
        else:chosen=[j for j,(t,_) in enumerate(blocks) if j>0 and (8060<=t<=8063 if mode=='tune' else t>=8064)][:1]
        for j in chosen:out.append((int(pid),blocks[:j],blocks[j],1/len(chosen)))
    return out

def main():
    p=argparse.ArgumentParser()
    for k in ['train','catalogue','embeddings','output-dir']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--test',type=Path);p.add_argument('--validation',type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'quarter_queries.npz').exists():raise FileExistsError('Use a new output directory')
    cat=json.loads(a.catalogue.read_text(encoding='utf8'));ci={r['code']:i for i,r in enumerate(cat)};emb=np.load(a.embeddings)['history']
    d=clean(a.train);splits=partitions(d)
    assert list(map(len,splits.values()))==[30000,3000,3000]
    q=pack(records(d,sum(splits.values(),[]),ci),emb)
    finals=np.r_[np.flatnonzero(q['pid'][:-1]!=q['pid'][1:]),len(q['pid'])-1]
    q.update(finals=finals,fit=np.flatnonzero(np.isin(q['pid'],splits['fit'])),tune=finals[np.isin(q['pid'][finals],splits['tune'])],screen=finals[np.isin(q['pid'][finals],splits['screen'])])
    np.savez(a.output_dir/'quarter_queries.npz',**q)
    (a.output_dir/'split_ids.json').write_text(json.dumps(splits))
    f=records(d,splits['fit'],ci,'fit');t=records(d,splits['tune'],ci,'tune');cq=pack(f+t,emb,True);cq.update(fit=np.arange(len(f)),tune=np.arange(len(f),len(f)+len(t)))
    np.savez(a.output_dir/'chronological_training.npz',**cq)
    for name,path in [('confirmation',a.test),('fusion',a.validation)]:
        if path:
            raw=pd.read_parquet(path);eq,support=build(raw,cat,emb);assert not set(eq['pid'])&set(sum(splits.values(),[]))
            np.savez(a.output_dir/(name+'_queries.npz'),**eq)
            cr=records(clean(path),eq['pid'],ci,'test');np.savez(a.output_dir/('chronological_'+name+'.npz'),**pack(cr,emb,True))
    print(json.dumps({'partitions':{k:len(v) for k,v in splits.items()},'queries':len(q['pid']),'chronological_fit':len(f),'chronological_tune':len(t)}))

if __name__=='__main__':main()
