"""Study held-out constructor and count baseline; no import-time data access."""
import re
import numpy as np

def quarter(x):
    if not isinstance(x,str):return None
    m=re.fullmatch(r'Q([1-4])\s+(\d{4})',x.strip())
    return int(m[2])*4+int(m[1])-1 if m else None

def build(df,catalogue,embeddings):
    df=df.copy();raw_n=len(df);raw_people=int(df.resume_id.nunique())
    df['q']=df.start_date.map(quarter).fillna(df.end_date.map(quarter))
    invalid=set(df.loc[df.q.isna(),'resume_id'])
    df=df.loc[~df.resume_id.isin(invalid)&(df.matched_code!='unknown')].copy()
    assert df.matched_code.map(lambda x:isinstance(x,str) and bool(x)).all()
    assert df.resume_id.notna().all()
    ci={r['code']:i for i,r in enumerate(catalogue)}
    extra=sorted(set(df.matched_code)-set(ci))
    raw_code_index=dict(ci);raw_code_index.update({c:len(ci)+i for i,c in enumerate(extra)})
    df['code']=df.matched_code.map(raw_code_index)
    df=df.sort_values(['resume_id','q'])
    bank=[()];lookup={():0};histories=[];targets=[];ids=[];one_quarter=0
    for pid,g in df.groupby('resume_id',sort=True):
        blocks=[tuple(sorted(set(b.code))) for _,b in g.groupby('q',sort=True)]
        if len(blocks)<2:
            one_quarter+=1;continue
        bb=[]
        for block in blocks:
            if block not in lookup:lookup[block]=len(bank);bank.append(block)
            bb.append(lookup[block])
        histories.append(bb[:-1]);targets.append(bb[-1]);ids.append(int(pid))
    assert ids
    length=np.asarray(list(map(len,histories)),np.int32)
    history=np.zeros((len(ids),int(length.max())),np.int32)
    for i,h in enumerate(histories):history[i,:len(h)]=h
    size=np.asarray(list(map(len,bank)),np.int32)
    members=np.full((len(bank),int(size.max())),-1,np.int32)
    vectors=np.zeros((len(bank),384),np.float32)
    for i,b in enumerate(bank[1:],1):
        members[i,:len(b)]=b
        known=[c for c in b if c<len(ci)]
        if known:
            v=embeddings[known].sum(0)/len(b)
            vectors[i]=v/max(float(np.linalg.norm(v)),1e-12)
    q=dict(pid=np.asarray(ids,np.int64),length=length,history_blocks=history,
           target_block=np.asarray(targets,np.int32),bank_members=members,bank_size=size,
           bank_history_embeddings=vectors)
    hist_oov=0;target_oov=0;target_entries_oov=0
    for i in range(len(ids)):
        t=bank[targets[i]];target_oov+=any(c>=len(ci) for c in t)
        target_entries_oov+=sum(c>=len(ci) for c in t)
        hist_oov+=any(c>=len(ci) for b in histories[i] for c in bank[b])
    return q,dict(raw_rows=raw_n,raw_resume_ids=raw_people,invalid_date_resumes=len(invalid),
        retained_rows=len(df),retained_resumes=int(df.resume_id.nunique()),
        fewer_than_two_retained_quarters=one_quarter,eligible_final_quarter_queries=len(ids),
        out_of_catalogue_raw_codes=extra,resumes_with_OOV_history=hist_oov,
        resumes_with_OOV_targets=target_oov,OOV_target_entries=target_entries_oov,
        catalogue_expanded=False,source_unknown_rows_removed=True,
        strict_earlier_quarter_history=True,maximum_history_quarters=int(length.max()))

def count_scores(q,indices,name,pop,transition):
    n=len(pop);scores=np.tile(pop,(len(indices),1))
    if name=='Popularity':return scores
    for row,i in enumerate(indices):
        blocks=q['history_blocks'][i,:q['length'][i]]
        if name=='QuarterTransition':
            b=blocks[-1];codes=q['bank_members'][b,:q['bank_size'][b]];known=codes[codes<n]
            scores[row]=transition[known].mean(0) if len(known) else pop
        elif name=='PersonalRecency':
            mass=np.zeros(n)
            for age,b in enumerate(blocks[::-1]):
                codes=q['bank_members'][b,:q['bank_size'][b]];known=codes[codes<n]
                mass[known]+=.8**age/len(codes)
            scores[row]=.8*mass/mass.sum()+.2*pop if mass.sum()>0 else pop
        else:raise ValueError(name)
    return scores
