import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,zipfile,csv,io,unicodedata,collections
import numpy as np
import argparse
ap=argparse.ArgumentParser(description='Build KARRIEREWEGE queries from local assets; no download.')
for name in ['archive','catalogue','embeddings','jobhop-fitting-queries','decorte-queries','output-dir']:ap.add_argument('--'+name,required=True,type=Path)
args=ap.parse_args();H=args.output_dir;H.mkdir(parents=True,exist_ok=True)
assert not (H/'queries.npz').exists(), 'Do not overwrite queries'

cat=json.loads((args.catalogue).read_text(encoding='utf-8'))
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
mapping=collections.defaultdict(list)
for i,r in enumerate(cat):mapping[norm(r['history'].split(' \n description:')[0].removeprefix('role: '))].append(i)
histories=collections.defaultdict(list);unmapped=collections.Counter();totalrows=0
with zipfile.ZipFile(args.archive) as z:
    names=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX')];assert len(names)==1,names
    with z.open(names[0]) as f:
        reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig'));columns=reader.fieldnames
        assert {'_id','experience_order','preferredLabel_en'}<=set(columns),columns
        for r in reader:
            totalrows+=1;label=norm(r['preferredLabel_en']);v=mapping.get(label,[]);code=v[0] if len(v)==1 else -1
            if code<0:unmapped[label]+=1
            try:order=int(r['experience_order'])
            except ValueError:order=-1
            histories[int(r['_id'])].append((order,code))
pid=[];seqs=[];excluded=collections.Counter();mapping_failed=0
for identifier,rows in sorted(histories.items()):
    if any(c<0 for _,c in rows):excluded['unmapped_or_ambiguous_label']+=1;mapping_failed+=1;continue
    rows.sort();orders=[a for a,_ in rows]
    if orders!=list(range(len(rows))):excluded['nonconsecutive_or_duplicate_order']+=1;continue
    if len(rows)<2:excluded['short_history']+=1;continue
    pid.append(identifier);seqs.append(tuple((c,) for _,c in rows))
def signature(q,i):
    blocks=list(q['history_blocks'][i,:q['length'][i]])+[q['target_block'][i]]
    return tuple(tuple(sorted(map(int,q['bank_members'][b,:q['bank_size'][b]]))) for b in blocks)
train=dict(np.load(args.jobhop_fitting_queries));last={}
for i in train['fit']:
    key=int(train['pid'][i])
    if key not in last or train['length'][i]>train['length'][last[key]]:last[key]=int(i)
train_sigs={signature(train,i) for i in last.values()};del train
dec=dict(np.load(args.decorte_queries));dec_sigs={signature(dec,i) for i in range(len(dec['pid']))};del dec
seen=set();sensitivity=[];trainmatch=decmatch=0
for i,sig in enumerate(seqs):
    trainmatch+=sig in train_sigs;decmatch+=sig in dec_sigs
    if sig not in seen and sig not in train_sigs and sig not in dec_sigs:sensitivity.append(i)
    seen.add(sig)
length=np.array([len(s)-1 for s in seqs],np.int64);assert len(length)>0
hist=np.zeros((len(seqs),int(length.max())),np.int64)
for i,s in enumerate(seqs):hist[i,:length[i]]=[x[0] for x in s[:-1]]
e=np.load(args.embeddings)['history'];bank=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-12)
np.savez_compressed(H/'queries.npz',pid=np.array(pid),history_blocks=hist,length=length,target_block=np.array([s[-1][0] for s in seqs]),bank_members=np.arange(len(cat))[:,None],bank_size=np.ones(len(cat),np.int64),bank_history_embeddings=bank.astype(np.float32),sensitivity_indices=np.array(sensitivity,np.int64))
report={'source_rows':totalrows,'source_histories':len(histories),'eligible_histories':len(seqs),'excluded':dict(excluded),'whole_history_mapping_coverage':1-mapping_failed/len(histories),'row_mapping_coverage':1-sum(unmapped.values())/totalrows,'unmapped_unique_labels':len(unmapped),'columns':columns,'unique_complete_sequences':len(seen),'duplicate_sequence_histories':len(seqs)-len(seen),'exact_JobHop_fit_sequence_matches':trainmatch,'exact_Decorte_test_sequence_matches':decmatch,'sensitivity_histories':len(sensitivity),'max_history_length':int(length.max()),'task':'ordered experience sequence with singleton target; dates unavailable','broad_validation_admitted':len(seqs)>=5000 and 1-mapping_failed/len(histories)>=.9,'queries_sha256':hashlib.sha256((H/'queries.npz').read_bytes()).hexdigest()}
(H/'support.json').write_text(json.dumps(report,indent=2));(H/'unmapped_labels.json').write_text(json.dumps(unmapped,indent=2));print(json.dumps(report,indent=2))
