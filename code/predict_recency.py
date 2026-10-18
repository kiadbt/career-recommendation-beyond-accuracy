"""Frozen personal-recency baseline, using fitting-only popularity counts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from quarter_constructor import count_scores
ap=argparse.ArgumentParser(description=__doc__)
for name in ['queries','counts','out']:ap.add_argument('--'+name,required=True,type=Path)
args=ap.parse_args();assert not args.out.exists();q=dict(np.load(args.queries,allow_pickle=False));c=np.load(args.counts,allow_pickle=False);rows=[]
for start in range(0,len(q['pid']),128):
 ix=np.arange(start,min(start+128,len(q['pid'])));scores=count_scores(q,ix,'PersonalRecency',c['popularity'],c['transition']);rows.append(np.argsort(-scores,axis=1,kind='stable')[:,:10])
np.savez_compressed(args.out,pid=q['pid'],top_PersonalRecency=np.concatenate(rows))
