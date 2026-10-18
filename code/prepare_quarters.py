"""Construct final-quarter queries from locally obtained JobHop or Decorte files."""
import argparse,csv,json,re,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from quarter_constructor import build
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dataset',choices=['jobhop','decorte'],required=True)
 for arg in ['source','catalogue','embeddings','out']:ap.add_argument('--'+arg,type=Path,required=True)
 args=ap.parse_args();assert not args.out.exists()
 cat=json.loads(args.catalogue.read_text(encoding='utf-8'));emb=np.load(args.embeddings,allow_pickle=False)['history'];assert len(cat)==3007 and emb.shape==(3007,384)
 if args.dataset=='jobhop':df=pd.read_parquet(args.source)
 else:
  uri={r['uri']:r['code'] for r in cat};rows=[]
  def date(s):
   m=re.fullmatch(r'(\d{2})/(\d{4})',s or '')
   return f'Q{(int(m[1])-1)//3+1} {m[2]}' if m and 1<=int(m[1])<=12 else None
  with args.source.open(encoding='utf-8-sig',newline='') as handle:
   for row in csv.DictReader(handle):
    n=int(row['number_of_experiences']);assert 1<=n<=17
    for j in range(n):
     assert row[f'ESCO_uri_{j}'] in uri
     rows.append(dict(resume_id=int(row['identifier']),matched_code=uri[row[f'ESCO_uri_{j}']],start_date=date(row[f'start_{j}']),end_date=date(row[f'end_{j}'])))
  df=pd.DataFrame(rows)
 q,support=build(df,cat,emb);assert not support['out_of_catalogue_raw_codes'],'Unexpected catalogue mismatch'
 np.savez_compressed(args.out,**q);support['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
 args.out.with_suffix('.support.json').write_text(json.dumps(support,indent=2));print(json.dumps(support))
if __name__=='__main__':main()
