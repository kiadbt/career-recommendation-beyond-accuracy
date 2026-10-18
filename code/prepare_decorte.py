"""Construct Decorte held-out queries from the owner's original test CSV."""
import argparse,csv,re,json
from pathlib import Path
import numpy as np
import pandas as pd
from quarter_constructor import build

def main():
    p=argparse.ArgumentParser()
    for k in ['csv','catalogue','embeddings','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cat=json.loads(a.catalogue.read_text(encoding='utf8'));uri={r['uri']:r['code'] for r in cat};events=[]
    def date(s):
        m=re.fullmatch(r'(\d{2})/(\d{4})',s or '')
        return f'Q{(int(m[1])-1)//3+1} {m[2]}' if m and 1<=int(m[1])<=12 else None
    with a.csv.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            n=int(row['number_of_experiences']);assert 1<=n<=17
            for j in range(n):events.append(dict(resume_id=int(row['identifier']),start_date=date(row[f'start_{j}']),end_date=date(row[f'end_{j}']),matched_code=uri[row[f'ESCO_uri_{j}']]))
    q,s=build(pd.DataFrame(events),cat,np.load(a.embeddings)['history']);np.savez(a.output,**q)
    a.output.with_suffix('.json').write_text(json.dumps(s,indent=2));print(json.dumps(s))

if __name__=='__main__':main()
