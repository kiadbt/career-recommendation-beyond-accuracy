"""Download checksum-pinned public source files; never bypass access conditions."""
import argparse,json,hashlib,urllib.request,shutil
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--accept-source-terms',action='store_true',help='Confirm you have reviewed and may use these upstream assets')
    p.add_argument('--only',nargs='*');a=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[1]/'SOURCE_ASSETS.json').read_text())
    if not a.accept_source_terms:raise SystemExit('Review SOURCE_ASSETS.json and upstream conditions, then pass --accept-source-terms.')
    for row in manifest['files']:
        if a.only and row['name'] not in a.only:continue
        target=a.output_dir/row['name'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=row['sha256']:raise ValueError('Existing asset checksum mismatch: '+row['name'])
            print('Verified cached',row['name']);continue
        temporary=target.with_suffix(target.suffix+'.partial')
        try:
            with urllib.request.urlopen(row['url'],timeout=60) as response,temporary.open('wb') as out:shutil.copyfileobj(response,out)
            if sha(temporary)!=row['sha256']:raise ValueError('Downloaded checksum mismatch: '+row['name'])
            temporary.replace(target);print('Downloaded and verified',row['name'],flush=True)
        except Exception:
            print('Obtain this exact revision from the owner if access requires credentials:',row['url']);raise

if __name__=='__main__':main()
