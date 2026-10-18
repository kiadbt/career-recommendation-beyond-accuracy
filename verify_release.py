"""Check stored chronological summaries without private data or training."""
from pathlib import Path
import csv, json, math, statistics, hashlib
B=Path(__file__).resolve().parent
for name,digest in json.loads((B/'MANIFEST.json').read_text())['sha256'].items():
    assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
r=json.loads((B/'chronological_results/results.json').read_text())
rows=[]
for sample,a in r['analyses'].items():
    assert set(a['seed_contrasts'])=={'42','43','44','45','46'}
    for key,c in a['contrasts'].items():
        values=[s[key] for s in a['seed_contrasts'].values()]
        assert math.isclose(statistics.mean(values),c['estimate'],abs_tol=1e-12)
        assert math.isclose(statistics.stdev(values),c['seed_SD'],abs_tol=1e-12)
        rows.append([sample,key,c['estimate'],c['seed_SD'],json.dumps(c)])
out=B/'generated_tables';out.mkdir(exist_ok=True)
with (out/'Chronological_contrasts.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.writer(f);w.writerow(['sample','contrast','estimate','seed_SD','stored_summary_json']);w.writerows(rows)
record={'file_hashes_passed':True,'chronological_seed_means_and_SD_passed':True,'chronological_contrasts':len(rows),'scope':'Checks stored aggregates; does not retrain models or reconstruct intervals from histories.'}
(B/'RELEASE_VERIFICATION.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
