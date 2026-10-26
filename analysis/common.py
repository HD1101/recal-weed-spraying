import csv
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent

def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def csv_write(p, rows):
    rows=list(rows); p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text(''); return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

