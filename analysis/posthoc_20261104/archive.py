"""Read the original raw backup once; retain compact CRC evidence privately.

Every selected tar member is checked against the pre-existing backup manifest.
This runs after the posthoc plan has been saved and hashed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
from aggregate import cross_seed
from common import csv_write, sha
from posthoc_20261104.core import mc_summary


def dump_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    if path.exists():
        raise FileExistsError(path)
    if path.suffix == '.gz':
        with path.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as out:
                out.write(payload)
    else:
        path.write_bytes(payload+b'\n')


def read_gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def extract_crc(archive, manifest, private, public):
    assert (public/'POSTHOC_PLAN_REVISION_1.sha256').is_file(), 'PLAN_REQUIRED_BEFORE_ANALYSIS'
    expected = {r['archive_path']:r for r in json.loads(manifest.read_text())['files']}
    per_seed, provenance, canonical_splits, checks = [], [], {}, []
    original_summary = {}
    with (ROOT/'results/PER_SEED_ALL_RESULTS.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['method'] == 'crc':
                key=tuple(row[k] for k in ('rotation','model','seed','dataset','scheme','nozzles','K'))
                original_summary[key]=row
    started=time.monotonic()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name=member.name.removeprefix('./')
            if not name.endswith('/FULL_RESULTS.json') or '/risk01/' not in name:
                continue
            payload=tar.extractfile(member).read()
            digest=hashlib.sha256(payload).hexdigest()
            assert digest==expected[name]['sha256'], ('ARCHIVE_MEMBER_HASH',name)
            obj=json.loads(payload);del payload
            parts=Path(name).parts;job=parts[1];rotation,model,seedstr=job.split('__');seed=int(seedstr[4:])
            dataset=parts[-2]
            split_payload=json.dumps(obj['splits'],sort_keys=True,separators=(',',':')).encode()
            split_hash=hashlib.sha256(split_payload).hexdigest()
            if dataset not in canonical_splits:
                canonical_splits[dataset]=split_hash
                dump_new(private/'canonical_splits'/(dataset+'.json.gz'),obj['splits'])
            else:
                assert split_hash==canonical_splits[dataset],('FORMAL_SPLITS_DIFFER_ACROSS_CONFIGS',job,dataset)
            cells=[]
            for cell in obj['results']:
                if cell['method']!='crc':continue
                runs=cell['runs']
                key=tuple(map(str,(rotation,model,seed,dataset,cell['scheme'],cell['nozzles'],cell['K'])))
                row=original_summary[key]
                vals=[r['test_metrics']['weed_image_mean_miss_rate'] if r['test_metrics'] else None for r in runs]
                checked=mc_summary(vals)
                for field,computed in [('mean_actual_weed_image_miss_rate',checked['mean_actual_weed_image_miss_rate']),
                                       ('weed_image_miss_rate_MC_standard_error',checked['M_MC_standard_error'])]:
                    stored=row[field]
                    assert (computed is None and stored=='') or (computed is not None and math.isclose(computed,float(stored),abs_tol=1e-12,rel_tol=1e-12)), ('ORIGINAL_CRC_SUMMARY_MISMATCH',key,field)
                checks.append({'job':job,'dataset':dataset,'scheme':cell['scheme'],'nozzles':cell['nozzles'],'K':cell['K'],'mean_and_SE_match_original':True})
                cells.append({k:cell[k] for k in ('method','K','scheme','nozzles','summary','runs')})
                if cell['nozzles']==4 and ((cell['scheme']=='grouped' and cell['K'] in (20,50)) or (cell['scheme']=='ideal' and cell['K']==50)):
                    per_seed.append({'rotation':rotation,'model':model,'seed':seed,'dataset':dataset,
                                     'method':'crc','scheme':cell['scheme'],'nozzles':4,'K':cell['K'],**checked})
            dump_new(private/'original_crc'/job/(dataset+'.json.gz'),{'job':job,'dataset':dataset,'config':obj['config'],
                'target_images':obj['target_images'],'source_member':name,'source_member_sha256':digest,
                'canonical_splits_sha256':split_hash,'cells':cells})
            provenance.append({'member':name,'bytes':member.size,'sha256':digest,'verified':True})
            del obj,cells
            print('ARCHIVE_CRC_EXTRACTED',len(provenance),job,dataset,'seconds',round(time.monotonic()-started,1),flush=True)
    assert len(provenance)==81, ('EXPECTED_81_TARGET_RESULTS',len(provenance))
    assert len(per_seed)==243
    csv_write(public/'D007_MC_PER_SEED.csv',per_seed)
    cross=cross_seed(per_seed,('rotation','model','dataset','method','scheme','nozzles','K'))
    csv_write(public/'D007_MC_CROSS_SEED.csv',cross)
    csv_write(public/'D007_MC_BOUNDARY_FLAGS.csv',[r for r in per_seed if r['mean_within_alpha_plus_minus_2SE']])
    dump_new(public/'D007_ARCHIVE_VERIFICATION.json',{'source_archive':str(archive),'archive_sha256_from_existing_record':
        (archive.parent/(archive.name+'.sha256')).read_text().split()[0],
        'selected_member_hashes_verified':provenance,'CRC_mean_and_SE_checks':len(checks),
        'all_checks_passed':True,'tolerance_absolute_and_relative':1e-12,
        'canonical_split_sha256_by_dataset':canonical_splits,'per_seed_rows':len(per_seed),'cross_seed_rows':len(cross)})
    csv_write(public/'D007_ORIGINAL_CRC_CHECKS.csv',checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/posthoc_20261104')
    a=p.parse_args();extract_crc(a.archive,a.manifest,a.private,a.output)


if __name__=='__main__':main()
