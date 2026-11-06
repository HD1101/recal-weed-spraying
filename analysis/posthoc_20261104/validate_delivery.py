"""Final additive-artifact, original-file, CSV and private-input verification."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from common import csv_write,sha
from posthoc_20261104.archive import read_gz


def csvrows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def write_manifest(out):
    files=[]
    for folder in (ROOT/'analysis/posthoc_20261104',out):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.name not in ('FILE_MANIFEST.csv','FILE_MANIFEST.sha256'):
                files.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
    csv_write(out/'FILE_MANIFEST.csv',files)
    (out/'FILE_MANIFEST.sha256').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in files))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/posthoc_20261104')
    p.add_argument('--private-provenance',type=Path,required=True);a=p.parse_args();out=a.output
    assert not (out/'FINAL_VERIFICATION.json').exists(),'PRESERVE_PRIOR_FINAL_AUDIT'
    baseline=json.loads((out/'ORIGINAL_PUBLIC_FILES_SHA256.json').read_text())
    original_changes=[name for name,digest in baseline['files'].items() if sha(ROOT/name)!=digest]
    assert not original_changes,('ORIGINAL_FILE_CHANGED',original_changes)
    history=json.loads((out/'POSTHOC_PLAN_HASH_HISTORY.json').read_text())
    assert sha(out/'POSTHOC_PLAN_20261104.md')==history['revision_1_SHA256']
    assert sha(out/'POSTHOC_PLAN_v1_before_revision.md')==history['original_SHA256']
    assert sha(ROOT/'plan/ANALYSIS_PLAN_FROZEN_20261016.md')=='e80795012ce4d4b492a0152a5e89acebe8246703c2fd579f7b4b98f64a7e81f9'
    archival=json.loads((out/'D007_ARCHIVE_VERIFICATION.json').read_text())
    assert sha(ROOT/archival['source_archive'])==archival['archive_sha256_from_existing_record'],'ORIGINAL_RAW_BACKUP_HASH'
    lineage=json.loads((out/'PREDICTION_RELEASE_LINEAGE_VERIFICATION.json').read_text())
    assert lineage['status']=='VERIFIED' and lineage['original_to_release_prediction_files_checked']==108
    private=json.loads(a.private_provenance.read_text())
    for r in private['verified_files']:
        assert sha(Path(r['local']))==r['sha256'],('PRIVATE_INPUT_CHANGED_DURING_ANALYSIS',r['source'])
    expected={'D005_PER_SEED_ALL_RESULTS.csv':6480,'D005_CROSS_SEED_ALL_RESULTS.csv':2160,
              'D005_CRC_VS_NAIVE_M_CROSS_SEED.csv':1080,'D006_PER_SEED_ALL_RESULTS.csv':1620,
              'D006_CROSS_SEED_ALL_RESULTS.csv':540,'D007_MC_PER_SEED.csv':243,
              'D007_MC_CROSS_SEED_WITH_FLAGS.csv':81}
    rowcounts={}
    for name,count in expected.items():
        rs=csvrows(out/name);assert len(rs)==count,(name,len(rs),count);rowcounts[name]=len(rs)
        for r in rs:
            if 'n_seeds' in r:assert int(r['n_seeds'])==3
    markers=list(out.glob('per_configuration/*/*/POSTHOC_COMPLETE.json'))
    assert len(markers)==81
    verified_draw_pairs=0;thresholds_checked=0;mc_draw_rows=[];mc_recomputed={}
    for marker in markers:
        complete=json.loads(marker.read_text())
        assert complete['plan_sha256']==history['revision_1_SHA256']
        for name,digest in complete['files'].items():assert sha(marker.parent/name)==digest
        obj=read_gz(marker.parent/'D005_DRAWS.json.gz')
        cells={(c['scheme'],c['nozzles'],c['K'],c['method']):c for c in obj['cells']}
        for (scheme,n,K,method),cell in cells.items():
            if method!='crc':continue
            other=cells[(scheme,n,K,'naive_m')]
            for cr,nm in zip(cell['runs'],other['runs']):
                for field in ('repeat','seed','K_prime','test_images','calibration_ids_sha256','test_ids_sha256'):
                    assert cr[field]==nm[field],('PAIRING_CHANGED',marker,field)
                verified_draw_pairs+=1
                assert int(nm['selection']['fallback'])>=0
                if cr['selection']['mode']=='threshold':
                    assert nm['selection']['mode']=='threshold' and nm['selection']['threshold']>=cr['selection']['threshold']
                if cr.get('crop') and nm.get('crop'):
                    assert cr['crop']['crop_pixels']==nm['crop']['crop_pixels']
                    assert nm['crop']['sprayed_crop_pixels']<=cr['crop']['sprayed_crop_pixels'],('NONMONOTONE_PAIRED_CROP_EXPOSURE',marker)
                thresholds_checked+=1
                if n==4 and ((scheme=='grouped' and K in (20,50)) or (scheme=='ideal' and K==50)):
                    ident=obj['identity'];m=cr['test_metrics']
                    mc_draw_rows.append({**ident,'scheme':scheme,'nozzles':n,'K':K,'repeat':cr['repeat'],
                        'M':m['weed_image_mean_miss_rate'],'WCR':m['wcr_per_weed'],'A':m['spray_area_fraction'],
                        'crop_exposure':cr['crop']['crop_exposure'] if cr.get('crop') else None,
                        'K_prime':cr['K_prime'],'test_images':cr['test_images'],
                        'full_spray_fallback':cr['selection']['fallback'],'threshold':cr['selection']['threshold']})
            if n==4 and ((scheme=='grouped' and K in (20,50)) or (scheme=='ideal' and K==50)):
                ident=obj['identity'];key=tuple(str(ident[k]) for k in ('rotation','model','seed','dataset'))+(scheme,str(K))
                vals=[r['test_metrics']['weed_image_mean_miss_rate'] for r in cell['runs']]
                assert all(v is not None for v in vals)
                mc_recomputed[key]=(statistics.fmean(vals),statistics.stdev(vals)/math.sqrt(200))
        for r in csvrows(marker.parent/'D006_DRAW_INDEX_AUDIT.csv'):
            for key in ('new_calibration_image_overlap','new_calibration_pool_image_overlap','new_calibration_group_overlap','new_calibration_pool_group_overlap'):
                assert int(r[key])==0,('NEW_TEST_CALIBRATION_OVERLAP',marker,key)
    assert verified_draw_pairs==648000
    mc=csvrows(out/'D007_MC_PER_SEED.csv')
    assert all(int(r['valid_M_repeats'])==200 and int(r['MC_SE_denominator'])==200 for r in mc)
    assert all(math.isclose(float(r['M_MC_standard_error']),float(r['M_sample_SD'])/math.sqrt(200),abs_tol=1e-15) for r in mc)
    for row in mc:
        key=tuple(row[k] for k in ('rotation','model','seed','dataset','scheme','K'))
        mean,se=mc_recomputed[key]
        assert math.isclose(mean,float(row['mean_actual_weed_image_miss_rate']),abs_tol=1e-15)
        assert math.isclose(se,float(row['M_MC_standard_error']),abs_tol=1e-15)
    assert len(mc_draw_rows)==48600
    csv_write(out/'D007_MC_RUN_VALUES.csv',mc_draw_rows)
    tests=(out/'TESTS_FINAL.log').read_text()
    assert re.search(r'Ran 89 tests\b',tests) and re.search(r'\nOK\s*$',tests),'FINAL_TEST_SUITE_NOT_PASSED'
    patterns={
       'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
       'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
       'assigned_password':r'(?i)\b(?:password|passwd|api_key|access_token)\s*[:=]\s*[\x22\x27][^\x22\x27\n]{5,}[\x22\x27]',
       'IPv4':r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',
    }
    findings=[];scanned=0
    paths=list((ROOT/'analysis/posthoc_20261104').rglob('*'))+list(out.rglob('*'))
    for path in paths:
        if path.is_file() and path.suffix in ('.py','.md','.json','.csv','.log','.sha256'):
            text=path.read_text();scanned+=1
            for name,pattern in patterns.items():
                if re.search(pattern,text):findings.append({'path':str(path.relative_to(ROOT)),'kind':name})
    assert not findings,('SENSITIVE_INFORMATION_SCAN',findings)
    result={'status':'PASSED','verified_utc':datetime.now(timezone.utc).isoformat(),
            'original_public_files_unchanged':len(baseline['files']),'original_files_changed':[],
            'initial_posthoc_plan_sha256':history['original_SHA256'],'revised_posthoc_plan_sha256':history['revision_1_SHA256'],
            'frozen_plan_unchanged':True,'private_input_files_rechecked':len(private['verified_files']),
            'original_raw_backup_archive_sha256_verified':True,
            'all_108_predictions_bound_to_original_release_lineage':True,
            'completed_configuration_targets':81,'CSV_row_counts':rowcounts,
            'CRC_vs_naive_M_exact_draw_pairs_checked':verified_draw_pairs,
            'D006_new_sampling_zero_overlap':True,'MC_all_243_rows_have_200_valid_draws':True,
            'MC_raw_draw_rows_exported_and_rechecked':48600,
            'tests_passed':89,'original_tests':70,'new_tests':19,
            'privacy_scan_files':scanned,'credential_or_IP_findings':findings,
            'privacy_scan_scope':'New text code/document/CSV/log/hash artifacts; compressed draw pairing and index separation are verified separately',
            'results_are_posthoc_deviations':['D005','D006','D007'],'pushed_to_GitHub':False,'Zenodo_modified':False}
    (out/'FINAL_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    # Flush the redirected verification log before hashing it.
    write_manifest(out)


if __name__=='__main__':main()
