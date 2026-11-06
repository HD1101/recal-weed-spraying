"""D005/D006 CPU runner. All writes are additive posthoc artifacts/caches."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'analysis'))
from common import csv_write,sha
from aggregate import cross_seed
from risk_control import RiskConfig,Selection,draw_split,repetition_seed,select_threshold,summarize_runs
from wcr import threshold_grid
from posthoc_20261104.archive import read_gz,dump_new
from posthoc_20261104.core import KS,select_naive_m,draw_nested_grouped,crop_summary,digest_indices,split_overlap
from posthoc_20261104.engine import ArrayDataset,load_records,load_or_build_cache

GRID=threshold_grid(.001)
CFG=RiskConfig(147,.1,.05,132.3)
IDENTITY=('rotation','model','dataset','method','scheme','nozzles','K')


def read(path):return json.loads(Path(path).read_text())


def assert_metrics_equal(actual,stored,context):
    assert (actual is None)==(stored is None),('METRICS_AVAILABILITY',context)
    if actual is None:return
    for key in ('images','images_with_weeds','weed_total','weed_sprayed','wcr_per_weed',
                'weed_image_mean_miss_rate','spray_area_px2','image_area_px2','spray_area_fraction'):
        a,b=actual[key],stored[key]
        assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12)),('ORIGINAL_METRIC_MISMATCH',context,key,a,b)
    assert actual['weed_image_rate_sum_exact']==stored['weed_image_rate_sum_exact'],('EXACT_RISK_SUM_MISMATCH',context)


def check_original_crc_optimum(ds,indices,choice,kp):
    boundary=Fraction('0.1')*(kp+1)
    if choice.mode=='full_spray':
        assert kp==0 or Fraction(1,kp+1)>Fraction('0.1') or ds.rate_sum_at(indices,0)+1>boundary
    else:
        assert choice.mode=='threshold' and kp>0
        assert ds.rate_sum_at(indices,choice.threshold)+1<=boundary
        col=ds.column(choice.threshold)
        if col+1<len(GRID):assert ds.rate_sum_at(indices,GRID[col+1])+1>boundary


def summarize(cells,identity):
    rows=[]
    for cell in cells:
        s=summarize_runs(cell['runs'],.1)
        s={k:v for k,v in s.items() if not isinstance(v,(dict,list))}
        kps=[r['K_prime'] for r in cell['runs'] if r.get('K_prime') is not None]
        images=[r['test_images'] for r in cell['runs']]
        rows.append({**identity,**{k:cell[k] for k in ('method','scheme','nozzles','K')},**s,
            'K_prime_mean':statistics.fmean(kps) if kps else None,'K_prime_min':min(kps) if kps else None,'K_prime_max':max(kps) if kps else None,
            'test_images_mean':statistics.fmean(images),'test_images_min':min(images),'test_images_max':max(images),
            'empty_test_repeats':sum(v==0 for v in images),
            'policy_full_spray_fallback_count_all_draws':sum(r['selection']['fallback'] for r in cell['runs']),
            **crop_summary(cell['runs'])})
    return rows


def prepare_sampling(records,canonical):
    bykey={}
    for split in canonical:
        assert 'error' not in split,('ORIGINAL_SPLIT_ERROR',split)
        s=draw_split(records,split['K'],split['scheme'],random.Random(repetition_seed(20261013,split['scheme'],split['K'],split['repeat'])))
        for key in ('calibration','test','unused_pool','pool_groups'):
            assert list(getattr(s,key))==split[key],('FORMAL_SPLIT_REGENERATION_MISMATCH',split['K'],split['scheme'],split['repeat'],key)
        bykey[(split['scheme'],split['K'],split['repeat'])]=split
    assert len(bykey)==2000
    nested=[draw_nested_grouped(records,r) for r in range(1,201)]
    return bykey,nested


def one_target(config,job,dataset):
    c=read(config);output=Path(c['output_root']);private=Path(c['private_root'])
    verification=output/'INPUT_VERIFICATION_20261104.json'
    assert read(verification)['status']=='ALL_REQUIRED_INPUTS_VERIFIED'
    plan=output/'POSTHOC_PLAN_20261104.md'
    assert sha(plan)==(output/'POSTHOC_PLAN_REVISION_1.sha256').read_text().split()[0]
    out=output/'per_configuration'/job/dataset
    if (out/'POSTHOC_COMPLETE.json').exists():
        completion=read(out/'POSTHOC_COMPLETE.json')
        assert completion['plan_sha256']==sha(plan)
        for name,digest in completion['files'].items():assert sha(out/name)==digest
        print('VERIFIED_POSTHOC_COMPLETE',job,dataset,flush=True);return
    assert not out.exists(),('PARTIAL_POSTHOC_PRESERVED',str(out))
    preds=list((Path(c['predictions_root'])/'runs'/job).glob('attempt*/predictions*/'+dataset+'.json'))
    assert len(preds)==1
    prediction=preds[0];dskey='LB' if dataset in ('bbro','near30','walledgarden') else dataset
    manifest=Path(c['GT_manifests'][dskey]);records,metadata=load_records(prediction,manifest)
    original=read_gz(private/'original_crc'/job/(dataset+'.json.gz'))
    assert original['target_images']==len(records)
    assert original['config']['threshold_grid']==GRID,'FORMAL_GRID_FLOAT_REPRESENTATION_MISMATCH'
    canonical=read_gz(private/'canonical_splits'/(dataset+'.json.gz'))
    splitmap,nested=prepare_sampling(records,canonical)
    started=time.monotonic()
    arrays=load_or_build_cache(records,metadata,prediction,manifest,private/'geometry_formal_grid_v2'/job/(dataset+'.npz'),verification,
                semantics=c['pheno_semantics'] if dataset=='PhenoBench' else None,
                progress=lambda i,total,secs:print('GEOMETRY_PROGRESS',job,dataset,i,total,secs,flush=True))
    d005=[];d006=[];audit=[];checks=Counter()
    for n in (1,2,3,4):
        ds=ArrayDataset(records,arrays,n)
        for oldcell in [v for v in original['cells'] if v['nozzles']==n]:
            scheme,K=oldcell['scheme'],oldcell['K'];crc_runs=[];naive_runs=[]
            for oldrun in oldcell['runs']:
                rep=oldrun['repeat'];s=splitmap[(scheme,K,rep)];cal=s['calibration'];test=s['test']
                kp=sum(ds.counts[i]>0 for i in cal);assert kp==s['K_prime']==oldrun['K_prime']
                crc=Selection(**oldrun['selection'])
                check_original_crc_optimum(ds,cal,crc,kp);checks['original_crc_threshold_optima_verified']+=1
                crcm=ds.metrics(test,crc);assert_metrics_equal(crcm,oldrun['test_metrics'],(job,dataset,n,scheme,K,rep))
                checks['original_crc_draw_metrics_verified']+=1
                naive=select_naive_m(ds,cal,CFG,GRID);nm=ds.metrics(test,naive)
                assert nm['weed_image_mean_miss_rate'] is None or crcm['weed_image_mean_miss_rate'] is None or nm['weed_image_mean_miss_rate']+1e-12>=crcm['weed_image_mean_miss_rate']
                assert nm['spray_area_fraction']<=crcm['spray_area_fraction']+1e-12
                if crc.mode=='threshold':assert naive.mode=='threshold' and naive.threshold>=crc.threshold
                common={'repeat':rep,'seed':s['seed'],'K_prime':kp,'test_images':len(test),
                        'calibration_ids_sha256':digest_indices(records,cal),'test_ids_sha256':digest_indices(records,test)}
                for choice,metric,store in ((crc,crcm,crc_runs),(naive,nm,naive_runs)):
                    store.append({**common,'selection':asdict(choice),'test_metrics':metric,'crop':ds.crop(test,choice)})
            for method,runs in (('crc',crc_runs),('naive_m',naive_runs)):
                d005.append({'method':method,'scheme':scheme,'nozzles':n,'K':K,'runs':runs})
        for K in KS:
            runs=[]
            for draw in nested:
                cal=draw.calibrations[K];test=draw.test;kp=sum(ds.counts[i]>0 for i in cal)
                overlap=split_overlap(records,cal,draw.pools[K],test)
                assert all(v==0 for v in overlap.values())
                choice=select_threshold(ds,cal,'crc',CFG,GRID)
                metric=ds.metrics(test,choice)
                runs.append({'repeat':draw.repeat_one_based-1,'repeat_one_based':draw.repeat_one_based,'seed':draw.seed,
                    'K_prime':kp,'test_images':len(test),'calibration_pool_images':len(draw.pools[K]),
                    'calibration_ids_sha256':digest_indices(records,cal),'calibration_pool_ids_sha256':digest_indices(records,draw.pools[K]),
                    'test_ids_sha256':digest_indices(records,test),'selection':asdict(choice),'test_metrics':metric,'crop':ds.crop(test,choice)})
                if n==1:
                    old=splitmap[('grouped',K,draw.repeat_one_based-1)]
                    old100=splitmap[('grouped',100,draw.repeat_one_based-1)]
                    oa=split_overlap(records,old['calibration'],old['calibration']+old['unused_pool'],old100['test'])
                    audit.append({'K':K,'repeat_one_based':draw.repeat_one_based,'new_seed':draw.seed,
                        'new_pool_images':len(draw.pools[K]),'new_test_images':len(test),**{'new_'+k:v for k,v in overlap.items()},
                        **{'superseded_definition_'+k:v for k,v in oa.items()},
                        'new_calibration_sha256':digest_indices(records,cal),'new_pool_sha256':digest_indices(records,draw.pools[K]),
                        'new_test_sha256':digest_indices(records,test)})
            d006.append({'method':'crc','scheme':'grouped_nested_fixed','nozzles':n,'K':K,'runs':runs})
        print('POSTHOC_NOZZLES_COMPLETE',job,dataset,n,'seconds',round(time.monotonic()-started,1),flush=True)
    rotation,model,seedstr=job.split('__');identity={'rotation':rotation,'model':model,'seed':int(seedstr[4:]),'dataset':dataset}
    summaries5=summarize(d005,identity);summaries6=summarize(d006,identity)
    # Repeated-test means for the original method must also match the immutable CSV.
    with (ROOT/'results/PER_SEED_ALL_RESULTS.csv').open() as f:
        ref={(r['scheme'],int(r['nozzles']),int(r['K'])):r for r in csv.DictReader(f)
             if r['rotation']==rotation and r['model']==model and int(r['seed'])==identity['seed'] and r['dataset']==dataset and r['method']=='crc'}
    for row in [r for r in summaries5 if r['method']=='crc']:
        old=ref[(row['scheme'],row['nozzles'],row['K'])]
        for key in ('mean_actual_weed_image_miss_rate','mean_actual_WCR','mean_spray_area_fraction','full_spray_fallback_count','weed_image_miss_rate_le_alpha_fraction'):
            a,b=row[key],old[key]
            assert (a is None and b=='') or (a is not None and math.isclose(a,float(b),abs_tol=1e-12,rel_tol=1e-12)),('SUMMARY_DIFFERENCE',identity,key,a,b)
    if dataset=='PhenoBench':
        with (ROOT/'results/CROP_EXPOSURE_SUMMARY_PER_SEED.csv').open() as f:
            cropref={(r['scheme'],int(r['nozzles']),int(r['K'])):r for r in csv.DictReader(f)
                if r['rotation']==rotation and r['model']==model and int(r['seed'])==identity['seed'] and r['method']=='crc'}
        for row in [r for r in summaries5 if r['method']=='crc']:
            expected=float(cropref[(row['scheme'],row['nozzles'],row['K'])]['mean_crop_exposure'])
            assert math.isclose(row['mean_crop_exposure'],expected,abs_tol=1e-12,rel_tol=1e-12),('CROP_PRIMARY_MISMATCH',identity)
            checks['original_crop_exposure_summaries_verified']+=1
    out.mkdir(parents=True)
    csv_write(out/'D005_PER_SEED.csv',summaries5);csv_write(out/'D006_PER_SEED.csv',summaries6)
    csv_write(out/'D006_DRAW_INDEX_AUDIT.csv',audit)
    dump_new(out/'D005_DRAWS.json.gz',{'identity':identity,'cells':d005})
    dump_new(out/'D006_DRAWS.json.gz',{'identity':identity,'cells':d006})
    dump_new(out/'VALIDATION.json',{'checks':dict(checks),'all_passed':True,'numeric_abs_rel_tolerance':1e-12,
                                  'runtime_seconds':time.monotonic()-started})
    dump_new(out/'POSTHOC_COMPLETE.json',{'plan_sha256':sha(plan),'prediction_sha256':sha(prediction),
        'input_verification_sha256':sha(verification),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print('POSTHOC_TARGET_COMPLETE',job,dataset,round(time.monotonic()-started,1),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-config',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--job');p.add_argument('--dataset')
    a=p.parse_args();c=read(a.input_config)
    assert (ROOT/'results/posthoc_20261104/D007_ARCHIVE_VERIFICATION.json').is_file()
    jobs=[a.job] if a.job else [f'{r}__{m}__seed{s}' for r in ('bbro','near30','walledgarden') for m in ('asset_detr','rtdetr_r18','yolo26m') for s in (0,1,2)]
    targets=[(job,ds) for job in jobs for ds in ([a.dataset] if a.dataset else ['PhenoBench',job.split('__')[0],'OD_SugarBeets'])]
    if a.workers==1:
        for job,ds in targets:one_target(a.input_config,job,ds)
    else:
        assert 1<a.workers<=4
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(one_target,a.input_config,job,ds):(job,ds) for job,ds in targets}
            for future in as_completed(futures):future.result()
    print('POSTHOC_REQUESTED_TARGETS_COMPLETE',len(targets),flush=True)


if __name__=='__main__':main()
