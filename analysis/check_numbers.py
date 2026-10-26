"""Independent reaggregation of 600 archived draw metrics, never rounded inputs."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def check():
    fixture=rows(ROOT/'results/BBRO_ASSET_CRC_N4_K50_RUN_METRICS.csv')
    assert len(fixture)==600
    buckets=defaultdict(list)
    for row in fixture:buckets[int(row['seed'])].append(row)
    assert sorted(buckets)==[0,1,2]
    assert all(sorted(int(r['repeat']) for r in rs)==list(range(200)) for rs in buckets.values())
    main=[r for r in rows(ROOT/'results/MAIN_TABLE_N4_GROUPED_K20_K50.csv')
          if all(r[k]==v for k,v in {'rotation':'bbro','model':'asset_detr','dataset':'bbro','method':'crc',
                                   'scheme':'grouped','nozzles':'4','K':'50'}.items())]
    assert len(main)==1;checks=[]
    for metric in ('mean_actual_weed_image_miss_rate','mean_actual_WCR','mean_spray_area_fraction'):
        seed_means=[statistics.fmean(float(r[metric]) for r in buckets[s]) for s in [0,1,2]]
        for suffix,fn in [('mean',statistics.fmean),('min',min),('max',max)]:
            recomputed=fn(seed_means);stored=float(main[0][metric+'_seed_'+suffix])
            assert math.isclose(recomputed,stored,rel_tol=0,abs_tol=1e-12),(metric,suffix)
            checks.append({'metric':metric,'seed_stat':suffix,'recomputed_fraction':recomputed,
                           'stored_fraction':stored,'absolute_difference':abs(recomputed-stored),
                           'percent':100*recomputed,'pass':True})
    assert format(checks[0]['percent'],'.1f')=='7.9'
    return {'status':'PASSED','scope':'bbro/ASSET-DETR/bbro/Main CRC/n=4/K=50/grouped',
            'draws_per_seed':200,'seeds':[0,1,2],'metrics':3,'comparisons':checks,
            'independence':'Recomputed equal draw means within each seed, then equal seed mean/range; no original CSV was modified.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=check()
    if a.output:
        assert not a.output.exists(),'EXISTING_CHECK_PRESERVED'
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
