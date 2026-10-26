"""Rebuild all paper tables from completed CPU replays, into a new directory."""
import argparse
import csv
from pathlib import Path
from common import read,save,csv_write,sha
from aggregate import cross_seed


def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def build(root,out):
    root,out=Path(root),Path(out)
    assert not out.exists(),'EXISTING_SUMMARY_PRESERVED'
    allrows=[];whole=[];aux=[]
    extras={k:[] for k in ('RQ3_SUMMARY','PHENO_DATE_DIRECTIONS','CROP_EXPOSURE_SUMMARY')}
    for rotation in ('bbro','near30','walledgarden'):
        for model in ('asset_detr','rtdetr_r18','yolo26m'):
            for seed in (0,1,2):
                job=f'{rotation}__{model}__seed{seed}';d=root/job
                complete=read(d/'REPLAY_COMPLETE.json')
                for name,h in complete['files'].items():assert sha(d/name)==h
                prefix={'rotation':rotation,'model':model,'seed':seed}
                allrows.extend({**prefix,**r} for r in csv_rows(d/'ALL_SUMMARIES.csv'))
                aux.extend(csv_rows(d/'MIN_THRESHOLD.csv'))
                for ds in (rotation,'PhenoBench','OD_SugarBeets'):
                    whole.extend({**prefix,**r} for r in csv_rows(d/ds/'WHOLE_DATASET_SOURCE_AND_GT_REFERENCE.csv'))
                    for key in extras:
                        f=d/ds/(key+'.csv')
                        if f.exists() and f.stat().st_size:extras[key].extend({**prefix,'dataset':ds,**r} for r in csv_rows(f))
    out.mkdir(parents=True)
    ids=['rotation','model','dataset','method','scheme','nozzles','K']
    cross=cross_seed(allrows,ids)
    csv_write(out/'PER_SEED_ALL_RESULTS.csv',allrows);csv_write(out/'CROSS_SEED_ALL_RESULTS.csv',cross)
    csv_write(out/'MAIN_TABLE_N4_GROUPED_K20_K50.csv',[r for r in cross if r['rotation'] in ('bbro','near30') and str(r['nozzles'])=='4' and str(r['K']) in ('20','50') and r['scheme']=='grouped'])
    csv_write(out/'SUPPLEMENT_WALLEDGARDEN_N4_GROUPED_K20_K50.csv',[r for r in cross if r['rotation']=='walledgarden' and str(r['nozzles'])=='4' and str(r['K']) in ('20','50') and r['scheme']=='grouped'])
    csv_write(out/'SUPPLEMENT_N2_GROUPED_K20_K50.csv',[r for r in cross if str(r['nozzles'])=='2' and str(r['K']) in ('20','50') and r['scheme']=='grouped'])
    wc=cross_seed(whole,['rotation','model','dataset','method','nozzles'])
    csv_write(out/'RQ1_WHOLE_TARGET_PER_SEED.csv',whole);csv_write(out/'RQ1_WHOLE_TARGET_CROSS_SEED.csv',wc)
    csv_write(out/'MAIN_RQ1_WHOLE_TARGET_N4.csv',[r for r in wc if r['rotation'] in ('bbro','near30') and str(r['nozzles'])=='4'])
    keys={'RQ3_SUMMARY':['rotation','model','dataset','nozzles','K','relation'],
          'PHENO_DATE_DIRECTIONS':['rotation','model','dataset','nozzles','K','table','calibration_dates','test_dates'],
          'CROP_EXPOSURE_SUMMARY':ids}
    for key,rs in extras.items():
        csv_write(out/(key+'_PER_SEED.csv'),rs);csv_write(out/(key+'_CROSS_SEED.csv'),cross_seed(rs,keys[key]))
    dates=csv_rows(out/'PHENO_DATE_DIRECTIONS_CROSS_SEED.csv')
    csv_write(out/'PHENO_DATE_DIRECTIONS_BBRO_N4_K50_GROUPED_CROSS_SEED.csv',
              [r for r in dates if r['rotation']=='bbro' and r['nozzles']=='4' and r['K']=='50'])
    ac=cross_seed(aux,['rotation','model','dataset','nozzles'])
    for name,rs in [('MIN_THRESHOLD_PER_SEED_ALL_N.csv',aux),('MIN_THRESHOLD_CROSS_SEED_ALL_N.csv',ac),
                    ('MIN_THRESHOLD_PER_SEED_N4.csv',[r for r in aux if str(r['nozzles'])=='4']),
                    ('MIN_THRESHOLD_CROSS_SEED_N4.csv',[r for r in ac if str(r['nozzles'])=='4'])]:csv_write(out/name,rs)
    save(out/'REPRODUCED_SUMMARY.json',{'completed_configs':27,'cross_seed_rows':len(cross),
       'note':'All checked-in result files were read-only; output is a new directory.'})
    print('SUMMARY_COMPLETE',len(cross),'cross-seed rows',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replays',type=Path,default=Path('reproduced'))
    p.add_argument('--output',type=Path,default=Path('reproduced-summary'));a=p.parse_args();build(a.replays,a.output)
