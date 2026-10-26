"""Compare complete PhenoBench replay cells against read-only formal results."""
import argparse
import csv
import json
import math
from pathlib import Path
import tarfile


TABLES=[
 ('PER_SEED_ALL_RESULTS.csv','PhenoBench/SUMMARY.csv',['method','scheme','nozzles','K']),
 ('RQ1_WHOLE_TARGET_PER_SEED.csv','PhenoBench/WHOLE_DATASET_SOURCE_AND_GT_REFERENCE.csv',['method','nozzles']),
 ('RQ3_SUMMARY_PER_SEED.csv','PhenoBench/RQ3_SUMMARY.csv',['nozzles','K','relation']),
 ('PHENO_DATE_DIRECTIONS_PER_SEED.csv','PhenoBench/PHENO_DATE_DIRECTIONS.csv',['nozzles','K','table','calibration_dates','test_dates']),
 ('CROP_EXPOSURE_SUMMARY_PER_SEED.csv','PhenoBench/CROP_EXPOSURE_SUMMARY.csv',['nozzles','K','scheme','method']),
 ('MIN_THRESHOLD_PER_SEED_ALL_N.csv','MIN_THRESHOLD.csv',['nozzles'])]
ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def compare_cells(expected,actual,stats,path='$'):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(expected)==set(actual),('KEYS',path)
        for k,v in expected.items():compare_cells(v,actual[k],stats,path+'.'+str(k))
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(expected)==len(actual),('LENGTH',path)
        for i,(a,b) in enumerate(zip(expected,actual)):compare_cells(a,b,stats,path+f'[{i}]')
    else:
        stats['leaf_values']+=1
        if isinstance(expected,bool) or expected is None or isinstance(expected,str):
            assert expected==actual,('VALUE',path,expected,actual)
        elif isinstance(expected,(int,float)):
            assert isinstance(actual,(int,float)) and not isinstance(actual,bool),('TYPE',path)
            difference=abs(expected-actual)
            assert math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12),('NUMERIC',path,expected,actual)
            stats['numeric_values']+=1;stats['max_absolute_numeric_difference']=max(stats['max_absolute_numeric_difference'],difference)
            stats['nonidentical_numeric_values']+=int(expected!=actual)
            if expected!=actual:
                field=path.rsplit('.',1)[-1]
                counts=stats.setdefault('roundoff_field_counts',{});counts[field]=counts.get(field,0)+1
                maxima=stats.setdefault('roundoff_field_max_absolute_difference',{})
                maxima[field]=max(maxima.get(field,0),difference)
        else:raise TypeError((path,type(expected)))


def verify(replays,reference,original_backup=None):
    reports=[]
    for model in ('asset_detr','yolo26m'):
        job=f'bbro__{model}__seed0';folder=Path(replays)/job
        complete=json.loads((folder/'REPLAY_COMPLETE.json').read_text())
        assert complete['datasets']==['PhenoBench']
        for name,digest in complete['files'].items():
            import hashlib
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
        for oldname,newname,keys in TABLES:
            expected=[r for r in rows(Path(reference)/oldname) if r['rotation']=='bbro' and r['model']==model
                      and r['seed']=='0' and r['dataset']=='PhenoBench']
            actual=[{'rotation':'bbro','model':model,'seed':'0','dataset':'PhenoBench',**r} for r in rows(folder/newname)]
            key=lambda r:tuple(r[k] for k in keys)
            e={key(r):r for r in expected};a={key(r):r for r in actual}
            assert len(e)==len(expected) and len(a)==len(actual) and set(e)==set(a),(job,oldname,'ROWS')
            stats={'leaf_values':0,'numeric_values':0,'max_absolute_numeric_difference':0.,'nonidentical_numeric_values':0}
            padded_columns=set()
            for identity,row in e.items():
                # Global CSVs union columns across datasets/rotations. A local
                # cell can omit a column that is blank for EVERY compared row.
                missing=set(row)-set(a[identity]);extra=set(a[identity])-set(row)
                assert all(row[k]=='' for k in missing),(job,oldname,'NONEMPTY_MISSING_COLUMN')
                assert all(a[identity][k]=='' for k in extra),(job,oldname,'NONEMPTY_EXTRA_COLUMN')
                padded_columns.update(missing)
                for field,value in row.items():
                    other=a[identity].get(field,'');stats['leaf_values']+=1
                    if value==other:continue
                    try:
                        av,bv=float(value),float(other)
                        assert math.isclose(av,bv,rel_tol=1e-12,abs_tol=1e-12)
                    except (ValueError,TypeError,AssertionError):raise AssertionError((job,oldname,identity,field,value,other))
                    stats['nonidentical_numeric_values']+=1
                    stats['max_absolute_numeric_difference']=max(stats['max_absolute_numeric_difference'],abs(av-bv))
            reports.append({'job':job,'table':oldname,'rows':len(expected),'status':'PASSED',
                            'global_union_blank_columns_padded_for_comparison_only':sorted(padded_columns),**stats})
        print('REPLAY_TABLES_MATCH',job,flush=True)
    full=[]
    if original_backup:
        wanted={f'runs/bbro__{model}__seed0/attempt01/risk01/PhenoBench/FULL_RESULTS.json':model for model in ('asset_detr','yolo26m')}
        with tarfile.open(original_backup,'r|gz') as tar:
            for member in tar:
                if member.name not in wanted:continue
                model=wanted.pop(member.name)
                old=json.load(tar.extractfile(member))
                new=json.loads((Path(replays)/f'bbro__{model}__seed0/PhenoBench/FULL_RESULTS.json').read_text())
                # Byte provenance intentionally changes after removing GT/path
                # metadata; all scientific fields, splits, runs and summaries
                # remain subject to the full recursive comparison.
                excluded=['prediction_sha256']
                for k in excluded:old.pop(k);new.pop(k)
                stats={'leaf_values':0,'numeric_values':0,'max_absolute_numeric_difference':0.,'nonidentical_numeric_values':0}
                compare_cells(old,new,stats)
                full.append({'job':f'bbro__{model}__seed0','scientific_FULL_RESULTS_comparison':'PASSED',
                     'excluded_provenance_fields':excluded,**stats})
                print('FULL_SCIENTIFIC_RESULTS_MATCH',model,flush=True)
                if not wanted:break
        assert not wanted,'ORIGINAL_FULL_RESULTS_NOT_FOUND'
    return {'status':'PASSED','scope':'bbro rotation; ASSET-DETR seed0 and YOLO26m seed0; PhenoBench',
            'methods':7,'Ks':[5,10,20,50,100],'nozzles':[1,2,3,4],'schemes':['ideal','grouped'],'repeats':200,
            'tables':reports,'table_rows_checked':sum(r['rows'] for r in reports),
            'table_cells_checked':sum(r['leaf_values'] for r in reports),'FULL_RESULTS':full,
            'original_results_modified':False,'scientific_functions_changed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replays',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=ROOT/'results');p.add_argument('--original-backup',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    result=verify(a.replays,a.reference,a.original_backup)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE_REPLAY_COMPARISON_PASSED',result['table_rows_checked'],result['table_cells_checked'])
