"""Compare regenerated analysis tables with delivered CSVs; original files stay read-only."""
import argparse
import csv
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=[
 'PER_SEED_ALL_RESULTS.csv','CROSS_SEED_ALL_RESULTS.csv','MAIN_TABLE_N4_GROUPED_K20_K50.csv',
 'MAIN_RQ1_WHOLE_TARGET_N4.csv','RQ1_WHOLE_TARGET_PER_SEED.csv','RQ1_WHOLE_TARGET_CROSS_SEED.csv',
 'SUPPLEMENT_N2_GROUPED_K20_K50.csv','SUPPLEMENT_WALLEDGARDEN_N4_GROUPED_K20_K50.csv',
 'RQ3_SUMMARY_PER_SEED.csv','RQ3_SUMMARY_CROSS_SEED.csv',
 'PHENO_DATE_DIRECTIONS_PER_SEED.csv','PHENO_DATE_DIRECTIONS_CROSS_SEED.csv',
 'PHENO_DATE_DIRECTIONS_BBRO_N4_K50_GROUPED_CROSS_SEED.csv',
 'CROP_EXPOSURE_SUMMARY_PER_SEED.csv','CROP_EXPOSURE_SUMMARY_CROSS_SEED.csv',
 'MIN_THRESHOLD_PER_SEED_ALL_N.csv','MIN_THRESHOLD_CROSS_SEED_ALL_N.csv',
 'MIN_THRESHOLD_PER_SEED_N4.csv','MIN_THRESHOLD_CROSS_SEED_N4.csv']


def read_csv(path):
    with path.open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);return reader.fieldnames,rows


def compare(generated):
    checked=0
    for name in NAMES:
        ah,a=read_csv(ROOT/'results'/name);bh,b=read_csv(Path(generated)/name)
        assert set(ah)==set(bh),('COLUMN_MISMATCH',name)
        key=[k for k in ['rotation','model','dataset','method','scheme','nozzles','K','seed','relation','table','calibration_dates','test_dates'] if k in ah]
        index=lambda rs:{tuple(r[k] for k in key):r for r in rs}
        aa,bb=index(a),index(b)
        assert len(aa)==len(a) and len(bb)==len(b) and set(aa)==set(bb),('ROW_KEYS',name)
        for identity,row in aa.items():
            for field,value in row.items():
                other=bb[identity][field]
                if value==other:checked+=1;continue
                try:equal=math.isclose(float(value),float(other),abs_tol=1e-12,rel_tol=1e-12)
                except (ValueError,TypeError):equal=False
                assert equal,('VALUE_MISMATCH',name,identity,field,value,other)
                checked+=1
        print('TABLE_MATCH',name,len(a),'rows')
    print('ALL_ANALYSIS_TABLES_MATCH',len(NAMES),'files',checked,'cells')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generated',type=Path,required=True)
    compare(p.parse_args().generated)
