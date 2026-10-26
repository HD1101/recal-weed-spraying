"""Post-result descriptive box statistics. Ground truth only; no predictions."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from common import read,csv_write
from grouping import lb_meta,image_date
from prepare import parse_xml,meta_rows


def collect(c):
    entries=[];raw=read(c['lb_coco']);ann=defaultdict(list)
    for a in raw['annotations']:
        x,y,w,h=a['bbox'];ann[a['image_id']].append((a['category_id'],x,y,x+w,y+h))
    for im in raw['images']:
        meta=lb_meta(im['file_name'])
        entries.append(('LB','all_annotated','field_date',meta['field'],'',meta['capture_date'],
            'LB_filename',im['width'],im['height'],meta['group_id'],ann[im['id']]))
    import numpy as np
    from PIL import Image
    from pheno_boxes import pheno_boxes
    for im in meta_rows('PhenoBench_IMAGE_GROUPS.csv'):
        root=Path(c['pheno_val'])
        with Image.open(root/'semantics'/im['file_name']) as f:sem=np.asarray(f)
        with Image.open(root/'plant_instances'/im['file_name']) as f:ins=np.asarray(f)
        boxes,bad=pheno_boxes(sem,ins,1,im['file_name']);assert not bad
        entries.append(('PhenoBench','validation','date','','',im['group_id'],'PhenoBench_filename_month_day',
                        int(im['width']),int(im['height']),im['group_id'],boxes))
    for im in meta_rows('OD_SugarBeets_IMAGE_GROUPS.csv'):
        boxes=parse_xml(Path(c['od_xml'])/(Path(im['file_name']).stem+'.xml'))
        date,basis=image_date(im,'OD_SugarBeets',{})
        basis='inferred_Unix_timestamp_UTC' if im['group_id'].startswith('inferred_timestamp_date:') else 'explicit_sequence_filename'
        entries.append(('OD_SugarBeets','test','group','',im['group_id'],date,basis,
                        int(im['width']),int(im['height']),im['group_id'],boxes))
    return entries


def summarize(entries):
    buckets=defaultdict(list)
    for r in entries:buckets[r[:7]].append(r)
    rows=[]
    for key,items in sorted(buckets.items()):
        for cls,name in [(0,'beet'),(1,'weed')]:
            areas=[];fractions=[];positive=0;outside=0
            for r in items:
                W,H=r[7:9];boxes=[b for b in r[10] if b[0]==cls];positive+=bool(boxes)
                for _,x1,y1,x2,y2 in boxes:
                    area=(x2-x1)*(y2-y1);areas.append(area);fractions.append(area/(W*H))
                    outside+=not (0<=x1<x2<=W and 0<=y1<y2<=H)
            row=dict(zip(('dataset','subset','aggregation_level','field','group_id','capture_date','date_basis'),key))
            row.update(class_id=cls,class_name=name,image_count=len(items),images_with_class=positive,box_count=len(areas),
                boxes_per_image_mean=statistics.fmean([len([b for b in r[10] if b[0]==cls]) for r in items]),
                box_area_px2_mean=statistics.fmean(areas) if areas else None,box_area_px2_median=statistics.median(areas) if areas else None,
                box_area_image_fraction_mean=statistics.fmean(fractions) if areas else None,
                box_area_image_fraction_median=statistics.median(fractions) if areas else None,
                width_px_min=min(r[7] for r in items),width_px_max=max(r[7] for r in items),
                height_px_min=min(r[8] for r in items),height_px_max=max(r[8] for r in items),
                source_group_count=len({r[9] for r in items}),outside_image_box_count=outside)
            for k,v in row.items():
                if isinstance(v,float):row[k]=format(v,'.9f' if 'image_fraction' in k else '.6f')
            rows.append(row)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'data_prep/config.json')
    p.add_argument('--output',type=Path,default=Path('reproduced-truth-statistics.csv'));a=p.parse_args()
    assert not a.output.exists();rows=summarize(collect(read(a.config)));csv_write(a.output,rows)
    print('POST_RESULT_DESCRIPTIVE_STATISTICS',len(rows),'rows; GT only')
