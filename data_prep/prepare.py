"""Prepare original labels in a new directory. No source data are changed."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from common import read,save
from grouping import lb_meta


def write_yolo(path,boxes,W,H):
    lines=[]
    for cls,x1,y1,x2,y2 in boxes:
        if cls not in (0,1) or not (0<=x1<x2<=W and 0<=y1<y2<=H):
            raise ValueError('INVALID_OR_OUTSIDE_GT_BOX: '+str(path))
        values=((x1+x2)/(2*W),(y1+y2)/(2*H),(x2-x1)/W,(y2-y1)/H)
        lines.append(str(cls)+' '+' '.join(format(v,'.17g') for v in values))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+('\n' if lines else ''))


def meta_rows(name):
    with (ROOT/'splits'/name).open(newline='') as f:return list(csv.DictReader(f))


def emit(out,dataset,items):
    rows=[]
    for meta,boxes in items:
        ident=meta['id']; W,H=int(meta['width']),int(meta['height'])
        label=out/dataset/'labels'/(ident+'.txt')
        write_yolo(label,boxes,W,H)
        rows.append({**meta,'width':W,'height':H,'gt_yolo':'labels/'+ident+'.txt','predictions':[]})
    save(out/dataset/'GT_ONLY_manifest.json',{'schema_version':1,'images':rows})
    return rows


def prepare_lb(c,out):
    raw=read(c['lb_coco'])
    assert {x['id']:x['name'] for x in raw['categories']}=={0:'sugar_beet',1:'weed'}
    annotations=defaultdict(list)
    for a in raw['annotations']:annotations[a['image_id']].append(a)
    items=[]
    for im in raw['images']:
        boxes=[]
        for a in annotations[im['id']]:
            x,y,w,h=a['bbox']
            assert all(float(v).is_integer() for v in (x,y,w,h))
            boxes.append((a['category_id'],x,y,x+w,y+h))
        meta={'id':Path(im['file_name']).stem,'file_name':im['file_name'],
              'width':im['width'],'height':im['height'],**lb_meta(im['file_name'])}
        items.append((meta,boxes))
    rows=emit(out,'LB',items)
    assert len(rows)==4402 and len(raw['annotations'])==39246
    return {'images':len(rows),'boxes':len(raw['annotations']),'duplicate_labels_preserved':True}


def prepare_pheno(c,out):
    import numpy as np
    from PIL import Image
    from pheno_boxes import pheno_boxes
    assert c['pheno_bbox_convention']=='max_minus_min_plus_one'
    assert c['pheno_tie_break']=='lowest_semantic_id'
    base=Path(c['pheno_val']);items=[];audit={};counts=[0,0]
    for meta in meta_rows('PhenoBench_IMAGE_GROUPS.csv'):
        name=meta['file_name']
        with Image.open(base/'semantics'/name) as f:semantics=np.asarray(f)
        with Image.open(base/'plant_instances'/name) as f:instances=np.asarray(f)
        assert semantics.shape==instances.shape==(int(meta['height']),int(meta['width']))
        boxes,bad=pheno_boxes(semantics,instances,1,name,audit,'lowest_semantic_id')
        assert not bad
        for b in boxes:counts[b[0]]+=1
        items.append((meta,boxes))
    rows=emit(out,'PhenoBench',items)
    assert (len(rows),counts)==(772,[6482,3926]),'PHENO_VERSION_OR_CONVERSION_MISMATCH'
    report={'images':len(rows),'beet_boxes':counts[0],'weed_boxes':counts[1],**audit}
    save(out/'PhenoBench/CONVERSION_REPORT.json',report)
    return report


def parse_xml(path):
    """Original XML floating coordinates: no int(float), ±1, or clipping."""
    result=[]
    for obj in ET.parse(path).findall('object'):
        name=obj.findtext('name').lower()
        assert name in ('crop','weed','beet','sugar_beet','sugar beet'),name
        cls=1 if name=='weed' else 0
        bb=obj.find('bndbox')
        xyxy=[float(bb.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        assert xyxy[2]>xyxy[0] and xyxy[3]>xyxy[1]
        result.append((cls,*xyxy))
    return result


def prepare_od(c,out):
    assert c['od_coordinate_convention']=='xml_float_xyxy'
    items=[];all_counts=[0,0];outside_crops=0
    for meta in meta_rows('OD_SugarBeets_IMAGE_GROUPS.csv'):
        filename=Path(meta['file_name']).with_suffix('.xml').name
        boxes=parse_xml(Path(c['od_xml'])/filename)
        for cls,x1,y1,x2,y2 in boxes:
            all_counts[cls]+=1
            outside_crops+=int(cls==0 and not (0<=x1<x2<=int(meta['width']) and 0<=y1<y2<=int(meta['height'])))
        # Formal OD risk manifests contain the 1,978 weeds only. Crop XML boxes
        # are parsed for the descriptive audit; they are not used for spraying.
        items.append((meta,[b for b in boxes if b[0]==1]))
    rows=emit(out,'OD_SugarBeets',items)
    assert len(rows)==964 and all_counts==[1890,1978],'OD_VERSION_MISMATCH'
    result={'test_images':len(rows),'original_xml_crop_boxes':all_counts[0],
            'formal_weed_boxes':all_counts[1],'outside_crop_boxes_not_clipped':outside_crops,
            'formal_labels':'weed-only, matching original evaluation manifests','groups':18,
            'validation_count_difference':'Original paper 387 versus supplied archive 385; nonblocking test-only evaluation.'}
    save(out/'OD_SugarBeets/OD_REPORT.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'data_prep/config.json')
    p.add_argument('--dataset',choices=['LB','PhenoBench','OD_SugarBeets','all'],default='all')
    a=p.parse_args();c=read(a.config);out=Path(c['output'])
    names=['LB','PhenoBench','OD_SugarBeets'] if a.dataset=='all' else [a.dataset]
    for ds in names:
        assert not (out/ds).exists(),'EXISTING_PREPARED_DATA_PRESERVED'
        result={'LB':prepare_lb,'PhenoBench':prepare_pheno,'OD_SugarBeets':prepare_od}[ds](c,out)
        print('PREPARED',ds,json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}))


if __name__=='__main__':main()
