"""Array cache of ORIGINAL confidence-event geometry, on the formal grid.

The original ImageCurve, CropCurve, prediction_regions and containment rules
remain the sole geometric implementations. This cache is a query/I/O adapter,
not a replacement nozzle model. Fractions are used at selection boundaries.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from additional import CropCurve
from common import sha
from wcr import Box, ImageCurve, ImageRecord, Prediction, load_yolo, threshold_grid

# The archived formal configuration passed grid_step as JSON float 0.001.
# Preserve those exact float representations, including e.g. 0.11800000000000001.
GRID = np.asarray(threshold_grid(.001), dtype=np.float64)


def load_records(prediction, manifest):
    raw=json.loads(Path(prediction).read_text())
    gt=json.loads(Path(manifest).read_text())
    lookup={im['id']:im for im in gt['images']}
    records=[]
    for im in raw['images']:
        local=lookup[im['id']]
        assert tuple(im[k] for k in ('width','height','group_id'))==tuple(local[k] for k in ('width','height','group_id'))
        truth=load_yolo(Path(manifest).parent/local['gt_yolo'],im['width'],im['height'])
        predictions=tuple(Prediction(Box(*p['xyxy'],class_id=p['class_id']),p['confidence']) for p in im['predictions'])
        records.append(ImageRecord(im['id'],im['width'],im['height'],truth,predictions,im['group_id']))
    return tuple(records),[{k:v for k,v in im.items() if k!='predictions'} for im in raw['images']]


def build_arrays(records, metadata, semantics=None, progress=None):
    size=len(records)
    data={'counts':np.asarray([sum(b.class_id==1 for b in r.ground_truth) for r in records],dtype=np.int32),
          'image_areas':np.asarray([r.width*r.height for r in records],dtype=np.int64),
          'ids':np.asarray([r.image_id for r in records])}
    for n in (1,2,3,4):
        data[f'hits_{n}']=np.zeros((size,len(GRID)),dtype=np.int16)
        data[f'areas_{n}']=np.zeros((size,len(GRID)),dtype=np.float64)
        if semantics is not None:data[f'crop_hits_{n}']=np.zeros((size,len(GRID)),dtype=np.int32)
    if semantics is not None:data['crop_totals']=np.zeros(size,dtype=np.int64)
    started=time.monotonic()
    for i,(record,meta) in enumerate(zip(records,metadata)):
        integral=None
        if semantics is not None:
            path=Path(semantics)/(Path(meta.get('file_name',record.image_id)).stem+'.png')
            with Image.open(path) as f:sem=np.asarray(f)
            assert sem.shape==(record.height,record.width)
            mask=np.isin(sem,(1,3));integral=np.zeros((record.height+1,record.width+1),dtype=np.int64)
            integral[1:,1:]=mask.cumsum(0,dtype=np.int64).cumsum(1,dtype=np.int64)
            data['crop_totals'][i]=int(integral[-1,-1])
        for n in (1,2,3,4):
            curve=ImageCurve(record,n)
            positions=np.searchsorted(curve.confidences,GRID,side='left')
            states=curve.states+[curve.empty]
            data[f'hits_{n}'][i]=np.asarray([s.weed_sprayed for s in states],dtype=np.int16)[positions]
            data[f'areas_{n}'][i]=np.asarray([s.spray_area_px2 for s in states],dtype=np.float64)[positions]
            assert np.all(np.diff(data[f'hits_{n}'][i].astype(np.int32))<=0)
            assert np.all(np.diff(data[f'areas_{n}'][i])<=max(1e-9,record.width*record.height*1e-12))
            if integral is not None:
                cc=CropCurve(record,n,integral)
                indices=np.searchsorted(cc.conf,GRID,side='left')
                data[f'crop_hits_{n}'][i]=np.asarray(cc.hit+[0],dtype=np.int32)[indices]
        if progress and ((i+1)%100==0 or i+1==size):
            progress(i+1,size,round(time.monotonic()-started,1))
    return data


def load_or_build_cache(records, metadata, prediction, manifest, cache, verification, semantics=None, progress=None):
    cache=Path(cache)
    binding={'prediction_sha256':sha(prediction),'manifest_sha256':sha(manifest),
             'input_verification_sha256':sha(verification),
             'original_geometry_code_sha256':sha(Path(__file__).resolve().parents[1]/'wcr.py'),
             'original_crop_code_sha256':sha(Path(__file__).resolve().parents[1]/'additional.py'),
             'engine_sha256':sha(Path(__file__)),'grid':{'start':0,'end':1,'step':.001}}
    marker=cache.with_suffix('.binding.json')
    if cache.exists():
        saved=json.loads(marker.read_text());assert saved['binding']==binding,'GEOMETRY_CACHE_BINDING_MISMATCH'
        assert saved['cache_sha256']==sha(cache),'GEOMETRY_CACHE_HASH'
        with np.load(cache,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        assert list(data['ids'])==[r.image_id for r in records]
        return data
    assert not marker.exists()
    data=build_arrays(records,metadata,semantics,progress)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,**data)
    marker.write_text(json.dumps({'binding':binding,'cache_sha256':sha(cache)},indent=2)+'\n')
    return data


class ArrayDataset:
    """CurveDataset-compatible exact queries to the unchanged cached geometry."""
    def __init__(self,records,data,n):
        self.records=records;self.data=data;self.nozzles=n
        self.count_array=data['counts'];self.counts=tuple(int(v) for v in self.count_array)
        self.hits=data[f'hits_{n}'];self.areas=data[f'areas_{n}']
        self.image_areas=data['image_areas']
        self._indices={float(t):i for i,t in enumerate(GRID)}

    def column(self,t):return self._indices[t]

    def counts_at(self,indices,threshold):
        ix=np.asarray(indices,dtype=np.intp);col=self.column(threshold)
        total=int(self.count_array[ix].sum());hits=int(self.hits[ix,col].sum())
        return total,total-hits

    def _rate_sum(self,indices,hits):
        counts=self.count_array[indices]
        missed=counts-hits
        numerators=np.bincount(counts,weights=missed,minlength=int(self.count_array.max())+1)
        return sum((Fraction(int(v),n) for n,v in enumerate(numerators) if n and v),Fraction(0))

    def rate_sum_at(self,indices,threshold):
        ix=np.asarray(indices,dtype=np.intp)
        return self._rate_sum(ix,self.hits[ix,self.column(threshold)])

    def validate_bound(self,config):
        if max(self.counts)>config.n_max:raise ValueError('count bound exceeded')

    def metrics(self,indices,selection):
        if selection.mode=='unavailable' or not indices:return None
        ix=np.asarray(indices,dtype=np.intp);counts=self.count_array[ix]
        if selection.mode=='full_spray':
            hits=counts;areas=self.image_areas[ix]
        else:
            col=self.column(selection.threshold);hits=self.hits[ix,col];areas=self.areas[ix,col]
        weeds=int(counts.sum());sprayed=int(hits.sum());positive=counts>0
        rate=self._rate_sum(ix,hits);npositive=int(positive.sum())
        area=math.fsum(areas);denom=int(self.image_areas[ix].sum())
        return {'images':len(indices),'images_with_weeds':npositive,'weed_total':weeds,'weed_sprayed':sprayed,
                'wcr_per_weed':sprayed/weeds if weeds else None,
                'wcr_image_mean':math.fsum(hits[positive]/counts[positive])/npositive if npositive else None,
                'spray_area_px2':area,'image_area_px2':denom,'spray_area_fraction':area/denom,
                'weed_image_mean_miss_rate':float(rate/npositive) if npositive else None,
                'weed_image_rate_sum_exact':[rate.numerator,rate.denominator],
                'mean_signed_loss':float(Fraction(weeds-sprayed)-Fraction('0.1')*weeds)/len(indices)}

    def crop(self,indices,selection):
        if 'crop_totals' not in self.data or not indices or selection.mode=='unavailable':return None
        ix=np.asarray(indices,dtype=np.intp);totals=self.data['crop_totals'][ix]
        hits=totals if selection.mode=='full_spray' else self.data[f'crop_hits_{self.nozzles}'][ix,self.column(selection.threshold)]
        total=int(totals.sum());sprayed=int(hits.sum());positive=totals>0
        return {'crop_pixels':total,'sprayed_crop_pixels':sprayed,
                'crop_exposure':sprayed/total if total else None,
                'crop_image_mean_exposure':float(np.mean(hits[positive]/totals[positive])) if positive.any() else None}
