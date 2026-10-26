"""CPU-only frozen risk methods, date stratification and pixel crop exposure."""
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
import math
from pathlib import Path
import statistics
import sys
from common import HERE,read,save,sha,csv_write
from data_prep.grouping import image_date

def semantics_path(c, im):
    return Path(c["pheno_semantics"])/(Path(im.get("file_name", im["id"])).stem+".png")

from wcr import (load_manifest,threshold_grid,ImageRecord,Prediction,spray_union,
                 image_metrics,aggregate,prediction_regions,merge_intervals)
from risk_control import (run_experiment,RiskConfig,CurveDataset,Selection,
                          summarize_runs,full_spray_metrics,select_threshold)


def relation(calibration_dates,test_date):
    if not calibration_dates or test_date is None:
        return 'date_unresolved'
    assert len({len(d) for d in calibration_dates+[test_date] if d is not None})==1,'MIXED_DATE_FORMAT_WITHIN_DATASET'
    if test_date in calibration_dates:return 'same_day_as_calibration'
    if any(d is None for d in calibration_dates):return 'date_unresolved'
    if test_date<min(calibration_dates):return 'earlier_than_all_calibration'
    if test_date>max(calibration_dates):return 'later_than_all_calibration'
    return 'other'


def pixel_rect_count(integral,left,right,top,bottom):
    """Pixels are unit squares indexed by centers; half-open spray boundaries."""
    H,W=integral.shape[0]-1,integral.shape[1]-1
    x0=max(0,min(W,math.ceil(left-.5)));x1=max(0,min(W,math.ceil(right-.5)))
    y0=max(0,min(H,math.ceil(top-.5)));y1=max(0,min(H,math.ceil(bottom-.5)))
    return int(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])


def union_crop_count(integral,strips,H):
    return sum(pixel_rect_count(integral,l,r,k*H/len(strips),(k+1)*H/len(strips))
               for k,intervals in enumerate(strips) for l,r in intervals)


class CropCurve:
    def __init__(self,record,n,integral):
        self.total=int(integral[-1,-1]);self.H=record.height
        groups=defaultdict(list)
        for pred in record.predictions:
            if pred.box.class_id==1:groups[pred.confidence].append(pred.box)
        strips=[() for _ in range(n)];values={}
        for conf in sorted(groups,reverse=True):
            for box in groups[conf]:
                for k,interval in prediction_regions(record,box,n):
                    strips[k]=merge_intervals([*strips[k],interval])
            values[conf]=union_crop_count(integral,strips,record.height)
        self.conf=sorted(values);self.hit=[values[x] for x in self.conf]
        assert all(a>=b for a,b in zip(self.hit,self.hit[1:]))
        truth=ImageRecord(record.image_id,record.width,record.height,record.ground_truth,
                    tuple(Prediction(b,1.) for b in record.ground_truth if b.class_id==1),record.group_id)
        self.reference=union_crop_count(integral,spray_union(truth,n,1.),record.height)

    def at(self,selection):
        if selection.mode=='unavailable':return None
        if selection.mode=='full_spray':return self.total
        i=bisect_left(self.conf,selection.threshold)
        return self.hit[i] if i<len(self.hit) else 0


def crop_metrics(curves,indices,selection,reference=False):
    total=sum(curves[i].total for i in indices)
    hits=[curves[i].reference if reference else curves[i].at(selection) for i in indices]
    if any(h is None for h in hits):return None
    macro=[h/curves[i].total for i,h in zip(indices,hits) if curves[i].total]
    return {'crop_pixels':total,'sprayed_crop_pixels':sum(hits),
            'crop_exposure':sum(hits)/total if total else None,
            'crop_image_mean_exposure':statistics.fmean(macro) if macro else None,
            'crop_image_exposure_sum':math.fsum(macro),
            'crop_positive_images':len(macro),'zero_crop_images':len(indices)-len(macro)}


def summarized_numeric(rows,keys):
    ans={'repeats_present':len(rows)}
    for key in keys:
        vals=[r[key] for r in rows if r and r.get(key) is not None]
        ans['mean_'+key]=statistics.fmean(vals) if vals else None
        ans['valid_'+key+'_repeats']=len(vals)
    return ans


def metric_summary(metrics):
    m=aggregate(metrics);eligible=[x for x in metrics if x.weed_total]
    rate=sum((Fraction(x.weed_total-x.weed_sprayed,x.weed_total) for x in eligible),Fraction(0))
    m.update(weed_image_mean_miss_rate=float(rate/len(eligible)) if eligible else None,
             weed_image_rate_sum_exact=[rate.numerator,rate.denominator])
    return m


def augment(base,records,raw_images,dataset,c,out):
    """Reuse exact stored split indices/selections; do not draw any new samples."""
    dates=[image_date(im,dataset,c['date_overrides'])[0] for im in raw_images]
    splitmap={(s['scheme'],s['K'],s['repeat']):s for s in base['splits'] if 'error' not in s}
    rq3=[];directions=[];crop_rows=[];reference=[];transfer=[];crop_curves={}
    if dataset=='PhenoBench':
        import numpy as np
        from PIL import Image
        for n in c['experiment']['nozzles']:crop_curves[n]=[]
        for idx,(record,im) in enumerate(zip(records,raw_images)):
            with Image.open(semantics_path(c,im)) as image:semantics=np.asarray(image)
            assert semantics.ndim==2 and semantics.shape==(record.height,record.width),'SEMANTICS_DIMENSION_MISMATCH'
            mask=np.isin(semantics,[1,3]);integral=np.zeros((record.height+1,record.width+1),dtype=np.int64)
            integral[1:,1:]=mask.cumsum(0,dtype=np.int64).cumsum(1,dtype=np.int64)
            for n in crop_curves:crop_curves[n].append(CropCurve(record,n,integral))
            if (idx+1)%100==0:print('CROP_PIXEL_CURVES',idx+1,flush=True)
    for n in c['experiment']['nozzles']:
        td=CurveDataset(records,n)
        truth=[ImageRecord(r.image_id,r.width,r.height,r.ground_truth,
                   tuple(Prediction(b,1.) for b in r.ground_truth if b.class_id==1),r.group_id) for r in records]
        refmetrics=[image_metrics(r,spray_union(r,n,1.)) for r in truth]
        assert all(m.weed_total==m.weed_sprayed for m in refmetrics)
        source_choice=Selection(**next({k:v for k,v in s.items() if k!='nozzles'} for s in base['source_selections'] if s['nozzles']==n))
        full=list(range(len(records)))
        for method,metrics in [('source',td.metrics(full,source_choice)),('perfect_detection_reference',metric_summary(refmetrics))]:
            row={'dataset':dataset,'nozzles':n,'method':method,'metrics':metrics}
            if crop_curves:row['crop']=crop_metrics(crop_curves[n],full,source_choice,method=='perfect_detection_reference')
            transfer.append(row)
        # Reference is paired to each existing method's frozen target test subset.
        for scheme in c['experiment']['schemes']:
            for K in c['experiment']['Ks']:
                runs=[]
                for repeat in range(c['experiment']['repeats']):
                    s=splitmap.get((scheme,K,repeat))
                    if s is None:
                        runs.append({'repeat':repeat,'K_prime':None,'selection':asdict(Selection('unavailable',None,False,'original_split_unavailable')),'test_metrics':None});continue
                    m=metric_summary([refmetrics[i] for i in s['test']])
                    m['mean_signed_loss']=-.1*m['weed_total']/m['images']
                    run={'repeat':repeat,'K_prime':s['K_prime'],'selection':asdict(Selection('threshold',1.,False,'test_GT_reference_not_deployable')),'test_metrics':m}
                    if crop_curves:run['crop']=crop_metrics(crop_curves[n],s['test'],source_choice,True)
                    runs.append(run)
                reference.append({'method':'perfect_detection_reference','K':K,'scheme':scheme,'nozzles':n,
                                  'summary':summarize_runs(runs,.1),'runs':runs})
        for cell in [x for x in base['results']+reference if x['nozzles']==n]:
            for run in cell['runs']:
                s=splitmap.get((cell['scheme'],cell['K'],run['repeat']))
                if s is None:continue
                selection=Selection(**run['selection'])
                if crop_curves and cell['method'] in c['crop_methods']:
                    value=crop_metrics(crop_curves[n],s['test'],selection,cell['method']=='perfect_detection_reference')
                    crop_rows.append({'dataset':dataset,'nozzles':n,'K':cell['K'],'scheme':cell['scheme'],'method':cell['method'],'repeat':run['repeat'],
                                      'available':value is not None,**(value if value is not None else {'crop_exposure':None,'crop_image_mean_exposure':None})})
                if cell['method']!='crc' or cell['scheme']!='grouped' or selection.mode=='unavailable':continue
                caldates=sorted({dates[i] for i in s['calibration'] if dates[i] is not None})
                calibration_missing=sum(dates[i] is None for i in s['calibration'])
                actual_dates=[dates[i] for i in s['calibration']]
                buckets=defaultdict(list);bydate=defaultdict(list)
                for i in s['test']:
                    buckets[relation(actual_dates,dates[i])].append(i);bydate[dates[i] or 'date_unresolved'].append(i)
                for kind in ['earlier_than_all_calibration','later_than_all_calibration','same_day_as_calibration','other','date_unresolved']:
                    indices=buckets.get(kind,[])
                    m=td.metrics(indices,selection) if indices else None
                    rq3.append({'dataset':dataset,'nozzles':n,'K':cell['K'],'repeat':run['repeat'],'relation':kind,
                          'calibration_dates':' + '.join(caldates),'calibration_date_missing':calibration_missing,
                          'images':len(indices),'weed_image_mean_miss_rate':m['weed_image_mean_miss_rate'] if m else None,
                          'weed_images':m['images_with_weeds'] if m else 0,
                          'image_area_px2':m['image_area_px2'] if m else 0,
                          'spray_area_px2':m['spray_area_px2'] if m else 0,
                          'spray_area_fraction':m['spray_area_fraction'] if m else None})
                if dataset=='PhenoBench':
                    # Both the original trial-style date-combination table and a single-test-date table.
                    for label,indices in [(' + '.join(sorted(bydate)),s['test']),*list(bydate.items())]:
                        m=td.metrics(indices,selection)
                        directions.append({'nozzles':n,'K':cell['K'],'repeat':run['repeat'],
                             'table':'date_combination' if indices is s['test'] else 'single_test_date',
                             'calibration_dates':' + '.join(caldates),'test_dates':label,'images':len(indices),
                             'weed_images':m['images_with_weeds'],'image_area_px2':m['image_area_px2'],
                             'K_prime':run['K_prime'],'threshold':selection.threshold,'fallback':selection.fallback,
                             'macro_pass':(Fraction(*m['weed_image_rate_sum_exact'])<=Fraction('0.1')*m['images_with_weeds']) if m['images_with_weeds'] else None,
                             'miss_contribution_in_draw':(m['weed_image_mean_miss_rate']*m['images_with_weeds']/run['test_metrics']['images_with_weeds']) if m['images_with_weeds'] and run['test_metrics']['images_with_weeds'] else None,
                             'area_contribution_in_draw':m['spray_area_px2']/run['test_metrics']['image_area_px2'],
                             'weed_image_mean_miss_rate':m['weed_image_mean_miss_rate'],'spray_area_fraction':m['spray_area_fraction']})
    base['results'].extend(reference)
    save(out/'WHOLE_DATASET_SOURCE_AND_GT_REFERENCE.json',transfer)
    csv_write(out/'WHOLE_DATASET_SOURCE_AND_GT_REFERENCE.csv',[
        {**{k:r[k] for k in ('dataset','nozzles','method')},
         **({k:v for k,v in r['metrics'].items() if not isinstance(v,(dict,list))} if r['metrics'] is not None else {'weed_image_mean_miss_rate':None,'spray_area_fraction':None}),
         **(r.get('crop') or {})} for r in transfer])
    save(out/'RQ3_RUNS.json',rq3);save(out/'PHENO_DIRECTION_RUNS.json',directions)
    if crop_rows:save(out/'CROP_EXPOSURE_RUNS.json',crop_rows)
    def group_summary(rows,keys,values):
        buckets=defaultdict(list)
        for r in rows:buckets[tuple(r[k] for k in keys)].append(r)
        return [{**dict(zip(keys,key)),**summarized_numeric(rs,values)} for key,rs in sorted(buckets.items())]
    csv_write(out/'RQ3_SUMMARY.csv',group_summary(rq3,['dataset','nozzles','K','relation'],['weed_image_mean_miss_rate','spray_area_fraction','images']))
    direction_keys=['nozzles','K','table','calibration_dates','test_dates']
    direction_summary=group_summary(directions,direction_keys,['weed_image_mean_miss_rate','spray_area_fraction','images','weed_images','threshold','K_prime'])
    for row in direction_summary:
        matches=[r for r in directions if all(r[k]==row[k] for k in direction_keys)]
        whole=[r for r in directions if r['nozzles']==row['nozzles'] and r['K']==row['K'] and r['table']=='date_combination']
        valid_macro=sum(r['weed_image_mean_miss_rate'] is not None for r in whole)
        row.update(repeats=len(matches),repeat_presence_fraction=len(matches)/c['experiment']['repeats'],
             weighted_contribution_to_overall_miss_rate=math.fsum(r['miss_contribution_in_draw'] for r in matches if r['miss_contribution_in_draw'] is not None)/valid_macro if valid_macro else None,
             weighted_contribution_to_overall_spray_area=math.fsum(r['area_contribution_in_draw'] for r in matches)/len(whole),
             macro_miss_le_0_1_count=sum(r['macro_pass'] is True for r in matches),
             macro_miss_valid_repeats=sum(r['macro_pass'] is not None for r in matches),
             full_spray_fallback_count=sum(r['fallback'] for r in matches),
             K_prime_min=min(r['K_prime'] for r in matches),K_prime_max=max(r['K_prime'] for r in matches))
    csv_write(out/'PHENO_DATE_DIRECTIONS.csv',direction_summary)
    crop_summary=group_summary(crop_rows,['dataset','nozzles','K','scheme','method'],['crop_exposure','crop_image_mean_exposure','crop_pixels','sprayed_crop_pixels','zero_crop_images'])
    for row in crop_summary:
        all_matches=[r for r in crop_rows if all(r[k]==row[k] for k in ['dataset','nozzles','K','scheme','method'])]
        matches=[r for r in all_matches if r['available']]
        denom=sum(r['crop_pixels'] for r in matches);num=sum(r['sprayed_crop_pixels'] for r in matches)
        row.update(crop_exposure_pooled_over_repeats=num/denom if denom else None,
                   repeated_crop_pixels=denom,repeated_sprayed_crop_pixels=num,
                   crop_image_exposure_pooled_over_repeats=(sum(r['crop_image_exposure_sum'] for r in matches)/sum(r['crop_positive_images'] for r in matches)) if sum(r['crop_positive_images'] for r in matches) else None,
                   zero_denominator_repeats=sum(r['crop_pixels']==0 for r in matches),
                   unavailable_policy_repeats=len(all_matches)-len(matches))
    csv_write(out/'CROP_EXPOSURE_SUMMARY.csv',crop_summary)
    base['additional_analysis']={'dates':dict(zip((r.image_id for r in records),dates)),
        'RQ3_mean_rule':'equal mean of per-draw subgroup metrics; empty/no-weed subgroups excluded only for their undefined metric; valid repeat counts retained',
        'crop_mean_rule':'each draw pools crop pixels; report both pooled counts over repeated subsets and equal mean of per-draw ratios; image macro excludes zero-crop images, with pooled and per-draw-mean supplements separate',
        'crop_pixel_rule':c['crop_pixel_rule'],'GT_reference_is_not_area_lower_bound':True}
    return base


def result_plots(folder,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ds in sorted({r['dataset'] for r in rows}):
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for method in sorted({r['method'] for r in rows}):
            selected=sorted([r for r in rows if r['dataset']==ds and r['scheme']=='grouped' and r['nozzles']==4 and r['method']==method],key=lambda r:r['K'])
            for ax,key in zip(axes,['weed_image_miss_rate_le_alpha_fraction','mean_spray_area_fraction']):
                ax.plot([r['K'] for r in selected],[r[key] if r[key] is not None else float('nan') for r in selected],marker='o',label=method)
        axes[0].set(xlabel='K',ylabel='P(observed macro miss <= 0.1)',ylim=(0,1.03))
        axes[1].set(xlabel='K',ylabel='Mean spray area fraction',ylim=(0,1.03));axes[1].legend(fontsize=7)
        fig.suptitle(ds+' | n=4 grouped');fig.savefig(folder/(ds+'_K_CURVES.png'),dpi=160);plt.close(fig)

