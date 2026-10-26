"""CPU replay from archived predictions and locally regenerated ground truth.

Original prediction JSONs and all checked-in result CSVs are read-only inputs.
This is an I/O adapter around the unchanged formal scientific functions.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import read,save,sha,csv_write
from wcr import Box,ImageRecord,Prediction,load_yolo,threshold_grid
from risk_control import RiskConfig,run_experiment
from additional import augment,result_plots
from ceilings import compute_rows


def load_portable(prediction,prepared,dataset):
    raw=read(prediction)
    manifest=Path(prepared)/('LB' if dataset in ('source_holdout','bbro','near30','walledgarden') else dataset)/'GT_ONLY_manifest.json'
    gt=read(manifest);lookup={x['id']:x for x in gt['images']}
    records=[];metadata=[]
    for im in raw['images']:
        local=lookup[im['id']]
        assert (im['width'],im['height'],im['group_id'])==(local['width'],local['height'],local['group_id'])
        path=manifest.parent/local['gt_yolo']
        truth=load_yolo(path,im['width'],im['height'])
        predictions=tuple(Prediction(Box(*p['xyxy'],class_id=p['class_id']),p['confidence']) for p in im['predictions'])
        records.append(ImageRecord(im['id'],im['width'],im['height'],truth,predictions,im['group_id']))
        metadata.append({k:v for k,v in im.items() if k not in ('predictions','gt_yolo','image_path','auxiliary_beet_ground_truth')})
    return tuple(records),metadata


def locate_predictions(root,job):
    root=Path(root)
    candidates=list(root.glob('runs/'+job+'/attempt*/predictions*/EXPORT_COMPLETE.json'))
    candidates+=list(root.glob(job+'/predictions*/EXPORT_COMPLETE.json'))
    assert len(candidates)==1,('EXPECTED_ONE_EXPORT',job,len(candidates))
    return candidates[0].parent


def replay_binding(c):
    """A completed output is reusable only for the same inputs/code/settings."""
    manifests={};labels={}
    requested=c.get('datasets',['PhenoBench','OD_SugarBeets'])
    needed=['LB']+[ds for ds in ('PhenoBench','OD_SugarBeets') if ds in requested]
    for ds in needed:
        manifest=Path(c['prepared_root'])/ds/'GT_ONLY_manifest.json'
        manifests[ds]=sha(manifest)
        for im in read(manifest)['images']:
            labels[ds+'/'+im['gt_yolo']]=sha(manifest.parent/im['gt_yolo'])
    masks={p.name:sha(p) for p in sorted(Path(c['pheno_semantics']).glob('*.png'))} if 'PhenoBench' in requested else {}
    if 'PhenoBench' in requested:assert masks,'PHENOBENCH_SEMANTICS_REQUIRED_FOR_CROP_EXPOSURE'
    code={name:sha(Path(__file__).parent/name) for name in
          ('wcr.py','risk_control.py','additional.py','reproduce.py','ceilings.py','common.py')}
    code['grouping.py']=sha(ROOT/'data_prep/grouping.py')
    return {'configuration_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),
            'GT_manifest_sha256':manifests,'GT_label_sha256':labels,
            'PhenoBench_semantics_sha256':masks,'analysis_code_sha256':code}


def one_job(c,rotation,model,seed):
    job=f'{rotation}__{model}__seed{seed}';pred=locate_predictions(c['predictions_root'],job)
    export=read(pred/'EXPORT_COMPLETE.json')
    for name,digest in export['files'].items():assert sha(pred/name)==digest,'PREDICTION_HASH_MISMATCH'
    plan_hash=sha(ROOT/'plan/ANALYSIS_PLAN_FROZEN_20261016.md')
    assert plan_hash=='e80795012ce4d4b492a0152a5e89acebe8246703c2fd579f7b4b98f64a7e81f9'
    datasets=c.get('datasets',[rotation,'PhenoBench','OD_SugarBeets'])
    assert datasets and len(set(datasets))==len(datasets) and set(datasets)<=set([rotation,'PhenoBench','OD_SugarBeets'])
    binding=replay_binding(c)
    output=Path(c['output_root'])/job
    if (output/'REPLAY_COMPLETE.json').exists():
        complete=read(output/'REPLAY_COMPLETE.json')
        assert complete['export_sha256']==sha(pred/'EXPORT_COMPLETE.json')
        assert complete['replay_binding']==binding,'REPLAY_INPUTS_OR_CODE_CHANGED'
        for name,digest in complete['files'].items():assert sha(output/name)==digest
        print('VERIFIED_ALREADY_COMPLETE',job);return
    assert not output.exists(),'PARTIAL_REPLAY_PRESERVED_USE_NEW_OUTPUT_ROOT'
    output.mkdir(parents=True);e=c['experiment'];cfg=RiskConfig(e['N_max'],e['alpha'],e['delta'],e['B_count'])
    source,_=load_portable(pred/'source_holdout.json',c['prepared_root'],'source_holdout')
    allrows=[];aux=[]
    for ds in datasets:
        target,meta=load_portable(pred/(ds+'.json'),c['prepared_root'],ds)
        out=output/ds;out.mkdir()
        result=run_experiment(source,target,cfg,threshold_grid(e['grid_step']),e['Ks'],e['repeats'],e['seed'],e['nozzles'],e['schemes'])
        result=augment(result,target,meta,ds,c,out)
        result.update(dataset=ds,checkpoint_sha256=export['checkpoint_sha256'],prediction_sha256=sha(pred/(ds+'.json')))
        save(out/'FULL_RESULTS.json',result)
        rows=[]
        for cell in result['results']:
            kp=[r['K_prime'] for r in cell['runs'] if r.get('K_prime') is not None]
            row={'dataset':ds,**{k:cell[k] for k in ('method','scheme','nozzles','K')},**cell['summary'],
                 'K_prime_mean':statistics.fmean(kp) if kp else None,'K_prime_min':min(kp) if kp else None,'K_prime_max':max(kp) if kp else None}
            rows.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
        csv_write(out/'SUMMARY.csv',rows);allrows.extend(rows)
        aux.extend({'rotation':rotation,'model':model,'seed':seed,'dataset':ds,**row} for row in compute_rows(target,e['nozzles']))
        print('RISK_DATASET_COMPLETE',job,ds,flush=True)
    csv_write(output/'ALL_SUMMARIES.csv',allrows);csv_write(output/'MIN_THRESHOLD.csv',aux)
    result_plots(output,allrows)
    assert replay_binding(c)==binding,'REPLAY_INPUTS_OR_CODE_CHANGED_DURING_RUN'
    save(output/'REPLAY_COMPLETE.json',{'job':job,'export_sha256':sha(pred/'EXPORT_COMPLETE.json'),
       'plan_sha256':plan_hash,'replay_binding':binding,'CPU_only':True,'datasets':datasets,
       'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print('REPLAY_COMPLETE',job,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'analysis/analysis_config.json')
    p.add_argument('--rotation',choices=['bbro','near30','walledgarden'])
    p.add_argument('--model',choices=['asset_detr','rtdetr_r18','yolo26m']);p.add_argument('--seed',type=int,choices=[0,1,2])
    p.add_argument('--datasets',nargs='+',choices=['bbro','near30','walledgarden','PhenoBench','OD_SugarBeets'],
                   help='Replay selected complete target datasets; methods/K/n/repeats are unchanged.')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
    c=read(a.config)
    if a.datasets:c['datasets']=a.datasets
    for r in ([a.rotation] if a.rotation else c['rotations']):
        for m in ([a.model] if a.model else c['models']):
            for s in ([a.seed] if a.seed is not None else c['seeds']):one_job(c,r,m,s)


if __name__=='__main__':main()
