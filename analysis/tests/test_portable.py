import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'));sys.path.insert(0,str(ROOT/'data_prep'))
import numpy as np
from PIL import Image
from wcr import Box,Prediction,ImageRecord,threshold_grid
from risk_control import RiskConfig,run_experiment
from additional import relation,pixel_rect_count,union_crop_count,augment
from reproduce import load_portable,one_job
from common import save,sha
from data_prep.pheno_boxes import pheno_boxes
from data_prep.prepare import parse_xml
from data_prep.reconstruct_splits import verify
from check_numbers import check


class PortableChecks(unittest.TestCase):
    def test_dates_all_four_relations(self):
        dates=['05-15','06-05']
        self.assertEqual(relation(dates,'05-01'),'earlier_than_all_calibration')
        self.assertEqual(relation(dates,'06-06'),'later_than_all_calibration')
        self.assertEqual(relation(dates,'05-15'),'same_day_as_calibration')
        self.assertEqual(relation(dates,'05-26'),'other')
        self.assertEqual(relation([],None),'date_unresolved')

    def test_pixel_boundary_and_union_no_double_count(self):
        mask=np.ones((4,4),dtype=np.int64);ii=np.zeros((5,5),dtype=np.int64)
        ii[1:,1:]=mask.cumsum(0).cumsum(1)
        self.assertEqual(pixel_rect_count(ii,0,2,0,4),8)
        self.assertEqual(union_crop_count(ii,[[(0,2)],[(0,2)]],4),8)
        self.assertEqual(pixel_rect_count(ii,.5,1.5,.5,1.5),1)

    def test_pheno_instance_semantics_and_plus_one(self):
        instances=np.array([[1,1,2],[1,1,2],[0,0,0]],dtype=np.uint16)
        semantics=np.array([[1,3,0],[1,3,0],[0,0,0]],dtype=np.uint8);audit={}
        boxes,bad=pheno_boxes(semantics,instances,1,'synthetic.png',audit)
        self.assertEqual(boxes,[(0,0,0,2,2)])
        self.assertEqual(audit['majority_tie_count'],1)
        self.assertEqual(audit['discarded_majority_background_count'],1)
        self.assertFalse(bad)

    def test_xml_floats_are_not_truncated_or_shifted(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'test.xml'
            p.write_text('<annotation><object><name>weed</name><bndbox><xmin>0.25</xmin><ymin>1.75</ymin><xmax>8.1</xmax><ymax>9.2</ymax></bndbox></object></annotation>')
            self.assertEqual(parse_xml(p),[(1,.25,1.75,8.1,9.2)])

    def test_export_old_paths_ignored_exact_id_pairing(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);folder=td/'LB';(folder/'labels').mkdir(parents=True)
            (folder/'labels/im.txt').write_text('1 0.5 0.5 0.2 0.2\n')
            im={'id':'im','file_name':'im.png','width':10,'height':10,'group_id':'g',
                'gt_yolo':'labels/im.txt','predictions':[]}
            (folder/'GT_ONLY_manifest.json').write_text(json.dumps({'schema_version':1,'images':[im]}))
            exported={**im,'gt_yolo':'historical-unavailable-label.txt','predictions':[{'xyxy':[4,4,6,6],'confidence':.5,'class_id':1}]}
            p=td/'pred.json';p.write_text(json.dumps({'schema_version':1,'images':[exported]}))
            records,meta=load_portable(p,td,'bbro')
            self.assertEqual(records[0].ground_truth,(Box(4,4,6,6),))
            self.assertEqual(records[0].predictions[0].confidence,.5)
            self.assertNotIn('gt_yolo',meta[0])

    def test_synthetic_risk_rq3_crop_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);out=td/'out';out.mkdir();records=[];meta=[]
            for i in range(12):
                date=['05-15','05-26','06-05'][i//4];name=f'{date}_{i}.png'
                b=Box(2,2,4,4);records.append(ImageRecord(name,8,8,(b,),(Prediction(b,.9),),f'{date}/g{i//2}'))
                meta.append({'id':name,'file_name':name,'width':8,'height':8,'group_id':f'{date}/g{i//2}'})
                Image.fromarray(np.ones((8,8),dtype=np.uint8)).save(td/name)
            c={'experiment':{'nozzles':[1,2,3,4],'Ks':[5],'schemes':['ideal','grouped'],'repeats':2},
               'date_overrides':{},'pheno_semantics':str(td),'crop_methods':['source','crc','perfect_detection_reference','full_spray'],
               'crop_pixel_rule':'pixel_center_in_half_open_union'}
            source=[ImageRecord('source_'+r.image_id,r.width,r.height,r.ground_truth,r.predictions,'source_'+r.group_id) for r in records[:6]]
            b=run_experiment(source,records,RiskConfig(147,.1,.05,132.3),threshold_grid(.1),[5],2,20261013,[1,2,3,4],['ideal','grouped'])
            result=augment(b,records,meta,'PhenoBench',c,out)
            self.assertEqual(len(result['results']),56)
            with (out/'CROP_EXPOSURE_SUMMARY.csv').open() as f:rows=list(csv.DictReader(f))
            full=[r for r in rows if r['method']=='full_spray']
            self.assertEqual(len(full),8)
            self.assertTrue(all(float(r['crop_exposure_pooled_over_repeats'])==1 for r in full))
            self.assertTrue((out/'RQ3_RUNS.json').is_file())
            self.assertTrue((out/'PHENO_DATE_DIRECTIONS.csv').is_file())

    def test_all_fixed_splits(self):verify()

    def test_three_headline_numbers(self):self.assertEqual(check()['status'],'PASSED')

    def test_portable_job_completion_and_hash_checked_skip(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);prepared=td/'prepared';pred=td/'pred/runs/bbro__asset_detr__seed0/attempt01/predictions01'
            masks=td/'semantics';masks.mkdir();data={}
            for dataset in ['source_holdout','bbro','PhenoBench','OD_SugarBeets']:
                ims=[]
                for i in range(6 if dataset=='source_holdout' else 12):
                    if dataset in ('source_holdout','bbro'):
                        field='near30' if dataset=='source_holdout' else 'bbro'
                        name=f'{field}_{field}_{15+i//4:02d}_05_2021_v_{i//2}_{i}.png'
                        group=f'{field}/2021-05-{15+i//4:02d}/video_{i//2}'
                    elif dataset=='PhenoBench':
                        date=['05-15','05-26','06-05'][i//4];name=f'{date}_{i}.png';group=date
                        Image.fromarray(np.ones((8,8),dtype=np.uint8)).save(masks/name)
                    else:
                        name=f'bonirob_2016-05-{11+i//4:02d}-10-36-32_0_frame{i}.png';group=f'sequence_{i//2}'
                    ident=Path(name).stem
                    im={'id':ident,'file_name':name,'width':8,'height':8,'group_id':group,'gt_yolo':'labels/'+ident+'.txt',
                        'predictions':[{'class_id':1,'confidence':.9,'xyxy':[2,2,4,4]}]}
                    ims.append(im)
                    base=prepared/('LB' if dataset in ('source_holdout','bbro') else dataset)
                    (base/'labels').mkdir(parents=True,exist_ok=True)
                    (base/im['gt_yolo']).write_text('1 0.375 0.375 0.25 0.25\n')
                data[dataset]=ims
                save(pred/(dataset+'.json'),{'schema_version':1,'images':ims})
            save(prepared/'LB/GT_ONLY_manifest.json',{'schema_version':1,'images':data['source_holdout']+data['bbro']})
            for ds in ('PhenoBench','OD_SugarBeets'):
                save(prepared/ds/'GT_ONLY_manifest.json',{'schema_version':1,'images':data[ds]})
            hashes={p.name:sha(p) for p in pred.glob('*.json')}
            save(pred/'EXPORT_COMPLETE.json',{'checkpoint_sha256':'synthetic_checkpoint','files':hashes})
            c={'predictions_root':str(td/'pred'),'prepared_root':str(prepared),'output_root':str(td/'replay'),
               'pheno_semantics':str(masks),'date_overrides':{},'crop_methods':['source','crc','perfect_detection_reference','full_spray'],
               'crop_pixel_rule':'pixel_center_in_half_open_union',
               'experiment':{'N_max':147,'alpha':.1,'delta':.05,'B_count':132.3,'grid_step':.1,'Ks':[5],
                             'repeats':2,'seed':20261013,'nozzles':[1,2,3,4],'schemes':['ideal','grouped']}}
            one_job(c,'bbro','asset_detr',0)
            done=td/'replay/bbro__asset_detr__seed0/REPLAY_COMPLETE.json'
            self.assertTrue(done.is_file());one_job(c,'bbro','asset_detr',0)
            self.assertTrue(all(sha(pred/k)==h for k,h in hashes.items()))
            changed=json.loads(json.dumps(c));changed['experiment']['seed']=20261014
            with self.assertRaisesRegex(AssertionError,'REPLAY_INPUTS_OR_CODE_CHANGED'):
                one_job(changed,'bbro','asset_detr',0)
            # The read interface can replay a complete PhenoBench cell without
            # unrelated target GT; the scientific functions/settings are reused.
            subset=json.loads(json.dumps(c));subset['datasets']=['PhenoBench']
            subset['output_root']=str(td/'pheno-only-replay')
            save(prepared/'LB/GT_ONLY_manifest.json',{'schema_version':1,'images':data['source_holdout']})
            (prepared/'OD_SugarBeets/GT_ONLY_manifest.json').rename(prepared/'OD_SugarBeets/not_requested.json')
            one_job(subset,'bbro','asset_detr',0)
            output=td/'pheno-only-replay/bbro__asset_detr__seed0'
            self.assertTrue((output/'PhenoBench/FULL_RESULTS.json').is_file())
            self.assertFalse((output/'OD_SugarBeets').exists())


if __name__=='__main__':unittest.main()
