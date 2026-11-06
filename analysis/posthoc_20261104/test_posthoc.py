from fractions import Fraction
import hashlib
import math
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from risk_control import CurveDataset, RiskConfig, draw_split, select_threshold
from wcr import Box, ImageRecord, Prediction, threshold_grid
from posthoc_20261104.core import (select_naive_m, draw_nested_grouped,
                                  nested_seed, split_overlap, mc_summary)


def record(i, count=1, confidence=.5, predicted=True):
    boxes = tuple(Box(10+j, 1, 12+j, 3) for j in range(count))
    predictions = tuple(Prediction(b, confidence) for b in boxes) if predicted else ()
    return ImageRecord(str(i), 100, 40, boxes, predictions, 'g'+str(i//10))


class NaiveMChecks(unittest.TestCase):
    def test_exact_boundary_largest_grid_threshold(self):
        rs=[record(i, confidence=.8) for i in range(9)]+[record(9, confidence=.3)]
        ds=CurveDataset(rs,4)
        choice=select_naive_m(ds,tuple(range(10)),RiskConfig(147),threshold_grid())
        self.assertEqual(choice.threshold,.8)
        self.assertEqual(choice.calibration_loss_mean,.1)

    def test_weed_image_mean_different_from_count_weighting(self):
        rs=[record(0,count=10,predicted=False)]+[record(i) for i in range(1,10)]
        ds=CurveDataset(rs,4);ix=tuple(range(10));c=RiskConfig(147)
        self.assertEqual(select_naive_m(ds,ix,c,threshold_grid()).threshold,.5)
        self.assertEqual(select_threshold(ds,ix,'naive',c,threshold_grid()).mode,'unavailable')

    def test_empty_positive_calibration_falls_back(self):
        ds=CurveDataset([record(0,count=0)],4)
        self.assertTrue(select_naive_m(ds,(0,),RiskConfig(147),threshold_grid()).fallback)

    def test_no_feasible_threshold_is_actual_full_spray(self):
        ds=CurveDataset([record(0,predicted=False)],4)
        s=select_naive_m(ds,(0,),RiskConfig(147),threshold_grid())
        self.assertEqual(s.mode,'full_spray');self.assertTrue(s.fallback)
        self.assertEqual(ds.metrics((0,),s)['spray_area_fraction'],1)

    def test_no_crc_sample_size_correction(self):
        ds=CurveDataset([record(i) for i in range(5)],4);ix=tuple(range(5));c=RiskConfig(147)
        self.assertEqual(select_naive_m(ds,ix,c,threshold_grid()).threshold,.5)
        self.assertTrue(select_threshold(ds,ix,'crc',c,threshold_grid()).fallback)

    def test_excludes_zero_weed_images(self):
        ds=CurveDataset([record(0,predicted=False)]+[record(i,count=0) for i in range(1,11)],4)
        self.assertTrue(select_naive_m(ds,tuple(range(11)),RiskConfig(147),threshold_grid()).fallback)

    def test_rejects_duplicate_calibration_indices(self):
        with self.assertRaises(ValueError):
            select_naive_m(CurveDataset([record(0)],4),(0,0),RiskConfig(147),threshold_grid())


class NestedSplitChecks(unittest.TestCase):
    def setUp(self):
        self.records=[SimpleNamespace(image_id=str(i),group_id='g%02d'%(i//13)) for i in range(260)]

    def test_seed_has_no_K_or_configuration(self):
        expected=int.from_bytes(hashlib.sha256(b'20261013:D006:1').digest()[:8],'big')
        self.assertEqual(nested_seed(20261013,1),expected)
        self.assertEqual(draw_nested_grouped(self.records,1),draw_nested_grouped(self.records,1))

    def test_common_test_and_nested_entire_pools_are_disjoint(self):
        d=draw_nested_grouped(self.records,2)
        for K,pool in d.pools.items():
            self.assertEqual(len(d.calibrations[K]),K)
            self.assertTrue(all(v==0 for v in split_overlap(self.records,d.calibrations[K],pool,d.test).values()))
        self.assertEqual(set(d.test),set(range(260))-set(d.pools[100]))

    def test_unequal_group_sizes_use_first_sufficient_prefix(self):
        rs=[SimpleNamespace(image_id=str(i),group_id='g'+str(i//70)) for i in range(210)]
        d=draw_nested_grouped(rs,5)
        self.assertEqual(len(d.pools[50]),70);self.assertEqual(len(d.pools[100]),140)

    def test_empty_test_is_not_redrawn_or_replaced(self):
        rs=[SimpleNamespace(image_id=str(i),group_id='one') for i in range(110)]
        self.assertEqual(draw_nested_grouped(rs,1).test,())

    def test_rng_call_order_is_explicit(self):
        d=draw_nested_grouped(self.records,3)
        rng=random.Random(nested_seed(20261013,3));order=sorted({r.group_id for r in self.records});rng.shuffle(order)
        self.assertEqual(tuple(order),d.group_order)
        for K in (5,10,20,50,100):
            self.assertEqual(tuple(sorted(rng.sample(list(d.pools[K]),K))),d.calibrations[K])

    def test_original_K_dependent_draws_can_overlap_K100_test(self):
        from risk_control import repetition_seed
        found=False
        for rep in range(20):
            a=draw_split(self.records,20,'grouped',random.Random(repetition_seed(20261013,'grouped',20,rep)))
            b=draw_split(self.records,100,'grouped',random.Random(repetition_seed(20261013,'grouped',100,rep)))
            found |= bool(set(a.calibration)&set(b.test))
        self.assertTrue(found)


class MCChecks(unittest.TestCase):
    def test_SE_uses_sample_SD_and_sqrt_200(self):
        r=mc_summary([0,.2]*100)
        self.assertAlmostEqual(r['mean_actual_weed_image_miss_rate'],.1)
        self.assertAlmostEqual(r['M_MC_standard_error'],math.sqrt(.01*200/199)/math.sqrt(200))
        self.assertTrue(r['mean_within_alpha_plus_minus_2SE'])

    def test_missing_M_not_replaced_by_zero_or_denominator_200(self):
        r=mc_summary([.1]*199+[None])
        self.assertEqual(r['valid_M_repeats'],199);self.assertIsNone(r['M_MC_standard_error'])

    def test_zero_SE_boundary_is_included(self):
        r=mc_summary([.1]*200)
        self.assertTrue(r['mean_within_alpha_plus_minus_2SE'])


class GeometryAdapterChecks(unittest.TestCase):
    def test_array_grid_matches_original_event_queries_and_metrics(self):
        from posthoc_20261104.engine import build_arrays,ArrayDataset
        from risk_control import Selection
        rs=[record(0,confidence=.50002),record(1,count=3,confidence=.001),record(2,count=0),record(3,predicted=False)]
        data=build_arrays(rs,[{'id':r.image_id} for r in rs])
        for n in (1,2,3,4):
            old=CurveDataset(rs,n);new=ArrayDataset(rs,data,n)
            for t in threshold_grid(.001):
                self.assertEqual(old.counts_at((0,1,2,3),t),new.counts_at((0,1,2,3),t))
                self.assertEqual(old.rate_sum_at((0,1,2,3),t),new.rate_sum_at((0,1,2,3),t))
            for s in (Selection('threshold',0,False,'test'),Selection('threshold',.5,False,'test'),Selection('full_spray',None,True,'test')):
                a,b=old.metrics((0,1,2,3),s),new.metrics((0,1,2,3),s)
                for key in a:self.assertEqual(a[key],b[key])

    def test_array_selection_equals_original_CRC_and_naive_M(self):
        from posthoc_20261104.engine import build_arrays,ArrayDataset
        rs=[record(i,confidence=.8 if i<38 else .3) for i in range(40)]
        data=build_arrays(rs,[{'id':r.image_id} for r in rs])
        for n in (1,2,3,4):
            old=CurveDataset(rs,n);new=ArrayDataset(rs,data,n);ix=tuple(range(40));c=RiskConfig(147)
            self.assertEqual(select_threshold(old,ix,'crc',c,threshold_grid(.001)),select_threshold(new,ix,'crc',c,threshold_grid(.001)))
            self.assertEqual(select_naive_m(old,ix,c,threshold_grid(.001)),select_naive_m(new,ix,c,threshold_grid(.001)))

    def test_crop_adapter_matches_original_per_draw_ratio(self):
        import numpy as np
        from PIL import Image
        from additional import CropCurve,crop_metrics
        from posthoc_20261104.engine import build_arrays,ArrayDataset
        from risk_control import Selection
        with tempfile.TemporaryDirectory() as td:
            rs=[record(0),record(1,predicted=False)];metadata=[{'id':r.image_id,'file_name':r.image_id+'.png'} for r in rs]
            integral=np.zeros((41,101),dtype=np.int64);integral[1:,1:]=np.ones((40,100),dtype=np.int64).cumsum(0).cumsum(1)
            for r in rs:Image.fromarray(np.ones((40,100),dtype=np.uint8)).save(Path(td)/(r.image_id+'.png'))
            data=build_arrays(rs,metadata,td)
            for n in (1,2,3,4):
                curves=[CropCurve(r,n,integral) for r in rs];ds=ArrayDataset(rs,data,n)
                for s in (Selection('threshold',.5,False,'test'),Selection('full_spray',None,True,'test')):
                    a,b=crop_metrics(curves,(0,1),s),ds.crop((0,1),s)
                    for k in b:self.assertEqual(a[k],b[k])


if __name__=='__main__':unittest.main()
