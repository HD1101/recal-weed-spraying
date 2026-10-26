"""Tests for statistical arithmetic, sampling semantics, and reproducibility."""

import math
from pathlib import Path
import random
import tempfile
import unittest

from risk_control import (CurveDataset, RiskConfig, Selection, draw_split, fixed_sequence,
                          full_spray, hb_p_value, binomial_cdf_table, run_experiment, select_threshold,
                          weed_count_max)
from fractions import Fraction
from run_simulation import export_manifest, generate_grouped_fixture
from wcr import Box, ImageRecord, Prediction, evaluate, load_manifest


GRID = [0, 0.3, 0.5, 0.8, 1]


def copies(count, prefix="image", confidence=0.5, group_size=4):
    box = Box(10, 1, 12, 3)
    return tuple(ImageRecord(f"{prefix}_{i}", 100, 40, (box,), (Prediction(box, confidence),),
                             f"{prefix}_group_{i // group_size}") for i in range(count))


class SelectionTests(unittest.TestCase):
    def test_empirical_max_and_90_percent_equality(self):
        records = copies(9, confidence=0.8) + copies(1, prefix="low", confidence=0.3)
        dataset = CurveDataset(records, 4)
        chosen = select_threshold(dataset, list(range(10)), "naive", RiskConfig(1), GRID)
        self.assertEqual(chosen.threshold, 0.8)
        self.assertEqual(chosen.calibration_missed, 1)
        self.assertEqual(chosen.calibration_loss_mean, 0.1)
        crc = select_threshold(dataset, list(range(10)), "crc_count", RiskConfig(1), GRID)
        self.assertEqual(crc.threshold, 0.3)
        self.assertLessEqual(crc.crc_corrected_risk, 0)

    def test_CRC_correction_exact_at_zero_boundary(self):
        dataset = CurveDataset(copies(9), 4)
        chosen = select_threshold(dataset, list(range(9)), "crc_count", RiskConfig(1), GRID)
        self.assertEqual(chosen.threshold, 0.5)
        self.assertEqual(chosen.crc_corrected_risk, 0)

    def test_CRC_no_candidate_falls_back_to_actual_full_image(self):
        dataset = CurveDataset(copies(5), 4)
        chosen = select_threshold(dataset, list(range(5)), "crc_count", RiskConfig(1), GRID)
        self.assertEqual(chosen.mode, "full_spray")
        self.assertTrue(chosen.fallback)
        metrics = dataset.metrics(list(range(5)), chosen)
        self.assertEqual(metrics["wcr_per_weed"], 1)
        self.assertEqual(metrics["spray_area_fraction"], 1)
        self.assertEqual(metrics["weed_image_mean_miss_rate"], 0)
        self.assertLess(evaluate(dataset.records, 4, 0)["spray_area_fraction"], 1)

    def test_B_parameter_changes_CRC_feasibility(self):
        dataset = CurveDataset(copies(10), 4)
        self.assertEqual(select_threshold(dataset, list(range(10)), "crc_count", RiskConfig(1, B=1), GRID).threshold, 0.5)
        self.assertTrue(select_threshold(dataset, list(range(10)), "crc_count", RiskConfig(1, B=1.1), GRID).fallback)

    def test_empirical_no_feasible_threshold_is_explicit_not_invented_fallback(self):
        records = tuple(ImageRecord(r.image_id, r.width, r.height, r.ground_truth, (), r.group_id)
                        for r in copies(10))
        chosen = select_threshold(CurveDataset(records, 4), list(range(10)), "naive", RiskConfig(1), GRID)
        self.assertEqual(chosen.mode, "unavailable")
        self.assertFalse(chosen.fallback)

    def test_empty_calibration_gt_is_undefined_empirical_and_safe_CRC_fallback(self):
        records = (ImageRecord("empty", 100, 40, (), (), "g"),)
        dataset = CurveDataset(records, 4)
        self.assertEqual(select_threshold(dataset, [0], "naive", RiskConfig(1), GRID).mode, "unavailable")
        self.assertTrue(select_threshold(dataset, [0], "crc", RiskConfig(1), GRID).fallback)

    def test_default_bound_and_invalid_config(self):
        config = RiskConfig(6)
        self.assertEqual(config.B, 5.4)
        self.assertAlmostEqual(config.loss_range, 6)
        for kwargs in ({"n_max": 0}, {"n_max": 6, "B": 5}, {"n_max": 1, "alpha": 0},
                       {"n_max": 1, "delta": 1}):
            with self.assertRaises(ValueError):
                RiskConfig(**kwargs)
        records = (ImageRecord("two", 100, 40, (Box(10, 1, 12, 3), Box(40, 1, 42, 3)), (), "g"),)
        with self.assertRaises(ValueError):
            select_threshold(CurveDataset(records, 4), [0], "crc_count", RiskConfig(1), GRID)
        self.assertTrue(select_threshold(CurveDataset(records, 4), [0], "crc", RiskConfig(1), GRID).fallback)

    def test_LTT_HB_zero_risk_endpoint_and_null_boundary(self):
        self.assertAlmostEqual(hb_p_value(0, 50), 0.9 ** 50)
        self.assertEqual(hb_p_value(Fraction(1, 10), 50), 1)
        self.assertEqual(hb_p_value(1, 50), 1)

    def test_LTT_stops_at_first_failure_even_if_later_pvalue_small(self):
        values = {0: 0.01, 0.5: 0.2, 1: 0.001}
        selected, p, attempts, failed = fixed_sequence([0, 0.5, 1], values.__getitem__, 0.05)
        self.assertEqual((selected, p, attempts, failed), (0, 0.01, 2, 0.5))

    def test_LTT_can_select_with_sufficient_calibration_size(self):
        dataset = CurveDataset(copies(200), 4)
        chosen = select_threshold(dataset, list(range(200)), "ltt", RiskConfig(1), GRID)
        self.assertEqual(chosen.threshold, 0.5)
        self.assertEqual(chosen.first_failed_threshold, 0.8)
        self.assertEqual(chosen.tests_attempted, 4)
        self.assertFalse(chosen.fallback)

    def test_HB_small_K_fallback_but_K50_can_select(self):
        dataset = CurveDataset(copies(50), 4)
        for K in (5, 10, 20):
            chosen = select_threshold(dataset, list(range(K)), "ltt", RiskConfig(1), GRID)
            self.assertTrue(chosen.fallback)
            self.assertEqual(chosen.tests_attempted, 1)
        self.assertFalse(select_threshold(dataset, list(range(50)), "ltt", RiskConfig(1), GRID).fallback)

    def test_full_spray_baseline_is_not_counted_as_fallback(self):
        rows = copies(8)
        chosen = select_threshold(CurveDataset(rows, 4), [0, 1], "full_spray", RiskConfig(1), GRID)
        self.assertFalse(chosen.fallback)
        metrics = full_spray(rows)
        self.assertEqual(metrics["wcr_per_weed"], 1)
        self.assertEqual(metrics["spray_area_fraction"], 1)


class RevisedImageRiskTests(unittest.TestCase):
    def test_count_bound_from_complete_manifest_counts_weeds_only(self):
        a=ImageRecord("a",100,40,(Box(10,1,12,3),Box(80,1,82,3,0)),(),"ga")
        b=ImageRecord("b",100,40,(Box(10,1,12,3),Box(80,1,82,3)),(),"gb")
        self.assertEqual(weed_count_max([a,b]),2)
        with self.assertRaises(ValueError):
            weed_count_max([ImageRecord("empty",100,40,(),(),"g")])

    def test_main_CRC_Kprime_minimum_nine_and_exact_boundary(self):
        dataset = CurveDataset(copies(9), 4)
        chosen = select_threshold(dataset, list(range(9)), "crc", RiskConfig(1), GRID)
        self.assertEqual(chosen.threshold, 0.5)
        self.assertEqual(chosen.crc_corrected_risk, 0.1)
        self.assertEqual(chosen.risk_bound_B, 1)
        self.assertTrue(select_threshold(dataset, list(range(8)), "crc", RiskConfig(1), GRID).fallback)

    def test_empty_images_do_not_increase_Kprime_for_CRC_or_LTT(self):
        rows = copies(20) + tuple(ImageRecord(f"empty_{i}", 100, 40, (), (), "empty_group") for i in range(80))
        data = CurveDataset(rows, 4)
        for method in ("crc", "ltt"):
            a = select_threshold(data, list(range(20)), method, RiskConfig(1), GRID)
            b = select_threshold(data, list(range(100)), method, RiskConfig(1), GRID)
            self.assertEqual((a.threshold, a.fallback, a.calibration_loss_mean, a.p_value),
                             (b.threshold, b.fallback, b.calibration_loss_mean, b.p_value))
            self.assertEqual(b.calibration_images_with_weeds, 20)

    def test_K100_with_only_eight_weed_images_still_falls_back(self):
        rows = copies(8) + tuple(ImageRecord(f"empty_{i}", 100, 40, (), (), "eg") for i in range(92))
        chosen = select_threshold(CurveDataset(rows, 4), list(range(100)), "crc", RiskConfig(1), GRID)
        self.assertTrue(chosen.fallback)
        self.assertEqual(chosen.calibration_images_with_weeds, 8)

    def test_main_CRC_uses_mean_rates_not_pooled_weed_counts(self):
        rows = list(copies(15, confidence=0.3))
        boxes = tuple(Box(10 + 15*j, 1, 12 + 15*j, 3) for j in range(6))
        for i in range(85):
            rows.append(ImageRecord(f"dense_{i}", 120, 40, boxes,
                                    tuple(Prediction(b, 0.8) for b in boxes), "dense_group"))
        data = CurveDataset(rows, 4)
        main = select_threshold(data, list(range(100)), "crc", RiskConfig(6), GRID)
        count = select_threshold(data, list(range(100)), "crc_count", RiskConfig(6), GRID)
        self.assertEqual(main.threshold, 0.3)
        self.assertEqual(count.threshold, 0.8)
        self.assertEqual(data.metrics(list(range(100)), count)["weed_image_mean_miss_rate"], 0.15)

    def test_main_B_is_one_even_when_count_B_changes(self):
        data = CurveDataset(copies(10), 4)
        a = select_threshold(data, list(range(10)), "crc", RiskConfig(1), GRID)
        b = select_threshold(data, list(range(10)), "crc", RiskConfig(100), GRID)
        self.assertEqual(a, b)

    def test_HB_matches_independent_combinatorial_binomial_CDF(self):
        for K in (5, 20, 50, 100):
            table = binomial_cdf_table(K, 0.1)
            for k in (0, 1, min(5, K), K):
                reference = sum(math.comb(K, j) * 0.1**j * 0.9**(K-j) for j in range(k+1))
                self.assertAlmostEqual(table[k], reference, places=12)

    def test_HB_fractional_sum_ceil_and_exact_integer_sum(self):
        for rate_sum in (Fraction(1), Fraction(11, 10), Fraction(2), Fraction(21, 10)):
            K, alpha = 50, 0.1
            mean = rate_sum / K
            x = float(mean)
            kl = x*math.log(x/alpha) + (1-x)*math.log((1-x)/(1-alpha))
            cutoff = math.ceil(rate_sum)
            tail = sum(math.comb(K,j)*alpha**j*(1-alpha)**(K-j) for j in range(cutoff+1))
            expected = min(math.exp(-K*kl), math.e*tail)
            self.assertAlmostEqual(hb_p_value(mean,K,alpha), expected, places=12)

    def test_HB_monotone_over_normalized_empirical_risk(self):
        values = [hb_p_value(Fraction(i,1000),100) for i in range(1001)]
        self.assertTrue(all(a <= b + 1e-14 for a,b in zip(values,values[1:])))

    def test_macro_loss_and_micro_WCR_can_have_different_pass_fail(self):
        a = (Box(10,1,12,3), Box(80,1,82,3))
        b = tuple(Box(10 + 20*j,1,12 + 20*j,3) for j in range(8))
        rows = (ImageRecord("a",200,40,a,(Prediction(a[0],0.8),),"g1"),
                ImageRecord("b",200,40,b,tuple(Prediction(x,0.8) for x in b),"g2"),
                ImageRecord("empty",200,40,(),(),"g3"))
        metrics = CurveDataset(rows,4).metrics([0,1,2],Selection("threshold",0.5,False,"unit"))
        self.assertEqual(metrics["wcr_per_weed"],0.9)
        self.assertEqual(metrics["weed_image_mean_miss_rate"],0.25)
        self.assertEqual(metrics["images_with_weeds"],2)

    def test_all_empty_calibration_LTT_falls_back_without_pvalue(self):
        rows = (ImageRecord("empty",100,40,(),(),"g"),)
        result = select_threshold(CurveDataset(rows,4),[0],"ltt",RiskConfig(1),GRID)
        self.assertTrue(result.fallback)
        self.assertIsNone(result.p_value)

    def test_normalized_methods_remain_available_when_count_bound_invalid(self):
        box1,box2=Box(10,1,12,3),Box(80,1,82,3)
        target = tuple(ImageRecord(f"t{i}",100,40,(box1,box2),
                                  (Prediction(box1,0.5),Prediction(box2,0.5)),f"g{i//4}") for i in range(16))
        result=run_experiment(copies(10,prefix="s"),target,RiskConfig(1),GRID,
                              Ks=(10,),repeats=2,nozzles=(4,))
        for cell in result["results"]:
            self.assertEqual(cell["summary"]["unavailable_repeats"],2 if cell["method"]=="crc_count" else 0)

    def test_every_split_and_method_records_target_Kprime(self):
        source,target,_ = generate_grouped_fixture()
        result=run_experiment(source,target,RiskConfig(6),GRID,Ks=(10,100),repeats=3,nozzles=(4,))
        index={(s["scheme"],s["K"],s["repeat"]):s for s in result["splits"]}
        for cell in result["results"]:
            for run in cell["runs"]:
                split=index[cell["scheme"],cell["K"],run["repeat"]]
                expected=sum(any(b.class_id==1 for b in target[i].ground_truth) for i in split["calibration"])
                self.assertEqual(run["K_prime"],expected)
                self.assertEqual(split["K_prime"],expected)
        self.assertTrue(any(s["K_prime"]<s["K"] for s in result["splits"]))


class SplitAndExperimentTests(unittest.TestCase):
    def test_ideal_keeps_all_remaining_images_in_test(self):
        rows = copies(16)
        split = draw_split(rows, 5, "ideal", random.Random(7))
        self.assertEqual((len(split.calibration), len(split.test), len(split.unused_pool)), (5, 11, 0))
        self.assertFalse(set(split.calibration) & set(split.test))

    def test_group_pool_unused_images_do_not_leak_into_test(self):
        rows = copies(16, group_size=8)
        split = draw_split(rows, 5, "grouped", random.Random(7))
        self.assertEqual((len(split.calibration), len(split.test), len(split.unused_pool)), (5, 8, 3))
        cal_groups = {rows[i].group_id for i in split.calibration + split.unused_pool}
        test_groups = {rows[i].group_id for i in split.test}
        self.assertFalse(cal_groups & test_groups)

    def test_missing_groups_and_consuming_all_groups_raise(self):
        rows = copies(16, group_size=8)
        with self.assertRaises(ValueError):
            draw_split(rows, 15, "grouped", random.Random(7))
        no_groups = tuple(ImageRecord(r.image_id, r.width, r.height, r.ground_truth, r.predictions) for r in rows)
        with self.assertRaises(ValueError):
            draw_split(no_groups, 5, "grouped", random.Random(7))

    def test_seed_reproducible_and_shared_across_methods_nozzles(self):
        a = run_experiment(copies(10, prefix="source"), copies(24), RiskConfig(1), GRID,
                           Ks=(5, 10), repeats=3, nozzles=(2, 4), seed=8)
        b = run_experiment(copies(10, prefix="source"), copies(24), RiskConfig(1), GRID,
                           Ks=(5, 10), repeats=3, nozzles=(2, 4), seed=8)
        self.assertEqual(a, b)
        seeds = {}
        for cell in a["results"]:
            for run in cell["runs"]:
                key = (cell["scheme"], cell["K"], run["repeat"])
                seeds.setdefault(key, run["seed"])
                self.assertEqual(seeds[key], run["seed"])
        for cell in a["results"]:
            self.assertEqual(cell["summary"]["valid_metrics_repeats"], 3)

    def test_source_threshold_does_not_use_new_field_calibration(self):
        result = run_experiment(copies(10, prefix="source", confidence=0.8),
                                copies(16, confidence=0.3), RiskConfig(1), GRID,
                                Ks=(5,), repeats=2, nozzles=(4,))
        source_cells = [c for c in result["results"] if c["method"] == "source"]
        self.assertTrue(all(r["selection"]["threshold"] == 0.8 for c in source_cells for r in c["runs"]))
        self.assertTrue(all(c["summary"]["mean_actual_WCR"] == 0 for c in source_cells))

    def test_no_test_group_records_unavailable_without_redrawing(self):
        result = run_experiment(copies(10, prefix="source"), copies(16, group_size=8), RiskConfig(1), GRID,
                                Ks=(15,), repeats=2, nozzles=(4,), schemes=("grouped",))
        self.assertTrue(all(c["summary"]["unavailable_repeats"] == 2 for c in result["results"]))
        self.assertTrue(all(c["summary"]["mean_actual_WCR"] is None for c in result["results"]))

    def test_simulation_group_structure_and_manifest_roundtrip(self):
        source, target, metadata = generate_grouped_fixture()
        self.assertEqual(metadata["observed_N_max"], 6)
        self.assertEqual(metadata["target"]["groups"], 24)
        self.assertGreater(metadata["target"]["between_group_mean_confidence_sd"],
                           metadata["target"]["mean_within_group_confidence_sd"])
        with tempfile.TemporaryDirectory() as tmp:
            path = export_manifest(target[:12], Path(tmp))
            loaded = load_manifest(path)
            self.assertEqual([r.group_id for r in loaded], [r.group_id for r in target[:12]])
            self.assertEqual([sum(b.class_id == 1 for b in r.ground_truth) for r in loaded],
                             [sum(b.class_id == 1 for b in r.ground_truth) for r in target[:12]])
            for n in (1, 2, 3, 4):
                self.assertEqual(evaluate(loaded, n, 0)["weed_sprayed"],
                                 evaluate(target[:12], n, 0)["weed_sprayed"])

    def test_perfect_prediction_extremely_low_threshold_all_nozzles(self):
        records = copies(8, confidence=0.01)
        for n in (1, 2, 3, 4):
            self.assertEqual(evaluate(records, n, 0.001)["wcr_per_weed"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
