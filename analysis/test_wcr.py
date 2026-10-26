"""Analytical cases + independent integer pixel oracle + curve equivalence."""

import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from wcr import (Box, ImageRecord, Prediction, check_monotonicity, evaluate,
                 fully_covered, load_manifest, load_yolo, merge_intervals,
                 spray_union, sweep, threshold_grid)


def record(gt=(), predictions=(), width=200, height=120, image_id="synthetic"):
    return ImageRecord(image_id, width, height, tuple(gt), tuple(predictions))


def pred(x1, y1, x2, y2, confidence=0.8, class_id=1):
    return Prediction(Box(x1, y1, x2, y2, class_id), confidence)


def pixel_oracle(image, n, threshold):
    """Independent mask implementation, used only on integer-aligned cases.

    Each cell's center is tested against EVERY spray rectangle, without the
    production interval-union or containment functions. Unit-cell area = 1.
    """
    rectangles = []
    h = image.height / n
    for p in image.predictions:
        b = p.box
        if b.class_id != 1 or p.confidence < threshold:
            continue
        cx = (b.x1 + b.x2) / 2
        half_width = max(h, b.x2 - b.x1) / 2
        for k in range(n):
            if b.y1 < (k + 1) * h and b.y2 > k * h:
                rectangles.append((max(0, cx - half_width), k * h,
                                   min(image.width, cx + half_width), (k + 1) * h))
    pixels = {(x, y) for y in range(image.height) for x in range(image.width)
              if any(l <= x + 0.5 < r and bottom <= y + 0.5 < top
                     for l, bottom, r, top in rectangles)}
    hits = 0
    for b in image.ground_truth:
        if b.class_id == 1:
            hits += all((x, y) in pixels for y in range(int(b.y1), int(b.y2))
                        for x in range(int(b.x1), int(b.x2)))
    return hits, len(pixels)


class GeometryTests(unittest.TestCase):
    def test_centered_nominal_width(self):
        image = record(predictions=[pred(70, 10, 90, 20)], width=200, height=100)
        self.assertEqual(spray_union(image, 2, 0.8), (((55.0, 105.0),), ()))
        self.assertEqual(evaluate([image], 2, 0.8)["spray_area_px2"], 2500)

    def test_wide_prediction(self):
        image = record(predictions=[pred(20, 10, 100, 20)], height=100)
        self.assertEqual(spray_union(image, 4, 0), (((20.0, 100.0),), (), (), ()))

    def test_clips_without_shifting(self):
        image = record(predictions=[pred(0, 5, 10, 10), pred(190, 70, 200, 80)], height=100)
        strips = spray_union(image, 2, 0)
        self.assertEqual(strips, (((0.0, 30.0),), ((170.0, 200.0),)))
        self.assertEqual(evaluate([image], 2, 0)["spray_area_px2"], 3000)

    def test_touching_boundary_does_not_trigger_another_strip(self):
        image = record(predictions=[pred(40, 20, 60, 50)], height=100)
        self.assertEqual(spray_union(image, 2, 0), (((25.0, 75.0),), ()))
        image = record(predictions=[pred(40, 50, 60, 80)], height=100)
        self.assertEqual(spray_union(image, 2, 0), ((), ((25.0, 75.0),)))

    def test_crosses_multiple_strips(self):
        image = record(gt=[Box(40, 20, 60, 80)], predictions=[pred(40, 20, 60, 80)], height=100)
        result = evaluate([image], 4, 0)
        self.assertEqual(result["weed_sprayed"], 1)
        self.assertEqual(result["spray_area_px2"], 2500)

    def test_fractional_strip_height(self):
        image = record(gt=[Box(10, 0, 90, 101)],
                       predictions=[pred(10, 0, 90, 101)], width=100, height=101)
        result = evaluate([image], 3, 0)
        self.assertEqual(result["weed_sprayed"], 1)
        self.assertAlmostEqual(result["spray_area_px2"], 8080)

    def test_overlap_duplicate_and_touching_union(self):
        self.assertEqual(merge_intervals([(10, 30), (20, 40), (40, 50), (10, 30)]), ((10, 50),))
        image = record(predictions=[pred(40, 0, 60, 10), pred(40, 0, 60, 10)], height=100)
        self.assertEqual(evaluate([image], 2, 0)["spray_area_px2"], 2500)

    def test_union_can_cover_without_a_single_region_covering(self):
        image = record(gt=[Box(40, 10, 110, 20)],
                       predictions=[pred(40, 10, 60, 20), pred(90, 10, 110, 20)], height=100)
        self.assertEqual(evaluate([image], 2, 0)["weed_sprayed"], 1)

    def test_different_strips_jointly_cover_one_weed(self):
        image = record(gt=[Box(40, 40, 60, 60)],
                       predictions=[pred(40, 40, 60, 50), pred(40, 50, 60, 60)], height=100)
        self.assertEqual(evaluate([image], 2, 0)["weed_sprayed"], 1)
        incomplete = record(gt=image.ground_truth, predictions=image.predictions[:1], height=100)
        self.assertEqual(evaluate([incomplete], 2, 0)["weed_sprayed"], 0)

    def test_partial_coverage_and_positive_gap_fail(self):
        self.assertFalse(fully_covered(Box(20, 10, 80, 20), [((0, 49), (51, 100)), ()], 100))
        self.assertFalse(fully_covered(Box(20, 10, 80, 20), [((0, 50), (50 + 1e-10, 100)), ()], 100))
        self.assertFalse(fully_covered(Box(20, 10, 80, 20), [((20, 79),), ()], 100))

    def test_beet_predictions_ignored_and_off_target_weed_hit_counted(self):
        image = record(gt=[Box(60, 10, 70, 20), Box(120, 10, 130, 20, 0)],
                       predictions=[pred(40, 10, 50, 20), pred(120, 10, 130, 20, 1, 0)], height=100)
        result = evaluate([image], 2, 0)
        self.assertEqual(result["weed_total"], 1)
        self.assertEqual(result["weed_sprayed"], 1)
        self.assertEqual(result["spray_area_px2"], 2500)

    def test_threshold_includes_equality_zero_and_one(self):
        image = record(gt=[Box(40, 10, 60, 20)], predictions=[pred(40, 10, 60, 20, 0)], height=100)
        self.assertEqual(evaluate([image], 2, 0)["weed_sprayed"], 1)
        self.assertEqual(evaluate([image], 2, math.nextafter(0, 1))["weed_sprayed"], 0)
        image = record(gt=image.ground_truth, predictions=[pred(40, 10, 60, 20, 1)], height=100)
        self.assertEqual(evaluate([image], 2, 1)["weed_sprayed"], 1)

    def test_narrow_image_region_can_exceed_full_width(self):
        image = record(gt=[Box(0, 0, 20, 120)], predictions=[pred(0, 0, 20, 120)], width=20)
        result = evaluate([image], 1, 0)
        self.assertEqual(result["spray_area_fraction"], 1)
        self.assertEqual(result["wcr_per_weed"], 1)


class AggregationTests(unittest.TestCase):
    def test_per_weed_per_image_and_area_denominators(self):
        a = record(gt=[Box(10, 1, 20, 5), Box(70, 1, 80, 5)],
                   predictions=[pred(10, 1, 20, 5)], width=100, height=80, image_id="a")
        b = record(gt=[Box(10, 1, 20, 5)], width=200, height=80, image_id="b")
        c = record(predictions=[pred(80, 1, 90, 5)], width=100, height=80, image_id="c")
        result = evaluate([a, b, c], 4, 0)
        self.assertEqual(result["weed_total"], 3)
        self.assertEqual(result["weed_sprayed"], 1)
        self.assertEqual(result["wcr_per_weed"], 1 / 3)
        self.assertEqual(result["wcr_image_mean"], 0.25)
        self.assertEqual(result["images_with_weeds"], 2)
        self.assertEqual(result["spray_area_px2"], 800)
        self.assertEqual(result["image_area_px2"], 32000)
        self.assertEqual(result["spray_area_fraction"], 0.025)
        self.assertIsNone(result["per_image"][2]["wcr"])

    def test_all_empty_weeds_are_undefined_not_perfect(self):
        result = evaluate([record()], 1, 0)
        self.assertIsNone(result["wcr_per_weed"])
        self.assertIsNone(result["wcr_image_mean"])
        self.assertEqual(result["spray_area_fraction"], 0)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([record(), record()], 1, 0)


class InputTests(unittest.TestCase):
    def test_yolo_conversion_and_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("1 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
            boxes = load_yolo(path, 200, 100)
            self.assertEqual(boxes[0], Box(80, 30, 120, 70))
            self.assertEqual(boxes[1].class_id, 0)
            path.write_text("", encoding="utf-8")
            self.assertEqual(load_yolo(path, 200, 100), ())
            with self.assertRaises(FileNotFoundError):
                load_yolo(Path(tmp) / "missing.txt", 200, 100)

    def test_invalid_yolo(self):
        rows = ["2 .5 .5 .2 .2", "1 nan .5 .2 .2", "1 .5 .5 0 .2",
                "1 .01 .5 .5 .5", "1 .5 .5 .2", "1 .5 .5 .2 .2 .8"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            for row in rows:
                with self.subTest(row=row):
                    path.write_text(row, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_yolo(path, 200, 100)

    def test_manifest_requires_explicit_pairing_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            (Path(tmp) / "a.txt").write_text("", encoding="utf-8")
            item = {"id": "a", "width": 200, "height": 100, "gt_yolo": "a.txt", "predictions": []}
            path.write_text(json.dumps({"schema_version": 1, "images": [item]}), encoding="utf-8")
            self.assertEqual(len(load_manifest(path)), 1)
            item.pop("predictions")
            path.write_text(json.dumps({"schema_version": 1, "images": [item]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_invalid_geometry_and_parameters(self):
        invalid = [lambda: Box(0, 0, 0, 1), lambda: Box(0, 0, math.inf, 1),
                   lambda: Box(0, 0, 1, 1, 2), lambda: pred(0, 0, 1, 1, -0.1),
                   lambda: pred(0, 0, 1, 1, True),
                   lambda: record(width=0), lambda: record(width=100.0),
                   lambda: record(gt=[Box(-1, 0, 1, 1)]),
                   lambda: evaluate([record()], 5, 0), lambda: evaluate([record()], 1, float("nan")),
                   lambda: evaluate([], 1, 0), lambda: threshold_grid("0")]
        for action in invalid:
            with self.assertRaises(ValueError):
                action()


class CurveAndOracleTests(unittest.TestCase):
    def test_grid_endpoints(self):
        grid = threshold_grid()
        self.assertEqual(len(grid), 1001)
        self.assertEqual((grid[0], grid[-1]), (0, 1))
        self.assertEqual(threshold_grid("0.3"), [0, 0.3, 0.6, 0.9, 1])

    def test_exact_confidence_event_and_just_above(self):
        image = record(gt=[Box(40, 10, 60, 20)], predictions=[pred(40, 10, 60, 20, 0.812345)], height=100)
        result = sweep([image], 2, [0, 0.5, 1])
        by_t = {row["threshold"]: row for row in result["curve"]}
        self.assertEqual(by_t[0.812345]["weed_sprayed"], 1)
        self.assertEqual(by_t[math.nextafter(0.812345, math.inf)]["weed_sprayed"], 0)
        self.assertTrue(result["monotonicity"]["passed"])

    def test_monotonicity_checker_detects_failure(self):
        a = {"threshold": 0, "weed_sprayed": 0, "wcr_image_mean": 0,
             "spray_area_px2": 0, "image_area_px2": 100}
        b = {**a, "threshold": 1, "weed_sprayed": 1, "wcr_image_mean": 1, "spray_area_px2": 10}
        self.assertFalse(check_monotonicity([a, b])["passed"])

    def test_160_integer_cases_against_independent_pixel_oracle(self):
        rng = random.Random(20261012)
        for case in range(40):
            gt, predictions = [], []
            for _ in range(rng.randrange(7)):
                x = rng.randrange(0, 21)
                y = rng.randrange(0, 21)
                gt.append(Box(x, y, x + rng.randrange(1, min(5, 25 - x)),
                              y + rng.randrange(1, min(5, 25 - y)), rng.randrange(2)))
            for _ in range(rng.randrange(8)):
                # Even x and even widths make spray x edges integral for H=24.
                x, y = rng.randrange(11) * 2, rng.randrange(21)
                w = rng.randrange(1, min(3, (24 - x) // 2 + 1)) * 2
                h = rng.randrange(1, min(5, 25 - y))
                predictions.append(pred(x, y, x + w, y + h,
                                        rng.choice([0, 0.2, 0.5, 0.8, 1]), rng.randrange(2)))
            image = record(gt, predictions, width=24, height=24, image_id=str(case))
            for n in (1, 2, 3, 4):
                threshold = rng.choice([0, 0.2, 0.5, 0.8, 1])
                with self.subTest(case=case, n=n):
                    expected_hits, expected_area = pixel_oracle(image, n, threshold)
                    result = evaluate([image], n, threshold)
                    self.assertEqual(result["weed_sprayed"], expected_hits)
                    self.assertEqual(result["spray_area_px2"], expected_area)

    def test_cached_sweep_matches_direct_evaluation_at_every_point(self):
        rng = random.Random(1183277)
        images = []
        for i in range(12):
            gt = [Box(15, 5, 25, 25), Box(40, 45, 50, 65)] if i % 3 else []
            predictions = []
            for j in range(10):
                x, y = rng.uniform(0, 80), rng.uniform(0, 80)
                predictions.append(pred(x, y, x + 10, y + 10,
                                        rng.choice([0, 0.123456, 0.5, 0.812345, 1]), j % 2))
            images.append(record(gt, predictions, width=100, height=100 + i, image_id=str(i)))
        for n in (1, 2, 3, 4):
            result = sweep(images, n, threshold_grid("0.01"))
            self.assertTrue(result["monotonicity"]["passed"])
            for row in result["curve"]:
                direct = evaluate(images, n, row["threshold"])
                for key in ("weed_total", "weed_sprayed", "wcr_per_weed", "wcr_image_mean",
                            "image_area_px2"):
                    self.assertEqual(row[key], direct[key])
                self.assertAlmostEqual(row["spray_area_px2"], direct["spray_area_px2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
