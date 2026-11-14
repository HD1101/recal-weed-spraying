"""Check partial coverage against independently reconstructed spray unions."""
import math
import random
import unittest

from run import REPO
from geometry import exact_ninety_percent, partial_events
from wcr import Box, ImageCurve, ImageRecord, Prediction, merge_intervals, prediction_regions


class GeometryChecks(unittest.TestCase):
    def test_exact_boundary(self):
        box = Box(0, 0, 10, 10)
        self.assertTrue(exact_ninety_percent(box, (((0, 9),),), 10))
        self.assertFalse(exact_ninety_percent(box, (((0, 8.99),),), 10))
        self.assertTrue(exact_ninety_percent(box, (((0, 8),), ((0, 10),)), 10))
        self.assertFalse(exact_ninety_percent(box, (((0, 7.99),), ((0, 10),)), 10))

    def test_cached_geometry_matches_direct_unions(self):
        rng = random.Random(20261114)

        def box():
            x, y = rng.randrange(89), rng.randrange(89)
            return Box(x, y, x + rng.randint(1, 11), y + rng.randint(1, 11))

        for _ in range(25):
            record = ImageRecord('sample', 100, 100, tuple(box() for _ in range(10)),
                                 tuple(Prediction(box(), rng.randrange(1, 11) / 10) for _ in range(50)))
            for n in (4, 8, 16):
                scores, hits, areas = partial_events(record, n)
                curve = ImageCurve(record, n)
                for t, h, a in zip(scores, hits, areas):
                    raw = [[] for _ in range(n)]
                    for prediction in record.predictions:
                        if prediction.confidence >= t:
                            for k, interval in prediction_regions(record, prediction.box, n):
                                raw[k].append(interval)
                    strips = tuple(merge_intervals(intervals) for intervals in raw)
                    self.assertEqual(h, sum(exact_ninety_percent(b, strips, 100) for b in record.ground_truth))
                    self.assertTrue(math.isclose(a, curve.at(t).spray_area_px2, rel_tol=1e-14))


if __name__ == '__main__':
    unittest.main()
