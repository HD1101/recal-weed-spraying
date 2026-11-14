"""Sensitivity-analysis geometry, isolated from the published implementation."""

from fractions import Fraction
import math

import numpy as np

from wcr import ImageCurve, ImageRecord, Prediction, merge_intervals, prediction_regions, threshold_grid

# Preserve the exact floating point grid used in the archived formal run.
GRID = tuple(threshold_grid(.001))
GRID_ARRAY = np.asarray(GRID)


def box_strips(box, height, n):
    step = height / n
    return [(k, depth) for k in range(n)
            if (depth := min(box.y2, (k + 1) * step) - max(box.y1, k * step)) > 0]


def overlap_width(box, intervals):
    return math.fsum(max(0., min(box.x2, right) - max(box.x1, left))
                     for left, right in intervals)


def exact_ninety_percent(box, strips, height):
    x1, y1, x2, y2 = map(Fraction, (box.x1, box.y1, box.x2, box.y2))
    step = Fraction(height, len(strips))
    covered = sum(
        max(0, min(y2, (k + 1) * step) - max(y1, k * step)) *
        sum(max(0, min(x2, Fraction(right)) - max(x1, Fraction(left)))
            for left, right in intervals)
        for k, intervals in enumerate(strips)
    )
    return 10 * covered >= 9 * (x2 - x1) * (y2 - y1)


def partial_events(record, n):
    weeds = [box for box in record.ground_truth if box.class_id == 1]
    strips = [()] * n
    groups = {}
    for prediction in record.predictions:
        if prediction.box.class_id == 1:
            groups.setdefault(prediction.confidence, []).append(prediction.box)
    scores = sorted(groups)
    areas = [0.] * len(scores)
    hits = [0] * len(scores)
    by_strip = [[] for _ in range(n)]
    contributions = [[0.] * n for _ in weeds]
    cutoffs = [(b.x2 - b.x1) * (b.y2 - b.y1) * .9 for b in weeds]
    covered = [False] * len(weeds)
    count = 0
    for idx, box in enumerate(weeds):
        for k, depth in box_strips(box, record.height, n):
            by_strip[k].append((idx, depth))
    for j in range(len(scores) - 1, -1, -1):
        new = {}
        for box in groups[scores[j]]:
            for k, interval in prediction_regions(record, box, n):
                new.setdefault(k, []).append(interval)
        affected = set()
        for k, intervals in new.items():
            old = strips[k]
            strips[k] = merge_intervals((*old, *intervals))
            if strips[k] == old:
                continue
            for idx, depth in by_strip[k]:
                if not covered[idx]:
                    contributions[idx][k] = depth * overlap_width(weeds[idx], strips[k])
                    affected.add(idx)
        for idx in affected:
            area = math.fsum(contributions[idx])
            cutoff = cutoffs[idx]
            if abs(area - cutoff) <= cutoff * 1e-12:
                passes = exact_ninety_percent(weeds[idx], strips, record.height)
            else:
                passes = area >= cutoff
            if passes:
                covered[idx] = True
                count += 1
        hits[j] = count
        areas[j] = math.fsum(r - l for intervals in strips for l, r in intervals)
        areas[j] *= record.height / n
    return scores, hits, areas


def geometry_arrays(records, n, partial=False):
    hits = np.empty((len(records), len(GRID)), dtype=np.int16)
    areas = np.empty(hits.shape, dtype=np.float64)
    for i, record in enumerate(records):
        if partial:
            scores, counts, sprayed = partial_events(record, n)
        else:
            curve = ImageCurve(record, n)
            scores = curve.confidences
            counts = [state.weed_sprayed for state in curve.states]
            sprayed = [state.spray_area_px2 for state in curve.states]
        indices = np.searchsorted(scores, GRID_ARRAY, side='left')
        hits[i] = np.asarray([*counts, 0], dtype=np.int16)[indices]
        areas[i] = np.asarray([*sprayed, 0.])[indices]
    if np.any(np.diff(hits, axis=1) > 0):
        raise ValueError('Nonmonotone weed coverage curve')
    return hits, areas


def truth_reference(records, n):
    areas = []
    for record in records:
        predictions = tuple(Prediction(box, 1.) for box in record.ground_truth if box.class_id == 1)
        truth = ImageRecord(record.image_id, record.width, record.height,
                            record.ground_truth, predictions, record.group_id)
        metrics = ImageCurve(truth, n).at(1.)
        if metrics.weed_total != metrics.weed_sprayed:
            raise ValueError('Ground-truth trigger failed to cover its own box')
        areas.append(metrics.spray_area_px2)
    return np.asarray(areas)
