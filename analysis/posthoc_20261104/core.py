"""Posthoc definitions; imports the original geometry/statistical primitives.

No original method is monkey-patched, registered, or modified. The new module
is invoked explicitly and writes only new posthoc output files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
import random
import statistics

from risk_control import Selection, checked_grid, largest_feasible

KS = (5, 10, 20, 50, 100)


def select_naive_m(dataset, indices, config, grid):
    """Largest grid t with exactly the CRC loss but no finite-sample correction."""
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("nonempty distinct calibration indices required")
    grid = checked_grid(grid)
    total = sum(int(dataset.counts[i]) for i in indices)
    kp = sum(dataset.counts[i] > 0 for i in indices)
    base = dict(calibration_weeds=total, calibration_images_with_weeds=int(kp),
                loss_object="weed_image_miss_rate", risk_bound_B=1.0)
    if not kp:
        return Selection("full_spray", None, True, "naive_M_no_weed_images", **base)
    alpha = Fraction(str(config.alpha))
    chosen = largest_feasible(grid, lambda t: dataset.rate_sum_at(indices, t) <= alpha * int(kp))
    if chosen is None:
        return Selection("full_spray", None, True, "naive_M_no_feasible_threshold", **base)
    _, missed = dataset.counts_at(indices, chosen)
    return Selection("threshold", chosen, False, "naive_M_largest_empirically_feasible",
                     calibration_missed=int(missed),
                     calibration_loss_mean=float(dataset.rate_sum_at(indices, chosen) / int(kp)),
                     **base)


@dataclass(frozen=True)
class NestedDraw:
    seed: int
    repeat_one_based: int
    group_order: tuple[str, ...]
    calibrations: dict[int, tuple[int, ...]]
    pools: dict[int, tuple[int, ...]]
    pool_groups: dict[int, tuple[str, ...]]
    test: tuple[int, ...]


def nested_seed(seed: int, repeat_one_based: int):
    if repeat_one_based < 1:
        raise ValueError("D006 repeat numbering starts at one")
    key = f"{seed}:D006:{repeat_one_based}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def draw_nested_grouped(records, repeat_one_based, seed=20261013, Ks=KS):
    """Revision 1: one shuffle, ordered sample calls, K100-pool complement.

    Empty tests are retained, not redrawn. Both image and group separation are
    asserted for every K, including unused calibration-pool images.
    """
    if tuple(sorted(set(Ks))) != tuple(Ks) or 100 not in Ks:
        raise ValueError("unique increasing K values including 100 required")
    groups = {}
    for i, record in enumerate(records):
        if record.group_id is None:
            raise ValueError("explicit groups required")
        groups.setdefault(record.group_id, []).append(i)
    if max(Ks) > len(records):
        raise ValueError("insufficient total images for K100 calibration")
    draw_seed = nested_seed(seed, repeat_one_based)
    rng = random.Random(draw_seed)
    order = sorted(groups)
    rng.shuffle(order)
    pools, calibrations, pool_groups = {}, {}, {}
    for K in Ks:
        pool, chosen_groups = [], []
        for group in order:
            pool.extend(groups[group]); chosen_groups.append(group)
            if len(pool) >= K:
                break
        pools[K] = tuple(pool)
        pool_groups[K] = tuple(chosen_groups)
        calibrations[K] = tuple(sorted(rng.sample(pool, K)))
    outside = set(range(len(records))) - set(pools[100])
    test = tuple(i for i in range(len(records)) if i in outside)
    test_groups = {records[i].group_id for i in test}
    previous = set()
    for K in Ks:
        assert previous <= set(pools[K]), "D006_POOLS_NOT_NESTED"
        assert not (outside & set(pools[K])), "D006_POOL_TEST_IMAGE_OVERLAP"
        assert not (test_groups & set(pool_groups[K])), "D006_POOL_TEST_GROUP_OVERLAP"
        assert set(calibrations[K]) <= set(pools[K])
        previous = set(pools[K])
    return NestedDraw(draw_seed, repeat_one_based, tuple(order), calibrations,
                      pools, pool_groups, test)


def split_overlap(records, calibration, calibration_pool, test):
    test_set = set(test)
    test_groups = {records[i].group_id for i in test}
    return {
        "calibration_image_overlap": len(set(calibration) & test_set),
        "calibration_pool_image_overlap": len(set(calibration_pool) & test_set),
        "calibration_group_overlap": len({records[i].group_id for i in calibration} & test_groups),
        "calibration_pool_group_overlap": len({records[i].group_id for i in calibration_pool} & test_groups),
    }


def mc_summary(values, alpha=0.1, requested=200):
    valid = [float(v) for v in values if v is not None and math.isfinite(v)]
    mean = statistics.fmean(valid) if valid else None
    sd = statistics.stdev(valid) if len(valid) > 1 else None
    se = sd / math.sqrt(requested) if len(valid) == requested and sd is not None else None
    return {"repeats_requested": requested, "valid_M_repeats": len(valid),
            "mean_actual_weed_image_miss_rate": mean, "M_sample_SD": sd,
            "M_MC_standard_error": se,
            "M_minus_2SE": mean - 2 * se if se is not None else None,
            "M_plus_2SE": mean + 2 * se if se is not None else None,
            "mean_within_alpha_plus_minus_2SE": abs(mean-alpha) <= 2*se if se is not None else None,
            "MC_SE_denominator": requested if se is not None else None}


def crop_summary(runs):
    values = [r["crop"]["crop_exposure"] for r in runs
              if r.get("crop") and r["crop"]["crop_exposure"] is not None]
    return {"mean_crop_exposure": statistics.fmean(values) if values else None,
            "valid_crop_exposure_repeats": len(values)}


def digest_indices(records, indices):
    return hashlib.sha256("\n".join(records[i].image_id for i in indices).encode()).hexdigest()
