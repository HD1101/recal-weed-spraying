"""Threshold selection and repeated holdouts for paper 3; standard library only.

Main CRC and LTT control average miss rate on weed-containing images (B=1).
The former signed-count CRC remains crc_count. LTT uses the paper's hybrid
Hoeffding-Bentkus p-value and a fixed ascending grid, stopping at first failure.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import random
import statistics
from typing import Sequence

from wcr import (ImageCurve, ImageMetrics, ImageRecord, aggregate, check_nozzles,
                 check_threshold, finite_number, load_manifest, threshold_grid,
                 validate_records, write_json)

METHODS = ("source", "naive", "crc", "crc_count", "ltt", "full_spray")
SCHEMES = ("ideal", "grouped")


@dataclass(frozen=True)
class RiskConfig:
    n_max: int
    alpha: float = 0.1
    delta: float = 0.05
    B: float | None = None

    def __post_init__(self):
        if type(self.n_max) is not int or self.n_max <= 0:
            raise ValueError("n_max must be a positive integer, fixed before calibration")
        alpha = finite_number(self.alpha, "alpha")
        delta = finite_number(self.delta, "delta")
        if not 0 < alpha < 1 or not 0 < delta < 1:
            raise ValueError("alpha and delta must lie in (0, 1)")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "delta", delta)
        default_B = float((1 - Fraction(str(alpha))) * self.n_max)
        B = default_B if self.B is None else finite_number(self.B, "B")
        if B < default_B:
            raise ValueError("B is smaller than (1-alpha)*n_max; declared loss bound is invalid")
        object.__setattr__(self, "B", B)

    @property
    def lower_bound(self):
        return -self.alpha * self.n_max

    @property
    def loss_range(self):
        return self.B - self.lower_bound


@dataclass(frozen=True)
class Selection:
    mode: str  # threshold / full_spray / unavailable
    threshold: float | None
    fallback: bool
    reason: str
    calibration_weeds: int = 0
    calibration_missed: int = 0
    calibration_loss_mean: float | None = None
    crc_corrected_risk: float | None = None
    p_value: float | None = None
    tests_attempted: int = 0
    first_failed_threshold: float | None = None
    calibration_images_with_weeds: int = 0
    loss_object: str = "weed_image_miss_rate"
    risk_bound_B: float | None = None


def full_spray_metrics(record: ImageRecord) -> ImageMetrics:
    weeds = sum(box.class_id == 1 for box in record.ground_truth)
    area = record.width * record.height
    # Empty-GT WCR remains mathematically undefined, as in the geometry module.
    return ImageMetrics(record.image_id, weeds, weeds, 1.0 if weeds else None,
                        float(area), area, 1.0)


def full_spray(records: Sequence[ImageRecord]) -> dict:
    validate_records(records)
    result = aggregate([full_spray_metrics(record) for record in records])
    result.update({"weed_image_mean_miss_rate": 0.0 if result["images_with_weeds"] else None,
                   "weed_image_rate_sum_exact": [0, 1]})
    return result


def weed_count_max(records: Sequence[ImageRecord]) -> int:
    """Compute N_max from a caller-supplied complete LB label manifest."""
    validate_records(records)
    maximum = max(sum(box.class_id == 1 for box in record.ground_truth) for record in records)
    if maximum == 0:
        raise ValueError("N_max is zero: count-CRC bound requires a dataset containing weeds")
    return maximum


class CurveDataset:
    """Compute geometry once, query confidence events during every resample."""

    def __init__(self, records: Sequence[ImageRecord], n: int):
        validate_records(records)
        check_nozzles(n)
        self.records = tuple(records)
        self.curves = tuple(ImageCurve(record, n) for record in records)
        self.nozzles = n
        self.counts = tuple(curve.empty.weed_total for curve in self.curves)
        self.monotonicity = self._check_monotonicity()

    def _check_monotonicity(self):
        transitions = 0
        for curve in self.curves:
            states = curve.states + [curve.empty]
            for previous, current in zip(states, states[1:]):
                transitions += 1
                if current.weed_sprayed > previous.weed_sprayed:
                    raise ValueError("nonmonotone WCR curve")
                if current.spray_area_px2 > previous.spray_area_px2 + max(
                        1e-9, previous.image_area_px2 * 1e-12):
                    raise ValueError("nonmonotone spray-area curve")
        return {"passed": True, "image_curves": len(self.curves), "transitions_checked": transitions}

    def counts_at(self, indices: Sequence[int], threshold: float):
        total = sum(self.counts[i] for i in indices)
        hits = sum(self.curves[i].at(threshold).weed_sprayed for i in indices)
        return total, total - hits

    def rate_sum_at(self, indices: Sequence[int], threshold: float) -> Fraction:
        # Pool numerators only within equal denominators; then sum exact fractions.
        # Empty images contribute neither loss nor sample size.
        missed_by_N = Counter()
        for i in indices:
            metric = self.curves[i].at(threshold)
            if metric.weed_total:
                missed_by_N[metric.weed_total] += metric.weed_total - metric.weed_sprayed
        return sum((Fraction(missed, N) for N, missed in missed_by_N.items()), Fraction(0))

    def metrics(self, indices: Sequence[int], selection: Selection):
        if selection.mode == "unavailable":
            return None
        if selection.mode == "full_spray":
            metrics = [full_spray_metrics(self.records[i]) for i in indices]
        else:
            metrics = [self.curves[i].at(selection.threshold) for i in indices]
        result = aggregate(metrics)
        eligible = [m for m in metrics if m.weed_total > 0]
        rate_sum = sum((Fraction(m.weed_total - m.weed_sprayed, m.weed_total)
                        for m in eligible), Fraction(0))
        result.update({"weed_image_mean_miss_rate": float(rate_sum / len(eligible)) if eligible else None,
                       "weed_image_rate_sum_exact": [rate_sum.numerator, rate_sum.denominator]})
        return result

    def validate_bound(self, config: RiskConfig):
        if max(self.counts) > config.n_max:
            raise ValueError("observed weed count exceeds the declared n_max; do not silently re-estimate B")


def checked_grid(grid: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted({check_threshold(t) for t in grid}))
    if not result or result[0] != 0 or result[-1] != 1:
        raise ValueError("selection grid must include endpoints 0 and 1")
    return result


def largest_feasible(grid, feasible):
    """Find the last true in a nonincreasing Boolean sequence; no interpolation."""
    left, right, selected = 0, len(grid) - 1, None
    while left <= right:
        mid = (left + right) // 2
        if feasible(grid[mid]):
            selected = grid[mid]
            left = mid + 1
        else:
            right = mid - 1
    return selected


def loss_sum_exact(missed: int, weeds: int, config: RiskConfig) -> Fraction:
    return Fraction(missed) - Fraction(str(config.alpha)) * weeds


@lru_cache(maxsize=256)
def binomial_cdf_table(K: int, alpha: float) -> tuple[float, ...]:
    """Stable binomial lower tails without SciPy; log PMFs avoid factorial overflow."""
    if type(K) is not int or K <= 0 or not 0 < alpha < 1:
        raise ValueError("positive integer K and alpha in (0,1) required")
    probabilities = [math.exp(math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)
                             + k * math.log(alpha) + (K - k) * math.log1p(-alpha))
                     for k in range(K + 1)]
    return tuple(min(1.0, math.fsum(probabilities[:k + 1])) for k in range(K + 1))


def hb_p_value(loss_mean: Fraction | float, K_prime: int, alpha: float = 0.1) -> float:
    """LTT Proposition 1, Eq (2): min(exp(-K' KL), e*BinomCDF(ceil(K' R)))."""
    if type(K_prime) is not int or K_prime <= 0 or not 0 < alpha < 1:
        raise ValueError("positive K_prime and alpha in (0,1) required")
    mean = loss_mean if isinstance(loss_mean, Fraction) else Fraction(str(loss_mean))
    if not 0 <= mean <= 1:
        raise ValueError("normalized empirical risk must lie in [0,1]")
    risk = min(float(mean), alpha)
    kl = ((risk * math.log(risk / alpha) if risk else 0.0) +
          (1 - risk) * math.log((1 - risk) / (1 - alpha)))
    hoeffding = min(1.0, math.exp(-K_prime * max(0.0, kl)))
    exact_sum = K_prime * mean
    cutoff = (exact_sum.numerator + exact_sum.denominator - 1) // exact_sum.denominator
    bentkus = min(1.0, math.e * binomial_cdf_table(K_prime, alpha)[cutoff])
    return min(hoeffding, bentkus)


def fixed_sequence(grid, p_value_at, delta: float):
    """Predeclared conservative-to-aggressive order, stopping on first failure."""
    selected, p_selected, attempted, first_failed = None, None, 0, None
    for threshold in grid:
        attempted += 1
        p = p_value_at(threshold)
        if p > delta:
            first_failed = threshold
            break
        selected, p_selected = threshold, p
    return selected, p_selected, attempted, first_failed


def select_threshold(dataset: CurveDataset, indices: Sequence[int], method: str,
                     config: RiskConfig, grid: Sequence[float]) -> Selection:
    if method not in METHODS:
        raise ValueError("unknown selection method")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("calibration indices must be nonempty and distinct")
    grid = checked_grid(grid)
    K = len(indices)
    total = sum(dataset.counts[i] for i in indices)
    K_prime = sum(dataset.counts[i] > 0 for i in indices)
    base = {"calibration_weeds": total, "calibration_images_with_weeds": K_prime}
    loss_object = ("signed_weed_counts" if method == "crc_count" else
                   "per_weed_miss_fraction" if method in ("source", "naive") else
                   "weed_image_miss_rate")
    base["loss_object"] = loss_object
    if method == "full_spray":
        return Selection("full_spray", None, False, "full_spray_baseline",
                         calibration_loss_mean=0.0 if K_prime else None, **base)

    def signed_at(t):
        weeds, missed = dataset.counts_at(indices, t)
        return loss_sum_exact(missed, weeds, config)

    corrected = p_value = first_failed = None
    attempted = 0
    alpha = Fraction(str(config.alpha))
    if method in ("source", "naive"):
        if total == 0:
            return Selection("unavailable", None, False, "empirical_WCR_undefined_no_weeds", **base)
        threshold = largest_feasible(grid, lambda t: signed_at(t) <= 0)
        if threshold is None:
            # The design specifies fallback for CRC and LTT only. Do not invent
            # an empirical-method fallback when no threshold achieves its target.
            return Selection("unavailable", None, False, "no_empirical_threshold_reaches_target", **base)
    elif method == "crc_count":
        dataset.validate_bound(config)
        B = Fraction(str(config.B))
        base["risk_bound_B"] = config.B
        threshold = largest_feasible(grid, lambda t: signed_at(t) + B <= 0)
        if threshold is None:
            return Selection("full_spray", None, True, "no_count_CRC_threshold_passes",
                             calibration_loss_mean=-config.alpha * total / K, **base)
        corrected = float((signed_at(threshold) + B) / (K + 1))
    elif method == "crc":
        base["risk_bound_B"] = 1.0
        if K_prime == 0 or Fraction(1, K_prime + 1) > alpha:
            return Selection("full_spray", None, True, "insufficient_K_prime_for_CRC",
                             calibration_loss_mean=0.0 if K_prime else None, **base)
        threshold = largest_feasible(grid, lambda t: (dataset.rate_sum_at(indices, t) + 1) <=
                                     alpha * (K_prime + 1))
        if threshold is None:
            return Selection("full_spray", None, True, "no_image_CRC_threshold_passes",
                             calibration_loss_mean=0.0, **base)
        corrected = float((dataset.rate_sum_at(indices, threshold) + 1) / (K_prime + 1))
    else:  # LTT
        base["risk_bound_B"] = 1.0
        if K_prime == 0:
            return Selection("full_spray", None, True, "no_weed_images_for_LTT", **base)
        # Cache p-values for repeated exact rates; keep the complete fixed grid.
        cached_rates = {}
        def p_at(t):
            mean = dataset.rate_sum_at(indices, t) / K_prime
            if mean not in cached_rates:
                cached_rates[mean] = hb_p_value(mean, K_prime, config.alpha)
            return cached_rates[mean]
        threshold, p_value, attempted, first_failed = fixed_sequence(
            grid, p_at, config.delta)
        if threshold is None:
            return Selection("full_spray", None, True, "first_LTT_test_fails",
                             calibration_loss_mean=0.0, p_value=p_at(grid[0]), tests_attempted=attempted,
                             first_failed_threshold=first_failed, **base)
    weeds, missed = dataset.counts_at(indices, threshold)
    loss_mean = (float(signed_at(threshold)) / K if method == "crc_count" else
                 missed / total if method in ("source", "naive") else
                 float(dataset.rate_sum_at(indices, threshold) / K_prime))
    return Selection("threshold", threshold, False, "largest_feasible_fixed_grid_threshold",
                     calibration_missed=missed, calibration_loss_mean=loss_mean,
                     crc_corrected_risk=corrected, p_value=p_value, tests_attempted=attempted,
                     first_failed_threshold=first_failed, **base)


@dataclass(frozen=True)
class Split:
    calibration: tuple[int, ...]
    test: tuple[int, ...]
    unused_pool: tuple[int, ...] = ()
    pool_groups: tuple[str, ...] = ()


def draw_split(records: Sequence[ImageRecord], K: int, scheme: str, rng: random.Random) -> Split:
    if type(K) is not int or not 0 < K < len(records):
        raise ValueError("K must be positive and leave a nonempty test set")
    if scheme == "ideal":
        calibration = tuple(sorted(rng.sample(range(len(records)), K)))
        chosen = set(calibration)
        return Split(calibration, tuple(i for i in range(len(records)) if i not in chosen))
    if scheme != "grouped":
        raise ValueError("scheme must be ideal or grouped")
    groups: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        if record.group_id is None:
            raise ValueError("grouped split requires explicit group_id for every image")
        groups.setdefault(record.group_id, []).append(i)
    order = sorted(groups)
    rng.shuffle(order)
    pool, pool_groups = [], []
    for group in order:
        pool.extend(groups[group])
        pool_groups.append(group)
        if len(pool) >= K:
            break
    pool_set = set(pool)
    test = tuple(i for i in range(len(records)) if i not in pool_set)
    if not test:
        raise ValueError("grouped calibration pool consumed all groups; no test group remains")
    calibration = tuple(sorted(rng.sample(pool, K)))
    chosen = set(calibration)
    unused = tuple(sorted(pool_set - chosen))
    return Split(calibration, test, unused, tuple(pool_groups))


def repetition_seed(seed: int, scheme: str, K: int, repeat: int) -> int:
    # Deterministic across Python runs; never use process-randomized hash().
    key = f"{seed}:{scheme}:{K}:{repeat}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return None
    result = {}
    for name, p in (("min", 0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("max", 1)):
        index = p * (len(ordered) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        result[name] = ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return result


def summarize_runs(runs, alpha):
    valid = [run for run in runs if run["test_metrics"] is not None]
    valid_wcr = [run for run in valid if run["test_metrics"]["wcr_per_weed"] is not None]
    wcrs = [run["test_metrics"]["wcr_per_weed"] for run in valid_wcr]
    miss_rates = [1 - value for value in wcrs]
    thresholds = [run["selection"]["threshold"] for run in valid
                  if run["selection"]["mode"] == "threshold"]
    alpha_exact = Fraction(str(alpha))
    success = sum(Fraction(run["test_metrics"]["weed_total"] -
                           run["test_metrics"]["weed_sprayed"]) <=
                  alpha_exact * run["test_metrics"]["weed_total"] for run in valid_wcr)
    mean_miss = statistics.fmean(miss_rates) if miss_rates else None
    se = statistics.stdev(miss_rates) / math.sqrt(len(miss_rates)) if len(miss_rates) > 1 else None
    valid_image_risks = [run for run in valid if run["test_metrics"]["weed_image_mean_miss_rate"] is not None]
    image_rates = [run["test_metrics"]["weed_image_mean_miss_rate"] for run in valid_image_risks]
    image_success = sum(Fraction(*run["test_metrics"]["weed_image_rate_sum_exact"]) <=
                        alpha_exact * run["test_metrics"]["images_with_weeds"]
                        for run in valid_image_risks)
    image_mean = statistics.fmean(image_rates) if image_rates else None
    image_se = statistics.stdev(image_rates) / math.sqrt(len(image_rates)) if len(image_rates) > 1 else None
    K_primes = [run["K_prime"] for run in runs if run.get("K_prime") is not None]
    return {
        "repeats_requested": len(runs), "valid_metrics_repeats": len(valid),
        "valid_WCR_repeats": len(wcrs), "unavailable_repeats": len(runs) - len(valid),
        "undefined_WCR_repeats": len(valid) - len(wcrs),
        "mean_actual_WCR": statistics.fmean(wcrs) if wcrs else None,
        "valid_weed_image_risk_repeats": len(image_rates),
        "mean_actual_weed_image_miss_rate": image_mean,
        "weed_image_miss_rate_le_alpha_count": image_success,
        "weed_image_miss_rate_le_alpha_fraction": image_success / len(image_rates) if image_rates else None,
        "weed_image_miss_rate_MC_standard_error": image_se,
        "weed_image_miss_rate_MC_95pct_normal_interval": [max(0.0, image_mean - 1.96 * image_se),
                                                         min(1.0, image_mean + 1.96 * image_se)]
                                                        if image_se is not None else None,
        "K_prime_distribution": {"quantiles": quantiles(K_primes),
                                 "frequencies": [{"K_prime": k, "count": c}
                                                 for k, c in sorted(Counter(K_primes).items())]},
        "mean_actual_miss_fraction": mean_miss,
        "WCR_ge_target_count": success,
        "WCR_ge_target_fraction": success / len(wcrs) if wcrs else None,
        "mean_spray_area_fraction": statistics.fmean(run["test_metrics"]["spray_area_fraction"]
                                                     for run in valid) if valid else None,
        "mean_actual_signed_loss": statistics.fmean(run["test_metrics"]["mean_signed_loss"]
                                                    for run in valid) if valid else None,
        "threshold_distribution": {"threshold_policy_count": len(thresholds),
            "quantiles": quantiles(thresholds),
            "frequencies": [{"threshold": t, "count": count}
                            for t, count in sorted(Counter(thresholds).items())],
            "full_spray_policy_count": sum(run["selection"]["mode"] == "full_spray" for run in valid)},
        "full_spray_fallback_count": sum(run["selection"]["fallback"] for run in valid),
        "miss_fraction_MC_standard_error": se,
        "miss_fraction_MC_95pct_normal_interval": [max(0.0, mean_miss - 1.96 * se),
                                                     min(1.0, mean_miss + 1.96 * se)]
                                                   if se is not None else None,
    }


def run_experiment(source: Sequence[ImageRecord], target: Sequence[ImageRecord],
                   config: RiskConfig, grid: Sequence[float] | None = None,
                   Ks=(5, 10, 20, 50, 100), repeats=200, seed=20261013,
                   nozzles=(1, 2, 3, 4), schemes=SCHEMES) -> dict:
    validate_records(source)
    validate_records(target)
    if {r.image_id for r in source} & {r.image_id for r in target}:
        raise ValueError("source holdout and target image IDs must be disjoint")
    if type(repeats) is not int or repeats <= 0 or type(seed) is not int:
        raise ValueError("repeats must be positive integer; seed must be integer")
    grid = checked_grid(threshold_grid() if grid is None else grid)
    if not Ks or any(type(K) is not int or not 0 < K < len(target) for K in Ks):
        raise ValueError("all K values must leave test images")
    if not nozzles or not schemes or any(s not in SCHEMES for s in schemes):
        raise ValueError("nonempty valid nozzles and schemes are required")
    if "grouped" in schemes and any(r.group_id is None for r in target):
        raise ValueError("grouped scheme requires explicit group IDs")
    results, all_splits, checks, source_choices = [], [], [], []
    for n in nozzles:
        source_data, target_data = CurveDataset(source, n), CurveDataset(target, n)
        checks.append({"nozzles": n, "source": source_data.monotonicity,
                       "target": target_data.monotonicity})
        source_selection = select_threshold(source_data, tuple(range(len(source))), "source", config, grid)
        source_choices.append({"nozzles": n, **asdict(source_selection)})
        for scheme in schemes:
            for K in Ks:
                runs_by_method = {method: [] for method in METHODS}
                for repeat in range(repeats):
                    draw_seed = repetition_seed(seed, scheme, K, repeat)
                    try:
                        split = draw_split(target, K, scheme, random.Random(draw_seed))
                    except ValueError as exc:
                        # Do not silently redraw: resampling until valid can bias the experiment.
                        for method in METHODS:
                            runs_by_method[method].append({"repeat": repeat, "seed": draw_seed, "K_prime": None,
                                "selection": asdict(Selection("unavailable", None, False, str(exc))),
                                "test_metrics": None})
                        if n == nozzles[0]:
                            all_splits.append({"scheme": scheme, "K": K, "repeat": repeat,
                                               "seed": draw_seed, "error": str(exc)})
                        continue
                    if n == nozzles[0]:
                        all_splits.append({"scheme": scheme, "K": K, "repeat": repeat,
                                           "seed": draw_seed,
                                           "K_prime": sum(target_data.counts[i] > 0 for i in split.calibration),
                                           **asdict(split)})
                    for method in METHODS:
                        try:
                            if method == "crc_count":
                                target_data.validate_bound(config)
                            selection = source_selection if method == "source" else select_threshold(
                                target_data, split.calibration, method, config, grid)
                        except ValueError as exc:
                            if method != "crc_count":
                                raise
                            selection = Selection("unavailable", None, False, str(exc),
                                                  loss_object="signed_weed_counts", risk_bound_B=config.B)
                        metrics = target_data.metrics(split.test, selection)
                        if metrics is not None:
                            metrics["mean_signed_loss"] = float(loss_sum_exact(
                                metrics["weed_total"] - metrics["weed_sprayed"],
                                metrics["weed_total"], config)) / len(split.test)
                        runs_by_method[method].append({"repeat": repeat, "seed": draw_seed,
                            "K_prime": sum(target_data.counts[i] > 0 for i in split.calibration),
                            "target_calibration_images": len(split.calibration),
                            "selection_calibration_images": len(source) if method == "source" else
                                (0 if method == "full_spray" else len(split.calibration)),
                            "test_images": len(split.test),
                            "unused_pool_images": len(split.unused_pool),
                            "selection": asdict(selection), "test_metrics": metrics})
                for method, runs in runs_by_method.items():
                    results.append({"method": method, "K": K, "scheme": scheme, "nozzles": n,
                                    "summary": summarize_runs(runs, config.alpha), "runs": runs})
    return {"schema_version": 3, "revision": "weed-image-risk_HB_K100",
            "config": {**asdict(config), "crc_image_B": 1.0, "crc_count_B": config.B,
                "count_loss_lower_bound": config.lower_bound, "count_loss_range": config.loss_range,
                "Ks": list(Ks), "repeats": repeats, "seed": seed,
                "nozzles": list(nozzles), "schemes": list(schemes),
                "threshold_grid": list(grid), "grid_is_predeclared": True,
                "ltt_bound": "Hoeffding-Bentkus, LTT Proposition 1 Equation 2",
                "ltt_loss_range": [0, 1], "ltt_uses_K_prime": True,
                "ltt_stops_at_first_failure": True},
            "source_images": len(source), "target_images": len(target),
            "source_selections": source_choices, "monotonicity_checks": checks,
            "splits": all_splits, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source holdout manifest")
    parser.add_argument("--target", type=Path, required=True, help="new field manifest with group_id")
    bound = parser.add_mutually_exclusive_group(required=True)
    bound.add_argument("--n-max", type=int, help="explicit bound; use 6 only for this synthetic fixture")
    bound.add_argument("--lb-manifest", type=Path, help="complete LB label manifest for its actual weed-count maximum")
    parser.add_argument("--B-count", "--B", dest="B", type=float, help="signed-count CRC bound only; main CRC B=1")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--nozzles", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--grid-step", default="0.001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.lb_manifest is not None:
            lb_records = load_manifest(args.lb_manifest)
            n_max = weed_count_max(lb_records)
            provenance = {"kind": "complete_LB_manifest_observed_max", "manifest": str(args.lb_manifest),
                          "images": len(lb_records), "N_max": n_max}
        else:
            n_max = args.n_max
            provenance = {"kind": "explicit_caller_value", "N_max": n_max}
        config = RiskConfig(n_max, args.alpha, args.delta, args.B)
        result = run_experiment(load_manifest(args.source), load_manifest(args.target), config,
                                threshold_grid(args.grid_step), args.K, args.repeats, args.seed,
                                args.nozzles, args.schemes)
        result["count_bound_provenance"] = provenance
        write_json(args.output, result)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}: {len(result['results'])} summary cells")


if __name__ == "__main__":
    main()
