"""Grouped synthetic source/target fixture and the specified 200-repeat experiment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import statistics

from risk_control import RiskConfig, full_spray, run_experiment
from wcr import Box, ImageRecord, Prediction, evaluate, load_manifest, write_json

SEED = 20261013


def generate_grouped_fixture(seed=SEED):
    rng = random.Random(seed)
    datasets = {}
    metadata = {}
    locations = [(25, 15), (75, 45), (155, 75), (240, 105), (325, 15), (410, 75)]
    for name, groups in (("source", 12), ("target", 24)):
        records, group_rows = [], []
        for group in range(groups):
            N = 4 + group % 3 if name == "source" else 2 + group % 5
            count = 8 if name == "source" else 10 + group % 3
            center = (0.72 + 0.025 * (group % 3) if name == "source"
                      else (0.18, 0.32, 0.48, 0.65)[group % 4])
            common_score_shift = rng.uniform(-0.015, 0.015)
            common_x_shift, common_y_shift = rng.uniform(-2, 2), rng.uniform(-2, 2)
            group_id, scores = f"{name}_group_{group:02d}", []
            empty_count = 0
            for image in range(count):
                weed_empty = name == "target" and image % 7 == 0
                empty_count += weed_empty
                labels, predictions = [], []
                for weed in range(N):
                    x, y = locations[weed]
                    x += common_x_shift + rng.uniform(-0.7, 0.7)
                    y += common_y_shift + rng.uniform(-0.7, 0.7)
                    box = Box(x - 4, y - 4, x + 4, y + 4)
                    labels.append(box)
                    score = round(min(0.99, max(0.02, center + common_score_shift +
                                              rng.uniform(-0.08, 0.08))), 3)
                    if not weed_empty:
                        scores.append(score)
                    predictions.append(Prediction(box, score))
                if weed_empty:
                    labels, predictions = [], []
                # False-positive spray adds area but contributes no new GT weed.
                if image % 3 == 0:
                    predictions.append(Prediction(Box(455, 55, 463, 63), 0.5))
                # Beet detections must remain ignored by the spray model.
                labels.append(Box(200, 20, 216, 36, 0))
                predictions.append(Prediction(labels[-1], 0.95))
                records.append(ImageRecord(f"{name}_{group:02d}_{image:02d}", 480, 120,
                                           tuple(labels), tuple(predictions), group_id))
            group_rows.append({"group_id": group_id, "images": count, "nominal_weeds_per_nonempty_image": N,
                               "images_with_weeds": count - empty_count, "weed_empty_images": empty_count,
                               "mean_weed_confidence": statistics.fmean(scores),
                               "within_group_confidence_sd": statistics.stdev(scores)})
        datasets[name] = tuple(records)
        metadata[name] = {"images": len(records), "groups": groups, "groups_detail": group_rows,
            "images_with_weeds": sum(sum(b.class_id == 1 for b in r.ground_truth) > 0 for r in records),
            "weed_empty_images": sum(not any(b.class_id == 1 for b in r.ground_truth) for r in records),
            "between_group_mean_confidence_sd": statistics.stdev(
                row["mean_weed_confidence"] for row in group_rows),
            "mean_within_group_confidence_sd": statistics.fmean(
                row["within_group_confidence_sd"] for row in group_rows)}
    metadata.update({"data_kind": "synthetic_only", "generation_seed": seed,
                     "known_generator_weed_count_bound": 6,
                     "observed_N_max": max(sum(b.class_id == 1 for b in r.ground_truth)
                                           for rows in datasets.values() for r in rows),
                     "group_structure": "shared positions, weed counts and confidence center within each group"})
    return datasets["source"], datasets["target"], metadata


def export_manifest(records, folder: Path):
    labels = folder / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        lines = []
        for box in record.ground_truth:
            cx, cy = (box.x1 + box.x2) / (2 * record.width), (box.y1 + box.y2) / (2 * record.height)
            bw, bh = (box.x2 - box.x1) / record.width, (box.y2 - box.y1) / record.height
            lines.append(f"{box.class_id} {cx:.16g} {cy:.16g} {bw:.16g} {bh:.16g}")
        (labels / f"{record.image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append({"id": record.image_id, "group_id": record.group_id,
                     "width": record.width, "height": record.height,
                     "gt_yolo": f"labels/{record.image_id}.txt",
                     "predictions": [{"xyxy": [p.box.x1, p.box.y1, p.box.x2, p.box.y2],
                                      "class_id": p.box.class_id, "confidence": p.confidence}
                                     for p in record.predictions]})
    path = folder / "manifest.json"
    write_json(path, {"schema_version": 1, "data_kind": "synthetic_only", "images": rows})
    return path


def simulation_checks(result, target, metadata):
    perfect = tuple(ImageRecord(r.image_id, r.width, r.height, r.ground_truth,
                               tuple(Prediction(b, 0.01) for b in r.ground_truth if b.class_id == 1),
                               r.group_id) for r in target)
    perfect_checks = [{"nozzles": n, "threshold": 0.001,
                       "WCR": evaluate(perfect, n, 0.001)["wcr_per_weed"]} for n in (1, 2, 3, 4)]
    full = full_spray(target)
    crc_cells = []
    for cell in result["results"]:
        if cell["method"] == "crc" and cell["scheme"] == "ideal":
            summary = cell["summary"]
            mean = summary["mean_actual_weed_image_miss_rate"]
            crc_cells.append({"K": cell["K"], "nozzles": cell["nozzles"],
                "mean_actual_weed_image_miss_rate": mean,
                "MC_standard_error": summary["weed_image_miss_rate_MC_standard_error"],
                "MC_95pct_normal_interval": summary["weed_image_miss_rate_MC_95pct_normal_interval"],
                "allowed_limit": result["config"]["alpha"],
                "passed": mean is not None and mean <= result["config"]["alpha"],
                "non_fallback_repeats": result["config"]["repeats"] - summary["full_spray_fallback_count"]})
    split_ok, unused_count, grouped_repeats = True, 0, 0
    for split in result["splits"]:
        if "error" in split:
            split_ok = False
            continue
        cal, test, unused = map(set, (split["calibration"], split["test"], split["unused_pool"]))
        split_ok &= len(cal) == split["K"] and not (cal & test or cal & unused or test & unused)
        split_ok &= cal | test | unused == set(range(len(target)))
        split_ok &= split["K_prime"] == sum(any(b.class_id == 1 for b in target[i].ground_truth) for i in cal)
        if split["scheme"] == "grouped":
            grouped_repeats += 1
            pool_groups = {target[i].group_id for i in cal | unused}
            test_groups = {target[i].group_id for i in test}
            split_ok &= not pool_groups & test_groups
            unused_count += len(unused)
    checks = {
        "perfect_prediction_low_threshold": {"passed": all(r["WCR"] == 1 for r in perfect_checks),
                                              "details": perfect_checks},
        "full_spray": {"passed": full["wcr_per_weed"] == 1 and full["spray_area_fraction"] == 1,
                       "metrics": full},
        "monotonicity": {"passed": all(c[field]["passed"] for c in result["monotonicity_checks"]
                                         for field in ("source", "target")),
                         "details": result["monotonicity_checks"]},
        "ideal_CRC_weed_image_miss_rate": {"passed": all(c["passed"] for c in crc_cells),
            "absolute_tolerance_used": 0.0,
            "uncertainty_note": "95% normal intervals quantify repeated-split Monte Carlo error conditional on this fixed synthetic population; not field-risk certificates",
            "nontrivial_K100_checked": all(c["non_fallback_repeats"] > 0 for c in crc_cells if c["K"] == 100),
            "cells": crc_cells},
        "LTT_large_K_non_full_spray": {
            "passed": all(c["summary"]["threshold_distribution"]["threshold_policy_count"] > 0
                          for c in result["results"] if c["method"] == "ltt" and c["K"] == 100),
            "cells": [{"K": c["K"], "nozzles": c["nozzles"], "scheme": c["scheme"],
                       "non_fallback_repeats": c["summary"]["threshold_distribution"]["threshold_policy_count"]}
                      for c in result["results"] if c["method"] == "ltt" and c["K"] in (50, 100)]},
        "grouped_split": {"passed": bool(split_ok), "grouped_repeats_checked": grouped_repeats,
                          "unused_pool_image_occurrences_excluded": unused_count,
                          "target_structure": metadata["target"]},
    }
    checks["all_passed"] = all(value["passed"] for value in checks.values() if isinstance(value, dict))
    checks["all_passed"] &= checks["ideal_CRC_weed_image_miss_rate"]["nontrivial_K100_checked"]
    return checks


def main():
    root = Path(__file__).resolve().parent
    source, target, metadata = generate_grouped_fixture()
    source_path = export_manifest(source, root / "grouped_synthetic" / "source")
    target_path = export_manifest(target, root / "grouped_synthetic" / "target")
    # Run on exported YOLO labels + JSON predictions, exercising actual input code.
    source, target = load_manifest(source_path), load_manifest(target_path)
    config = RiskConfig(metadata["observed_N_max"])
    result = run_experiment(source, target, config)
    result["count_bound_provenance"] = {"kind": "synthetic_observed_max",
        "N_max": metadata["observed_N_max"], "known_generator_bound": 6,
        "real_LB_read": False}
    write_json(root / "experiment_results.json", result)
    compact = {k: v for k, v in result.items() if k not in ("results", "splits")}
    compact["results"] = [{k: v for k, v in c.items() if k != "runs"} for c in result["results"]]
    write_json(root / "experiment_summary.json", compact)
    checks = simulation_checks(result, target, metadata)
    write_json(root / "simulation_checks.json", checks)
    write_json(root / "simulation_metadata.json", metadata)
    print(f"Synthetic source={len(source)}, target={len(target)}, N_max={config.n_max}, main B=1, count B={config.B}")
    print(f"Summary cells={len(result['results'])}; runs={sum(len(c['runs']) for c in result['results'])}")
    print("All revised simulation checks:", checks["all_passed"])
    if not checks["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
