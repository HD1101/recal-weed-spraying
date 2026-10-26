"""Continuous-coordinate Darbyshire spray geometry; Python 3.9+, standard library.

Class 0 = sugar beet, class 1 = weed. A box is xyxy in original image pixels.
Only positive-area intersections trigger strips. Threshold filtering is >=.
See README.md for source evidence and boundary/numerical conventions.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

EPS = 1e-9  # Pixel-coordinate arithmetic tolerance, not a coverage percentage.
NOZZLES = (1, 2, 3, 4)
Interval = tuple[float, float]


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def check_threshold(threshold: float) -> float:
    threshold = finite_number(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    return threshold


def check_nozzles(n: int) -> None:
    if type(n) is not int or n not in NOZZLES:
        raise ValueError("nozzles must be one of 1, 2, 3, 4")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 1

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))
        if type(self.class_id) is not int or self.class_id not in (0, 1):
            raise ValueError("class_id must be integer 0 (beet) or 1 (weed)")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have strictly positive width and height")


@dataclass(frozen=True)
class Prediction:
    box: Box
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise ValueError("prediction box must be a Box")
        object.__setattr__(self, "confidence", check_threshold(self.confidence))


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    width: int
    height: int
    ground_truth: tuple[Box, ...]
    predictions: tuple[Prediction, ...]
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image_id must be a nonempty string")
        if self.group_id is not None and (not isinstance(self.group_id, str) or not self.group_id):
            raise ValueError("group_id must be a nonempty string when supplied")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("image width and height must be integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image width and height must be positive")
        object.__setattr__(self, "ground_truth", tuple(self.ground_truth))
        object.__setattr__(self, "predictions", tuple(self.predictions))
        boxes = list(self.ground_truth)
        for prediction in self.predictions:
            if not isinstance(prediction, Prediction):
                raise ValueError("predictions must contain Prediction objects")
            boxes.append(prediction.box)
        for box in boxes:
            if not isinstance(box, Box):
                raise ValueError("ground_truth must contain Box objects")
            if (box.x1 < -EPS or box.y1 < -EPS or
                    box.x2 > self.width + EPS or box.y2 > self.height + EPS):
                raise ValueError(f"{self.image_id}: box outside original image")


def load_yolo(path: Path, width: int, height: int) -> tuple[Box, ...]:
    """Read required YOLO txt; an existing empty file means no labeled objects."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("image dimensions must be positive integers")
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5 or parts[0] not in ("0", "1"):
                raise ValueError("expected class 0/1 followed by cx cy width height")
            cx, cy, bw, bh = (float(value) for value in parts[1:])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, bw, bh)):
                raise ValueError("YOLO coordinates must be finite and in [0, 1]")
            x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
            y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
            if x1 < -EPS or y1 < -EPS or x2 > width + EPS or y2 > height + EPS:
                raise ValueError("YOLO box extends outside image")
            boxes.append(Box(max(0.0, x1), max(0.0, y1), min(width, x2),
                             min(height, y2), int(parts[0])))
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
    return tuple(boxes)


def load_manifest(path: Path) -> tuple[ImageRecord, ...]:
    """Load explicit per-image pairing; never infer missing predictions or labels."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("manifest schema_version must be integer 1")
    if not isinstance(raw.get("images"), list) or not raw["images"]:
        raise ValueError("manifest must contain a nonempty images list")
    records = []
    for item in raw["images"]:
        if not isinstance(item, dict):
            raise ValueError("each image entry must be an object")
        try:
            image_id, width, height = item["id"], item["width"], item["height"]
            label_path = Path(item["gt_yolo"])
            if label_path.suffix.lower() != ".txt":
                raise ValueError("gt_yolo must name a .txt file")
            if not label_path.is_absolute():
                label_path = path.parent / label_path
            if not isinstance(item["predictions"], list):
                raise ValueError("predictions must be an explicit list, possibly []")
            predictions = []
            for pred in item["predictions"]:
                coords = pred["xyxy"]
                if not isinstance(coords, list) or len(coords) != 4:
                    raise ValueError("prediction xyxy must contain four pixel coordinates")
                predictions.append(Prediction(Box(*coords, class_id=pred["class_id"]),
                                              pred["confidence"]))
            records.append(ImageRecord(image_id, width, height,
                                       load_yolo(label_path, width, height), tuple(predictions),
                                       item.get("group_id")))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid image entry: {exc}") from exc
    validate_records(records)
    return tuple(records)


def validate_records(records: Sequence[ImageRecord]) -> None:
    if not records:
        raise ValueError("dataset must contain at least one image")
    if any(not isinstance(record, ImageRecord) for record in records):
        raise ValueError("dataset entries must be ImageRecord objects")
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("duplicate image IDs would double-count observations")


def merge_intervals(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """Exact 1-D union; touching intervals merge, positive gaps never merge."""
    result: list[list[float]] = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return tuple((left, right) for left, right in result)


def prediction_regions(record: ImageRecord, box: Box, n: int) -> Iterable[tuple[int, Interval]]:
    """Centered nominal spray width, independently cropped at image borders."""
    strip_height = record.height / n
    center = (box.x1 + box.x2) / 2
    width = max(strip_height, box.x2 - box.x1)
    interval = (max(0.0, center - width / 2), min(record.width, center + width / 2))
    for strip in range(n):
        bottom, top = strip * strip_height, (strip + 1) * strip_height
        if max(box.y1, bottom) < min(box.y2, top):
            yield strip, interval


def spray_union(record: ImageRecord, n: int, threshold: float) -> tuple[tuple[Interval, ...], ...]:
    check_nozzles(n)
    threshold = check_threshold(threshold)
    strips: list[list[Interval]] = [[] for _ in range(n)]
    for prediction in record.predictions:
        if prediction.box.class_id == 1 and prediction.confidence >= threshold:
            for strip, interval in prediction_regions(record, prediction.box, n):
                strips[strip].append(interval)
    return tuple(merge_intervals(intervals) for intervals in strips)


def fully_covered(box: Box, strips: Sequence[Sequence[Interval]], height: int) -> bool:
    """Containment in the UNION; several regions may jointly cover one weed."""
    strip_height = height / len(strips)
    for strip, intervals in enumerate(strips):
        if max(box.y1, strip * strip_height) < min(box.y2, (strip + 1) * strip_height):
            if not any(left <= box.x1 + EPS and right >= box.x2 - EPS
                       for left, right in intervals):
                return False
    return True


@dataclass(frozen=True)
class ImageMetrics:
    image_id: str
    weed_total: int
    weed_sprayed: int
    wcr: float | None
    spray_area_px2: float
    image_area_px2: int
    area_fraction: float


def image_metrics(record: ImageRecord, strips: Sequence[Sequence[Interval]]) -> ImageMetrics:
    weeds = [box for box in record.ground_truth if box.class_id == 1]
    hits = sum(fully_covered(box, strips, record.height) for box in weeds)
    area = math.fsum(right - left for intervals in strips for left, right in intervals)
    area *= record.height / len(strips)
    image_area = record.width * record.height
    return ImageMetrics(record.image_id, len(weeds), hits, hits / len(weeds) if weeds else None,
                        area, image_area, area / image_area)


def aggregate(metrics: Sequence[ImageMetrics]) -> dict:
    total = sum(metric.weed_total for metric in metrics)
    hits = sum(metric.weed_sprayed for metric in metrics)
    image_area = sum(metric.image_area_px2 for metric in metrics)
    area = math.fsum(metric.spray_area_px2 for metric in metrics)
    eligible = [metric.wcr for metric in metrics if metric.wcr is not None]
    return {
        "images": len(metrics), "images_with_weeds": len(eligible),
        "weed_total": total, "weed_sprayed": hits,
        "wcr_per_weed": hits / total if total else None,
        "wcr_image_mean": math.fsum(eligible) / len(eligible) if eligible else None,
        "spray_area_px2": area, "image_area_px2": image_area,
        "spray_area_fraction": area / image_area,
    }


def evaluate(records: Sequence[ImageRecord], n: int, threshold: float) -> dict:
    validate_records(records)
    check_nozzles(n)
    threshold = check_threshold(threshold)
    metrics = [image_metrics(record, spray_union(record, n, threshold)) for record in records]
    return {"nozzles": n, "threshold": threshold, **aggregate(metrics),
            "per_image": [asdict(metric) for metric in metrics]}


def threshold_grid(step: str = "0.001") -> list[float]:
    """Decimal-generated grid including both endpoints; no threshold selection."""
    try:
        delta = Decimal(step)
    except Exception as exc:
        raise ValueError("grid step must be a decimal number") from exc
    if not delta.is_finite() or not Decimal(0) < delta <= Decimal(1):
        raise ValueError("grid step must lie in (0, 1]")
    points = int(Decimal(1) / delta)
    if points > 1_000_000:
        raise ValueError("grid exceeds one million intervals")
    return sorted({float(delta * i) for i in range(points + 1)} | {0.0, 1.0})


class ImageCurve:
    """Cache exact confidence events, avoiding recomputation at every grid point."""

    def __init__(self, record: ImageRecord, n: int):
        groups: dict[float, list[Box]] = {}
        for pred in record.predictions:
            if pred.box.class_id == 1:
                groups.setdefault(pred.confidence, []).append(pred.box)
        strips: list[tuple[Interval, ...]] = [() for _ in range(n)]
        self.empty = image_metrics(record, strips)
        states: dict[float, ImageMetrics] = {}
        for confidence in sorted(groups, reverse=True):
            for box in groups[confidence]:
                for strip, interval in prediction_regions(record, box, n):
                    strips[strip] = merge_intervals((*strips[strip], interval))
            states[confidence] = image_metrics(record, strips)
        self.confidences = sorted(states)
        self.states = [states[confidence] for confidence in self.confidences]

    def at(self, threshold: float) -> ImageMetrics:
        index = bisect_left(self.confidences, threshold)
        return self.states[index] if index < len(self.states) else self.empty


def check_monotonicity(curve: Sequence[dict]) -> dict:
    """For increasing t, sprayed weed counts and union area cannot increase."""
    violations = []
    for previous, current in zip(curve, curve[1:]):
        if current["threshold"] <= previous["threshold"]:
            raise ValueError("thresholds must be strictly increasing")
        count_ok = current["weed_sprayed"] <= previous["weed_sprayed"]
        image_mean_ok = (current["wcr_image_mean"] is None or
                         current["wcr_image_mean"] <= previous["wcr_image_mean"] + 1e-15)
        # Area uses a relative tolerance for accumulated floating-point arithmetic.
        area_ok = current["spray_area_px2"] <= previous["spray_area_px2"] + max(
            EPS, previous["image_area_px2"] * 1e-12)
        if not count_ok or not area_ok or not image_mean_ok:
            violations.append({"from": previous["threshold"], "to": current["threshold"],
                               "weed_count_ok": count_ok, "area_ok": area_ok,
                               "image_mean_ok": image_mean_ok})
    return {"passed": not violations, "violations": violations,
            "direction": "nonincreasing as confidence threshold increases"}


def sweep(records: Sequence[ImageRecord], n: int, thresholds: Iterable[float],
          include_events: bool = True) -> dict:
    validate_records(records)
    check_nozzles(n)
    grid = {check_threshold(threshold) for threshold in thresholds}
    if not grid:
        raise ValueError("threshold grid must not be empty")
    grid_count = len(grid)
    if include_events:
        for record in records:
            for prediction in record.predictions:
                if prediction.box.class_id == 1:
                    grid.add(prediction.confidence)
                    if prediction.confidence < 1:
                        grid.add(math.nextafter(prediction.confidence, math.inf))
    curves = [ImageCurve(record, n) for record in records]
    result = [{"threshold": threshold, **aggregate([curve.at(threshold) for curve in curves])}
              for threshold in sorted(grid)]
    return {"nozzles": n, "grid_points_requested": grid_count,
            "grid_points_evaluated": len(result), "exact_events_included": include_events,
            "monotonicity": check_monotonicity(result), "curve": result,
            "threshold_selection": {"status": "DIAGNOSTIC_SWEEP_ONLY",
                                    "entrypoint": "risk_control.py"}}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "sweep"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--nozzles", type=int, nargs="+", choices=NOZZLES, default=list(NOZZLES))
        cmd.add_argument("--output", type=Path, required=True)
        if command == "evaluate":
            cmd.add_argument("--threshold", type=float, required=True)
        else:
            cmd.add_argument("--grid-step", default="0.001")
    args = parser.parse_args()
    try:
        records = load_manifest(args.manifest)
        if args.command == "evaluate":
            results = [evaluate(records, n, args.threshold) for n in args.nozzles]
        else:
            grid = threshold_grid(args.grid_step)
            results = [sweep(records, n, grid) for n in args.nozzles]
        output = {"schema_version": 1, "geometry": "continuous_xyxy_centered_clipped",
                  "weed_class_id": 1, "results": results}
        write_json(args.output, output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
