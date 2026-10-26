# Fixed splits and groups

Each `train.csv`, `val.csv`, and `target.csv` contains identifiers and metadata
only, never box annotations. `val` is the 200-image source holdout. All models
and training seeds share one fixed split per rotation. Target-field images and
groups never enter training or source holdout. The source holdout is used by
the training validator and for source-threshold selection; it is never a target
test set.

| Held-out field | Train images/groups | Source holdout images/groups | Target images/groups |
|---|---:|---:|---:|
| bbro | 1,200 / 24 | 200 / 5 | 1,104 / 24 |
| near30 | 1,203 / 30 | 200 / 6 | 2,999 / 49 |
| walledgarden | 1,200 / 16 | 200 / 4 | 299 / 12 |

Numbers in this table must be checked against `SPLIT_SUMMARY.json`; that record
and the CSVs are authoritative if a transcription differs.

LB grouping: a video is one group; all single photographs (`_i_`) from the same
field/day form one group. Dates and video identifiers are parsed from filenames.
PhenoBench has 772 validation images grouped by filename month/day (three
groups); the collection year is not inferred. OD has 964 test images in 18
groups: explicit sequence names define 17 groups, and 17 numeric filenames are
pooled into `inferred_timestamp_date:2016-05-11`, based on Unix time in UTC.

The bbro split is inherited from the pilot **without retraining-data selection**.
Its original COCO reference hashes are retained inside the original
`bbro/SPLIT_BINDING.json`. The raw COCO files are not redistributed here. This
historical binding records hashes of original private-layout files; its `files`
hashes must not be interpreted as hashes of the new metadata-only CSVs.

The pilot chose the source holdout first using sorted groups shuffled with
seed 0 and a dynamic program selecting the whole-group total nearest 200
(ties favor the smaller count). It then chose training groups nearest 1,200
from the remainder using seed 1. near30 and walledgarden use the original joint
dynamic program after sorted-group shuffling with seed 0, attaining the exact
approved training/holdout sizes. Neither method examines labels or predictions.

Run `python data_prep/reconstruct_splits.py` to recompute assignments from
group-size metadata and verify all fixed IDs. Repeated target calibration/test
holdouts are separate from these training splits: their seed is 20261013 and
their RNG key is `(scheme, K, repeat)`, shared across models, training seeds,
and nozzle counts, provided the target-image order is unchanged.
