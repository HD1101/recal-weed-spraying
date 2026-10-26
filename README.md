# ReCal: recalibrating weed-spraying risk after field transfer

ReCal studies how weed-spraying coverage changes when a detector moves to a
new field, and whether a small labeled calibration set can keep the average missed-weed fraction in weed-positive images below a target. This repository contains
CPU analysis, dataset preparation, fixed split identifiers, the frozen plan and
aggregate results. Final manuscript section/table references and the
version-specific code archive DOI will be added when available.

The analysis plan was frozen on 2026-10-16. Its original file is in `plan/`,
with SHA256 `e80795012ce4d4b492a0152a5e89acebe8246703c2fd579f7b4b98f64a7e81f9`.
Deviations D001–D004, including the analyses added after the results were produced, are recorded in plan/ and match Supplementary Material S1.3.
Original experimental result files have not been modified for this repository.

## Contents and manuscript correspondence

| Directory/file | Contents and manuscript component |
|---|---|
| `README.md` | Scope, reproduction steps and figure-panel data mapping |
| `LICENSE`, `LICENSE-DATA`, `CITATION.cff` | Code/data licensing and citation metadata |
| `analysis/` | Nozzle geometry, WCR, threshold selection, repeated ideal/grouped holdouts, RQ3, crop exposure, coverage-ceiling diagnostics, replay/summary scripts and all tests |
| `data_prep/` | Dataset conversion, grouping, fixed-split verification and descriptive-statistics scripts |
| `splits/` | Three fixed train/source-holdout/target ID lists, group metadata and split bindings |
| `plan/` | Original frozen analysis plan, deviations D001–D004 and post-result analysis chronology matching Supplementary Material S1.3 |
| `results/` | Main/supplementary CSV tables, all K/schemes/n, RQ3, crop exposure and GT descriptive statistics |
| `training/TRAINING_SETTINGS.md` | Plain-text training/export settings for all three detectors |
| `environment/requirements_cpu.txt` | Pinned CPU analysis dependencies |

Main tables use n=4, grouped splitting and K=20/50. RQ1 evaluates the entire
target range; RQ2 evaluates the remaining test images in each calibration draw.
See `results/README.md` for field definitions, missingness and aggregation.

## Experimental scope

Three held-out LB fields, three detectors and training seeds 0/1/2 give
27 fresh 200-epoch runs. bbro and walledgarden use 1,200 training and 200 source
holdout images; near30 uses 1,203/200 under D001. The bbro fixed split is
inherited from the pilot, but the resumed pilot weights/results are excluded.
All models and seeds share a fixed split within each rotation. External
evaluation uses only PhenoBench validation (772 images, three date groups) and
OD-SugarBeets test (964 images, 18 groups).

ASSET-DETR and standard RT-DETR-R18 use the archived RTDETR-main source and
fresh initialization. YOLO26m uses official pretrained weights and pinned
Ultralytics defaults with patience=0. Only last-epoch weights are exported,
after SHA256 verification: confidence floor 0.001, maximum 300 boxes/image,
FP32, no TTA. The native DETR implementation is not interchangeable with a
clean `pip install ultralytics==8.0.201`. Settings are in
`training/TRAINING_SETTINGS.md`; detector implementations, model YAMLs,
training/export integration code and weights are not distributed here.

The ASSET-DETR implementation of Liu et al. (2026) is available from the authors upon request.

In a reproduction check on a regenerated data split, the test mAP50 of the best-epoch weights was 0.720, against 0.742 reported by Liu et al. (2026), just outside the pre-specified ±0.02 interval; as pre-specified, the code was used without changes (Supplementary Material S3).

RT-DETR-R18 uses the standard rtdetr-r18 configuration (20.08 M parameters), which differs from the R18 comparison model in Liu et al. (2026).

## Data acquisition and preparation


Obtain original data from their providers; this repository redistributes no
raw images, masks, original label files or trained weights.

- [Lincoln Beet](https://github.com/LAR/lincolnbeet_dataset): download the
  official images and `all_fields_lincolnbeet.json` annotations.
- [PhenoBench](https://www.phenobench.org/dataset.html): download validation
  images, semantics and plant-instance masks. Its official dataset license is
  CC BY-SA 4.0. Retain the original filenames.
- [SugarBeets2016](https://www.ipb.uni-bonn.de/data/sugarbeets2016/index.html):
  original data underlying OD-SugarBeets, released under CC BY-SA 4.0.
- [WeedNet-R / OD-SugarBeets](https://github.com/GOOJJJ/WeedNet-R): use the
  authors' linked detection-box dataset, not raw SugarBeets2016 as a substitute.
  The evaluated copy has 964 test images. The original article lists 387
  validation images, while the supplied copy has 385; validation is unused.

Public availability does not itself establish redistribution permission.
No explicit standalone data license was verified for the LB/WeedNet-R
repositories; preserve provider notices and resolve permissions before sharing
any derived annotation bundle.

From the repository root, use Python 3.12 and install CPU dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements_cpu.txt
```

Set local paths in `data_prep/config.json` and `analysis/analysis_config.json`:

```bash
python data_prep/prepare.py --config data_prep/config.json
python data_prep/reconstruct_splits.py
```

LB labels preserve official integer COCO boxes, annotation order and the
original duplicate. PhenoBench uses every nonzero plant-instance ID and the
majority raw semantic label (ties select the lowest ID), maps 1/3 to crop and
2/4 to weed, drops/counts majority 0, and uses max−min+1 bounds. OD preserves
XML floating xyxy coordinates without rounding or an added pixel. Formal OD
risk labels contain weeds only; crop XML boxes are used only by the separate
GT descriptive-statistics script. Generated labels remain local inputs, not
part of the prediction release.

## Reproduce tables from exported predictions, without a GPU

The sanitized prediction package will be published as an independent Zenodo
data record: **https://doi.org/10.5281/zenodo.23654670**. It is CC BY 4.0, contains
predictions and necessary image/group/format metadata, and excludes images,
masks, GT annotations, auxiliary crop annotations and historical paths.

Prepared archive: `ReCal_Predictions_20261020_v1.tar.gz` (801,387,223 bytes).
SHA256: `27683f1c53d0d5d6cf4ea1ae338e05e8186789c192a03ac8f081aa7a39c4e3e6`.
Extract it outside this repository, verify its `SHA256SUMS`, and set
`predictions_root` to the extracted `recal-predictions-v1` directory.
The original-to-release hash mapping remains private.

Each of the 27 configurations has this layout, with rotation/model/seed changed:

```text
runs/bbro__asset_detr__seed0/attempt01/predictions01/
  EXPORT_COMPLETE.json
  source_holdout.json
  bbro.json
  PhenoBench.json
  OD_SugarBeets.json
```

Prediction boxes use original-image pixel coordinates. Completion hashes are
checked before reading. Obtain GT annotations/PhenoBench semantics separately
and prepare local manifests. Labels are joined by exact image ID; prediction
order and audited groups are retained. Predictions alone cannot define recall,
miss rates or crop exposure. Scientific risk/geometry functions are unchanged
from formal execution.

```bash
python analysis/reproduce.py --config analysis/analysis_config.json
python analysis/summarize.py --replays reproduced --output reproduced-summary
python analysis/compare_tables.py --generated reproduced-summary
```

To replay one job, or only its complete PhenoBench target analysis:

```bash
python analysis/reproduce.py --rotation bbro --model asset_detr --seed 0
python analysis/reproduce.py --rotation bbro --model asset_detr --seed 0 --datasets PhenoBench
```

These are alternative selections: use separate new output roots when changing
the target selection. Set paths in the configuration, not scientific functions.
The dataset selector retains the same source holdout, methods, K, n and draws.
Completed outputs are skipped only after input/result/code/configuration hash
checks; partial outputs are preserved and require a new output root. Release
verification completed the entire PhenoBench analysis for the bbro-rotation
ASSET-DETR seed 0 and YOLO26m seed 0 jobs: all seven methods, five K values,
four nozzle counts, both schemes and 200 draws, including date and crop-pixel
analyses. This is not a replay of all three target ranges for ASSET-DETR or of
all 27 jobs. Full 27-job replay has not been timed or rerun during repository
preparation.

For the post-result annotation-only descriptive table:

```bash
python data_prep/describe_truth.py --output reproduced-truth-statistics.csv
```

This step reads only original GT, not model predictions. Main/supplementary
analysis tables are generated by the replay/summarize commands.

## Local tests and numerical checks

All tests are included under `analysis/`, including `analysis/tests/`:

```bash
python -m unittest discover -s analysis -p "test_*.py"
python analysis/check_numbers.py
```

The headline bbro→bbro / ASSET / main CRC / n=4 / grouped / K=50 image-wise
miss rate recomputes as 7.88143882%, displayed as 7.9%. Result CSVs retain their
stored precision. Operational audit records are held separately by the authors.

## Manuscript figure-panel data sources

Only data sources are listed here; paper figure scripts, PDFs and caption
drafts are not distributed. This mapping follows the final figure definitions
supplied by the authors on 2026-10-21 and clarified on 2026-10-22, replacing the earlier draft mapping.
It verifies stored CSV fields, not the rendering of final figure PDFs. All
paths below are relative to `results/`.

M is the mean missed-weed fraction among weed-positive test images; A is the
sprayed-area fraction. A metric root below denotes its exact `_seed_mean`,
`_seed_min` and `_seed_max` columns unless a full column name is given. These
means/ranges are across training seeds, not confidence intervals. Multiply
fractions by 100 to display percentages. The 18 evaluation cells are the two
principal rotations (bbro, near30), three detectors and three target ranges
(the held-out LB field, PhenoBench, OD-SugarBeets).

| Figure or component | CSV in `results/` | Exact metric roots/fields and selection |
|---|---|---|
| Fig. 1 | None | Schematic; no experimental data. |
| Fig. 2 | `CROSS_SEED_ALL_RESULTS.csv` | `mean_actual_weed_image_miss_rate`, `mean_spray_area_fraction`; `nozzles=4`, `scheme=grouped`, `K=50`, `rotation` in bbro/near30, `method` in source/crc/perfect_detection_reference. The selection has 18 evaluation cells and 54 method rows. |
| Fig. 3 | `CROSS_SEED_ALL_RESULTS.csv` | M: `mean_actual_weed_image_miss_rate`; draw-level attainment proportion: `weed_image_miss_rate_le_alpha_fraction`; A: `mean_spray_area_fraction`. `model=asset_detr`, `rotation` in bbro/near30, `nozzles=4`, both grouped/ideal, all five K values, methods crc/naive/ltt/source/full_spray/perfect_detection_reference; dataset equal to rotation (held-out LB field only; 120 rows). |
| Fig. 4(a) | `PHENO_DATE_DIRECTIONS_BBRO_N4_K50_GROUPED_CROSS_SEED.csv` | M: `mean_weed_image_mean_miss_rate`; attainment counts: `macro_miss_le_0_1_count_seed_mean/min/max`; valid-count denominator: `macro_miss_valid_repeats_seed_mean`. `rotation=bbro`, `dataset=PhenoBench`, `nozzles=4`, `K=50`, `table=single_test_date`, calibration_dates → test_dates, all three detectors. These are main-CRC grouped-date analyses; seed-mean counts need not be integers. |
| Fig. 4(b) | `RQ3_SUMMARY_CROSS_SEED.csv` | `mean_weed_image_mean_miss_rate`; `nozzles=4`, `K=50`, the 18 evaluation cells, `relation=earlier_than_all_calibration` or `later_than_all_calibration` (36 rows). Availability: `valid_weed_image_mean_miss_rate_repeats_seed_mean`; absent relations remain missing, not zero. These rows describe main CRC under grouped splitting. |
| Fig. 5 — crop exposure | `CROP_EXPOSURE_SUMMARY_CROSS_SEED.csv` | Primary field **`mean_crop_exposure_seed_mean`**; seed range: `mean_crop_exposure_seed_min/max`. `dataset=PhenoBench`, `rotation` in bbro/near30, all detectors, `nozzles=4`, `scheme=grouped`, `K=50`, methods source/crc/perfect_detection_reference (18 rows). |
| Fig. 5 — corresponding A | `CROSS_SEED_ALL_RESULTS.csv` | `mean_spray_area_fraction`; use exactly the Fig. 5 crop-exposure selection and join on rotation/model/dataset/method/scheme/nozzles/K (18 rows). The same selected values also occur in `MAIN_TABLE_N4_GROUPED_K20_K50.csv`. |
| Fig. 6 — minimum-threshold M and box counts | `MIN_THRESHOLD_CROSS_SEED_N4.csv` | M: `weed_image_mean_miss_rate_at_min`; median weed predictions per image: `weed_predictions_per_image_median`; the 18 evaluation cells, `nozzles=4`. Threshold metadata: `minimum_grid_threshold_seed_mean` (0) and `prediction_confidence_floor_seed_mean` (0.001). |
| Fig. 6 — approximate feasibility upper bound | `CROSS_SEED_ALL_RESULTS.csv` | Approximate feasibility limit of Eq. (8), computed as (α(K̄′ + 1) − 1)/K̄′ with α = 0.1 and K̄′ = K_prime_mean_seed_mean of main CRC at n = 4, grouped, K = 50 in CROSS_SEED_ALL_RESULTS.csv (identical across detectors and rotations for the same target). Values: bbro 6.375%, near30 7.703%, PhenoBench 8.144%, OD-SugarBeets 6.491% (shown as 6.4, 7.7, 8.1 and 6.5%). This is an approximation based on mean K′, as stated in the manuscript, not a per-draw bound. |

Fig. 2 and Fig. 3 use repeated-holdout source/GT-reference rows from the main
results. Do not replace them with whole-target RQ1 values from
`MAIN_RQ1_WHOLE_TARGET_N4.csv`. The Fig. 6 feasibility limit is a display
quantity derived from the result table using mean K′, not missing data or a
per-draw bound. Final figure PDF rendering was not checked here.

The manuscript's primary crop-exposure estimate is the pooled ratio within each test set averaged over draws (field mean_crop_exposure), consistent with the paper's two-step aggregation; pooling pixels over all draws (field crop_exposure_pooled_over_repeats) is reported in Supplementary Material S8. Fig. 5(b) uses the primary estimate.

## Risk interpretation

Main CRC controls the expected missed-weed fraction among weed-positive images
under its exchangeability assumptions, with α=0.1 and B=1. Count CRC is a
comparator with B=132.3 and declared Nmax=147. HB-LTT uses δ=0.05 and fixed-
sequence testing over t=0,0.001,…,1. Calibration K=5/10/20/50/100, repeats=200,
analysis seed 20261013. Grouped splitting tests only groups outside the
calibration pool; unused pool images are excluded from testing.

Field/date shift can violate exchangeability. Neither a per-draw 90% guarantee
nor a physical spraying-performance guarantee is inferred. External camera
geometry is assumed in image coordinates. The GT-box trigger is a reference,
not a minimum possible spray-area bound.

## Licensing, citation and release

The author selected publication option A. `LICENSE` (MIT) covers repository
code only. Result tables in `results/` and the separate prediction-data package
are CC BY 4.0 (`LICENSE-DATA`). Original datasets retain the providers' licenses
and attribution requirements listed above; repository licenses do not
relicense those datasets or external software dependencies.

Ultralytics training/export integration is held outside this public repository.
The MIT grant must not be read as permission to redistribute an integrated
Ultralytics model stack under MIT. Plain-text settings are supplied in
`training/TRAINING_SETTINGS.md`; scripts and the ASSET implementation are
available from the authors upon request.

Author and software citation metadata are provided in `CITATION.cff`. Use a
version-specific code Zenodo DOI and the independent prediction-data DOI:
https://doi.org/10.5281/zenodo.23654670. Account operations and uploads are performed by
the authors; nothing has been uploaded during this local preparation.
