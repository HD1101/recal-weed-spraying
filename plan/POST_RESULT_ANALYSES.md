# Deviations and post-result analysis chronology

The original frozen plan is retained byte-for-byte. Its SHA256 is
`e80795012ce4d4b492a0152a5e89acebe8246703c2fd579f7b4b98f64a7e81f9`.

The plan was frozen on 2026-10-16. Its deviation rule is:

> 冻结后、相关结果产生之后的修改一律列为偏离。

Deviations D001–D004 are registered in `DEVIATIONS.json` and match Supplementary Material S1.3.
D001 was approved before any formal results and remains registered as a
deviation. D002 and D003 are post-result descriptive analyses. D004 is a
post-result reporting choice. Registering these items does not change the
original frozen text, scientific functions or existing numerical results.

The prespecified analyses include source-threshold transfer, naive
recalibration, main image-wise CRC, count CRC, HB-LTT, full spray, the GT-box
trigger reference, ideal/grouped repeated holdouts, RQ3 date relationships,
PhenoBench date directions and PhenoBench crop exposure. Crop exposure was
defined as a crop-pixel ratio; the plan did not specify how to aggregate it
across draws, which is documented under D004 below. The three fixed splits
(with the approved D001 count adjustment) and all 27 fresh 200-epoch runs are
formal. The interrupted/resumed pilot is excluded.

| Deviation | Timing | Classification | Supplementary material |
|---|---|---|---|
| D001 | Approved 2026-10-16, before any formal results | Approved source sample-count change | S1.3 |
| D002 | Decided 2026-10-17 after Batch A; computed after all formal results were completed | Post-result descriptive coverage-ceiling analysis | S10 |
| D003 | Computed 2026-10-19 using existing annotations only | Post-result descriptive GT statistics | S4.4 |
| D004 | Reporting choice after results; exact decision date not specified | Cross-draw crop-exposure aggregation choice | S8 |

## D001: approved departure from the frozen sample count

Holding out near30 leaves 1,403 source images. Exact 1,200/200 whole-group
sampling is impossible because the unused three images are smaller than the
smallest source group (seven). The author approved 1,203 training and 200 source
holdout images on 2026-10-16, before any formal results. This remains D001 in
`DEVIATIONS.json`, where the original approval details are retained. The frozen
text was not edited. bbro and walledgarden remain 1,200/200.

## D002: minimum-threshold coverage-ceiling analysis (S10)

This analysis was decided on 2026-10-17 after Batch A results to explain the
frequent full-spray fallback for YOLO26m. After all formal results were
completed, the auxiliary analysis used existing exported predictions and
corresponding GT. It is a post-result descriptive analysis and does not replace
or change any prespecified analysis.

The final auxiliary table covers every model × rotation × training seed ×
target dataset × nozzle count. IoU≥0.5 recall uses any same-class prediction,
not one-to-one detection matching. No inference or training was repeated. The
author retained the pinned YOLO one-to-many, class-aware NMS at IoU 0.7; no
NMS-free branch or few-shot fine-tuning was tested.

## D003: ground-truth descriptive statistics by date (S4.4)

On 2026-10-19, counts and box areas by LB field/date, PhenoBench date and OD
group were computed using existing annotations only. No model predictions were
used. These statistics were added for discussion after the results. They are
post-result descriptive comparisons, not a causal test or a prespecified
primary outcome, and do not replace or change any prespecified analysis.

## D004: reporting crop exposure across draws (S8)

The frozen plan defined crop exposure as the number of sprayed crop pixels
divided by the number of crop pixels, but did not specify aggregation across
repeated draws. After results were produced, the manuscript selected
`mean_crop_exposure` for the main text: compute the pooled crop-pixel ratio
within each draw's test set, then average those ratios over draws. This is
consistent with the manuscript's two-step aggregation.

The alternative field, `crop_exposure_pooled_over_repeats`, pools the sprayed
crop pixels and total crop pixels across all draws before taking their ratio.
Both aggregations are reported in Supplementary Material S8. D004 records a
reporting choice using existing result fields; it did not change any numerical
result or scientific function. The exact calendar date of this post-result
choice was not supplied and is left unspecified in `DEVIATIONS.json`.

## Repository preparation and interpretation notes

Portable I/O wrappers, privacy filtering, independent numerical checks and
local documentation support public reproducibility. They do not replace or
alter the archived analysis outputs or the original scientific functions.

The GT-box trigger is a comparison reference, **not a universal lower bound on
sprayed area**. The model-specific export floor limits all minimum-threshold
diagnostics. A fallback `calibration_loss_mean = 0` in historical records is a
placeholder; it is not evidence of a zero minimum empirical calibration loss.

In the reproduction check of ASSET-DETR (Supplementary Material S3), the test mAP50 of the best-epoch weights on a regenerated data split was 0.720, against 0.742 reported by Liu et al. (2026), slightly below the pre-specified ±0.02 interval; as pre-specified, the archived code was used without changes or retuning.
