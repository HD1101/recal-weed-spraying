# Result fields and manuscript mapping

All CSV fractions are in [0,1]; multiply by 100 for percentages. Blank values
are undefined/unavailable, never zero. `_seed_mean/min/max` are the arithmetic
mean and range of available seed-level statistics, not confidence intervals.
`n_seeds`, `all_three_seeds`, and metric-specific `*_valid_seeds` preserve the
population. The formal models use seeds 0, 1, and 2.

| File | Role |
|---|---|
| `MAIN_TABLE_N4_GROUPED_K20_K50.csv` | Main RQ2 table: bbro/near30 rotations, n=4, grouped, K=20/50 |
| `MAIN_RQ1_WHOLE_TARGET_N4.csv` | RQ1: source and GT reference on the entire target range |
| `CROSS_SEED_ALL_RESULTS.csv` / `PER_SEED_ALL_RESULTS.csv` | All rotations, models, datasets, methods, K, schemes, n |
| `SUPPLEMENT_N2_GROUPED_K20_K50.csv` | n=2 supplement |
| `SUPPLEMENT_WALLEDGARDEN_N4_GROUPED_K20_K50.csv` | Supplemental held-out walledgarden rotation |
| `RQ3_SUMMARY_*` | Main CRC/grouped, four date relationships; full originals also retain empty `date_unresolved` rows |
| `PHENO_DATE_DIRECTIONS_*` | Main CRC/grouped, calibration-date combination → test date/combinations |
| `CROP_EXPOSURE_SUMMARY_*` | PhenoBench crop exposure; source, main CRC, GT reference, full spray; n=1–4 |
| `MIN_THRESHOLD_*` | Post-result coverage ceiling/recall diagnostic within the fixed stored export |
| `GT_DESCRIPTIVE_STATISTICS_20261019.csv` | Post-result annotation-only descriptive statistics |
| `BBRO_ASSET_CRC_N4_K50_RUN_METRICS.csv` | 600 archived draw-level metric rows for independent numerical checks |

Section numbers and final table numbering are author placeholders; RQ labels
and filenames identify the analysis unambiguously. The example headline M=7.9%
is bbro rotation / ASSET / bbro target / main CRC / n=4 / K=50 / grouped.

`mean_actual_weed_image_miss_rate` is the equal mean over calibration repeats of
the test-set average miss fraction restricted to weed-positive images. It is
not `1 - mean_actual_WCR`. `weed_image_miss_rate_le_alpha_fraction` reports the
fraction of defined test draws with that image-wise average ≤0.1. This empirical
attainment fraction is not the CRC expectation guarantee.

`mean_actual_WCR` pools sprayed/total weed counts within each test draw, then
averages over draws. `WCR_ge_target_fraction` is its ≥0.9 attainment fraction.
`mean_spray_area_fraction` averages each draw's total spray-union area divided
by total image area. `K_prime_mean` counts calibration images containing weeds;
the original K includes all sampled images. `full_spray_fallback_count` counts
fallback policies, whereas `unavailable_repeats` counts undefined policies.

RQ1 whole-target evaluation is not the same estimand as averages over the
remaining test images after repeated target calibration. `perfect_detection_reference`
uses true weed boxes to trigger the same spray geometry; it is a comparison
reference, not a lower bound and not a deployable detector.

RQ3 relations are `earlier_than_all_calibration`, `later_than_all_calibration`,
`same_day_as_calibration`, and `other`. Per-draw subgroup averages exclude
undefined empty/no-weed subgroups for the relevant metric only, with valid draw
counts retained. PhenoBench `table=date_combination` and `single_test_date`
are distinct aggregation levels and must not be summed together.

Crop pixels are semantics labels 1 or 3, irrespective of plant-instance ID.
Sprayed pixels use pixel centers inside the half-open spray-region union.
`mean_crop_exposure` is the equal mean of test-draw pixel ratios **within each
seed**. `crop_exposure_pooled_over_repeats` pools sprayed and total crop pixels
over all test draws **within that seed** before taking their ratio; its
`_seed_mean/min/max` fields are supplementary pooled-over-draws statistics.
`crop_image_exposure_pooled_over_repeats` is the supplementary image-wise
quantity (zero-crop images excluded). These statistics must not be exchanged.
**The manuscript's primary crop-exposure estimate is the pooled ratio within
each test set averaged over draws (`mean_crop_exposure`), consistent with the
paper's two-step aggregation. Pooling pixels over all draws
(`crop_exposure_pooled_over_repeats`) is reported in Supplementary Material S8.
Fig5(b) uses the primary estimate. No result values were changed.**

Minimum-threshold recall is the fraction of GT weeds matched by any same-image
weed prediction at IoU≥0.5 and confidence≥0.001; it is not one-to-one matching.
The median prediction count includes all target images. The threshold grid
starts at t=0, but predictions below 0.001 were never exported. Full spray is a
separate whole-image action. Historical fallback `calibration_loss_mean=0`
placeholders are not used in these summaries or explanations.

All summary CSVs are byte-identical copies of their listed source exports;
they have not been recomputed or rounded for this repository. The numerical
fixture is a separately extracted subset of archived draw metrics. Original
source hashes and per-file provenance are retained separately by the authors.
