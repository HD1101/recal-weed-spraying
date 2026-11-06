# Post-result analyses D005--D007, 2026-11-04

This additive module is invoked explicitly. It does not register a new method
in the original risk-control module or change any existing function, split,
training recipe, result CSV, or frozen plan. It was executed locally; its code and result snapshot are prepared for release v1.1.0.

The time-stamped plan and its pre-computation revision are in
`results/posthoc_20261104/`. D006 revision 1 supersedes the originally proposed
K100-test reuse because formal random seeds depend on K. All results, favorable
or unfavorable, are reported as post-result deviations D005--D007.

- `core.py`: exact same-loss naive-M; nested grouped sampling; MC SE.
- `engine.py`: a query cache built exclusively by the original `ImageCurve`
  and `CropCurve`, on the exact formal floating-point threshold grid.
- `archive.py`: read and hash-check all 81 original FULL_RESULTS members;
  compute D007 and verify archived means and SEs against the immutable CSV.
- `run.py`: compute all D005 and revised-D006 results, with original CRC
  thresholds, per-draw metrics and crop-exposure verification.
- `summarize.py`: paired tables, cross-seed means/ranges, exceedance counts,
  test-size audits, boundary flags, and Chinese report.
- `test_posthoc.py`: new statistical, split-construction and geometry-interface
  tests; original tests are retained unchanged.

The settings are the formal grid 0--1 with step 0.001, n=1--4,
K=5/10/20/50/100, R=200, base seed 20261013, alpha=0.1. Three detector training
seeds are evaluated separately. D005 uses the exact formal index lists and
full-spray handling. D006 uses SHA256(`20261013:D006:r`) first eight bytes,
big endian, r=1..200: shuffle sorted groups once, then call random.sample
sequentially for K=5/10/20/50/100. Pools are nested; calibration image samples
need not be nested. Test groups are the K100-pool complement. Empty tests are
retained and marked NA. No train/target-pool or GPU operations occur here.

All ground truth, masks, prediction archives and computation caches remain
private external inputs. A private input JSON supplies only local paths:

```json
{
  "GT_manifests": {"LB": ".../GT_ONLY_manifest.json", "PhenoBench": ".../GT_ONLY_manifest.json", "OD_SugarBeets": ".../GT_ONLY_manifest.json"},
  "pheno_semantics": ".../semantics",
  "predictions_root": ".../recal-predictions-v1",
  "private_root": ".../posthoc-private",
  "output_root": ".../results/posthoc_20261104"
}
```

The input-verification gate must match all image IDs, formal source hashes,
and split bindings before risk calculation. The archive extraction writes only
new compact evidence/caches. Completion files and result hashes allow completed
targets to be verified and skipped; partial output directories are preserved.

From the repository root, with the original CPU environment available:

```bash
python -m unittest discover -s analysis -p 'test_*.py'
python analysis/posthoc_20261104/archive.py --archive /path/to/original-backup.tar.gz --manifest /path/to/FILE_MANIFEST.json --private /path/to/posthoc-private
python analysis/posthoc_20261104/run.py --input-config /path/to/private-inputs.json --workers 2
python analysis/posthoc_20261104/summarize.py
```

Existing numeric summaries use the per-test-set ratio, then an equal mean over
draws; crop exposure is `mean_crop_exposure`. MC SE is conditional on a fixed
trained detector and sampled dataset, not a confidence interval over model
training seeds or a risk guarantee. Cell counts use the cross-seed mean M.
