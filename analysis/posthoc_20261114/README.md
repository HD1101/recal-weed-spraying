# D008-D010 post hoc sensitivity analyses (v1.2.0)

ReCal v1.2.0 adds the registered revision 2 plan, code, tests and results
for D008 (nozzle count), D009 (risk target) and D010 (weed-box coverage criterion).
See [the plan](../../plan/POSTHOC_PLAN_20261114_REV2.md) and
[the report](../../results/posthoc_20261114/POSTHOC_REPORT_20261114.md).

Run the geometry tests from the repository root with the dependencies in
`environment/requirements_cpu.txt`:

```bash
python -m unittest discover -s analysis/posthoc_20261114 -p 'test_*.py'
```

The checked-in CSVs are a result snapshot. To replay the CPU calculations,
regenerate the input manifests and geometry caches from the original annotations
and the archived predictions, using the verified formal draws. The archived
verification inputs are author-side artifacts. Set `RECAL_WORKSPACE` to the
directory holding the input layout used by `run.py`. Set
`RECAL_POSTHOC_OUTPUT` to a separate empty directory that you create before
running `run.py regression`, `run.py compute` and `report.py` in that order.

The historical engine hash in the run receipts identifies the executed script.
This release adjusts repository and output path resolution in `run.py` and
makes traceback paths repository relative. Analysis logic and numeric outputs
retain their original values.

The figure and supplement builder in `results/posthoc_20261114/for_manuscript/`
reads the checked-in key CSVs and rebuilds its outputs in that directory.
