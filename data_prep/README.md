# Dataset conversion and grouping

Configure paths in `config.json`; source datasets are read-only. Every prepared
dataset goes to a new directory. Existing prepared outputs are preserved.

`prepare.py --dataset LB` reads the official combined COCO annotation, retains
annotation order and duplicates, and writes high-precision normalized YOLO
labels. No original LB text label is rewritten. `--dataset PhenoBench` calls
the unchanged formal `pheno_boxes` function (majority semantics before mapping,
lowest-ID tie break, max−min+1). The original audit has zero majority ties and
zero majority-background discards; the portable converter verifies counts.
`--dataset OD_SugarBeets` parses original XML floats, evaluates only the fixed
964-image test set, and keeps the formal weed-only labels. The crop XML boxes
are available to the descriptive-statistics script without clipping.

The provider-specific directory names in `config.json` are examples; set the
actual author-obtained locations once. Missing/incorrect counts fail rather
than silently changing the evaluated version. Group/ID metadata is supplied
in `splits/*_IMAGE_GROUPS.csv`; no label geometry is redistributed.

`reconstruct_splits.py` checks the size-based whole-group assignments against
all three fixed CSV splits. `materialize_training.py` creates new symlinks
using those exact ID lists and never includes target images in a trainer's
dataset YAML. The source holdout may be monitored during training but formal
weights are always the last fixed epoch, not best/source-selected weights.

`describe_truth.py` reproduces the post-result field/date/group descriptive
table from original GT only; it never imports a detector or prediction file.
