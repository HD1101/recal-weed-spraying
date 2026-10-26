# Training and export settings

This repository follows publication option A: it distributes independent CPU
analysis code and settings, without Ultralytics training/export integration or
detector implementation/configuration files. The original invocation scripts
are archived locally outside the public repository, not deleted.

| Setting | ASSET-DETR | RT-DETR-R18 | YOLO26m |
|---|---|---|---|
| Training epochs | Fixed 200 | Fixed 200 | Fixed 200 |
| Early stopping / patience | Disabled / 0 | Disabled / 0 | Disabled / 0 |
| Training seeds | 0, 1, 2 | 0, 1, 2 | 0, 1, 2 |
| Initialization | From scratch, archived final ASSET architecture | From scratch, standard R18 architecture | Official pretrained YOLO26m weights |
| Pretrained weight SHA256 | Not applicable | Not applicable | 401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7 |
| Ultralytics version | 8.0.201 string in archived RTDETR-main | Same archived RTDETR-main | Official 8.4.176 |
| Physical batch | 8 | 8 | 16 |
| Gradient accumulation | Fixed 2 | Fixed 2 | Official default warmup: 1→4; nominal 4 afterward |
| Nominal effective batch | 16 | 16 | 64 |
| Training input | 640×640 | 640×640 | imgsz=640, official training preprocessing |
| Optimizer | AdamW | AdamW | Official optimizer=auto resolved to AdamW |
| Initial learning rate | 0.0001 | 0.0001 | Actual auto-resolved 0.001667; pre-resolution lr0 argument 0.01 |
| AdamW beta values | Native framework defaults | Native framework defaults | 0.9, 0.999 |
| Weight decay | 0.0001 | 0.0001 | 0.0005 for decayed weights; 0 for biases/norms |
| Warmup | Five epochs of linear batch warmup | Same | Official three-epoch warmup |
| Warmup bias learning rate | 0.1, historical framework default | Same | Auto optimizer resets to 0.0 |
| Schedule | Cosine after warmup | Same | Official linear multiplicative decay (cos_lr=False) |
| lrf | 0.01 | Same | Official default 0.01 |
| Final native learning rate | 0.000001 | 0.000001 | Official schedule; no native lr override |
| AMP during training | False | False | True, official checks passed |
| Augmentation | Mosaic 1.0; final 10 epochs closed; MixUp 0.10; historical HSV/affine/flip defaults | Same | Pinned official defaults; no native/smoke overrides |
| Selected export checkpoint | Last epoch only | Last epoch only | last.pt only |
| Checkpoint verification | SHA256 checked before export | Same | Same |
| Save/memory handling | Save-time garbage collection; per-epoch CUDA/timing records | Same | Same |
| Inference input | Stretch original image to 640×640 | Same | imgsz=640 with aspect-preserving default rectangular letterbox |
| Export confidence floor | 0.001 | 0.001 | 0.001 |
| Maximum exported boxes per image | 300 | 300 | 300 |
| Export numeric precision | FP32 | FP32 | FP32 |
| Test-time augmentation | None | None | None |
| Postprocessing | Pinned native RT-DETR processing | Same | Pinned official one-to-many, class-aware NMS, IoU 0.7 |
| Coordinate output | Original-image pixel xyxy; clip x to [0,W], y to [0,H]; discard nonpositive boxes | Same | Same, after reversing padding/scaling |

All models/seeds share each rotation's fixed group-disjoint split: bbro
1,200/200; near30 1,203/200 under D001; walledgarden 1,200/200. External datasets
are inference-only. The native implementation is not equivalent to a clean
PyPI installation with the same 8.0.201 version string. Detailed resolved
settings and checkpoint hashes are retained separately by the authors.

The observed training runtime was Python 3.12.3, PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128 and PyTorch CUDA runtime 12.8. CPU analysis dependencies
are pinned in `environment/requirements_cpu.txt`.

Native augmentation defaults were HSV h/s/v = 0.015/0.7/0.4; degrees = 0;
translate = 0.1; scale = 0.5; shear = 0; perspective = 0; vertical flip = 0;
horizontal flip = 0.5. These are the existing framework defaults, not new
hyperparameters introduced for this public copy.

Training and export scripts, and the ASSET-DETR implementation of Liu et al. (2026), are available from the authors upon request.
