# Overhead detection pilot — 2 November 2026

Historical report. The subsequent expanded training, matched comparison,
transfer failure and decision to retain the portal model are recorded in
[the master log, E13–E14](PROJECT_LOG.md). Statements below about training not
having run describe the state on 2 November, not the current project status.

Stage 2 is still open. The pretrained detector misses too many people in
overhead fisheye views; an identity/memory change cannot repair an absent observation.

## Independent annotations and preserved data

`datasets/wepdtof-pilot/` contains 128 images, source person annotations and ROI
maps from 16 WEPDTOF clips / 14 source scenes. Uniform midpoint samples were
chosen before inference. Paired segments of the same source video stay together.
All downloaded members have SHA-256 and ZIP CRC provenance. HTTP range requests
read 197,435,892 archive payload bytes, rather than downloading the 5,435,073,011-byte archive.
Source notices are preserved; the dataset is for noncommercial research.

The fixed scene split is 56 development images (seven groups), 32 validation
images (three groups), and 40 sealed final-test images (four groups). Final-test
files were checksum-verified but **not decoded, inferred, exported for training,
or exposed in the review**. This is a small custom scene split, not the authors'
published train/test protocol.

## Measured comparison

`runs/overhead-detector-pilot-v1/` stores raw observations, source code snapshots,
input hashes, split lock, a development selection frozen before validation,
and a review of the same lossless model inputs. Open the detector study from
the new camera workspace, or serve this directory on port 8768.

| Candidate | Validation precision | Recall | F1 | Mean detector ms |
|---|---:|---:|---:|---:|
| YOLO26n-seg, 640 | 0.744 | 0.233 | 0.355 | 60.6 |
| YOLO26n-seg, 960 | 0.748 | 0.351 | 0.478 | 78.2 |
| Four orientations, 640 | 0.634 | 0.317 | 0.422 | 230.1 |
| Weak-view rotation, 640 | 0.782 | 0.233 | 0.359 | 140.5 |

There are 262 annotated validation people across 32 images. The development
quality winner was rotation, not 960; its poorer validation performance must
remain visible. The 960 result is an exploratory comparison, not an independently
confirmed production choice. None met the 50 ms development P95 budget.

Scores use confidence .25, one-to-one matching at IoU .5 and **axis-aligned
boxes enclosing the original clipped rotated annotations**. Targets and
predictions must have their centers inside the source ROI; excluded input
pixels are neutral 114 for every candidate. These are box precision/recall/F1,
not published rotated AP, segmentation accuracy, pose accuracy or IDF1.

Latency includes detector preprocessing, transfers, mask-polygon materialization
and rotation/NMS; it excludes decode, pose, ReID and export. It was measured
on this laptop, without controlling unrelated OS work. V1 P95 uses the lower
empirical percentile; its exact runner is archived. Future runs use nearest-rank
P95 and automatically snapshot source hashes. These numbers do not establish
live three-camera throughput. Peak Torch reserved allocation was 252 MiB for
this sequential detector comparison; other GPU processes and drivers are excluded.

## Optional recovery and training preparation

`--person-rotation-mode memory` lets the original cognitive engine request
a same-checkpoint orientation probe when a recent missing owner's region has
no native detection. Regions expire after two seconds, are camera/epoch scoped,
and never become observations themselves. A per-camera .5-second throttle limits
extra passes. Native objects remain; extra passes add person evidence only.
The default remains `off`.

A 32-observation paired replay in `runs/combined-memory-recovery-016699/`
completed with CPU pose/GPU inference overlap: 16 probes, 80 total detector
passes, nine memory-region requests. This checks integration; it does not
establish a tracking gain. No cross-camera link was produced in that short replay.

`datasets/wepdtof-detector-pilot/` contains 56 development images / 698 person
boxes and 32 validation images / 262 boxes, using the same preprocessing.
It includes `data.yaml`, per-file hashes and a bounded training recipe (batch 4,
640 pixels, 20 epochs, zero data-loader workers). **No fine-tuning has run.**
It has person boxes only, so a separate detection experiment is appropriate;
it cannot supervise segmentation masks, hands, products, depth or ReID. Retain
the current segmentation/whole-body path while evaluating any trained branch.
Fifty-six training images are a pipeline pilot, not sufficient evidence of
general retail performance. Expand training-only scenes before a main experiment.

## Research implications and next work

The older [WEPDTOF paper](https://openaccess.thecvf.com/content/WACV2022/papers/Tezcan_WEPDTOF_A_Dataset_and_Benchmark_Algorithms_for_In-the-Wild_People_Detection_WACV_2022_paper.pdf)
studies temporal extensions to a rotation-aware detector. Its methods include
offline use of future frames; an online comparison must explicitly restrict
available evidence. We use the dataset to expose current errors, not to claim
those algorithms are reproduced.

[LOAF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Large-Scale_Person_Detection_and_Localization_Using_Overhead_Fisheye_Cameras_ICCV_2023_paper.pdf)
supports rotation-aware training and tests generalization to unseen scenes.
Its equivariant training is more than ordinary image augmentation. Its metric
localization also uses a camera model and camera altitude; it does not justify
turning an arbitrary depth map into measured store distances. Both original
PDFs are saved in `Desktop/SELFWATCH_RESEARCH/papers/`; the library now has 32 verified PDFs.

Next: profile the complete camera workspace, expand development labels in the
measured failure slices, train one small person detector separately, compare
against its own pretrained checkpoint, and keep the final test sealed until
configurations and reporting are locked. Geometry, item custody and POS gates
remain later work. Rotation recovery alone is not a new research contribution.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.download_wepdtof_subset --frames-per-sequence 8
.\.venv\Scripts\python.exe -m selfwatch_v2.overhead_benchmark --output runs/overhead-new
.\.venv\Scripts\python.exe -m selfwatch_v2.overhead_review --run runs/overhead-new
.\.venv\Scripts\python.exe -m selfwatch_v2.prepare_overhead_training --output datasets/wepdtof-detector-new
```

The downloader is payload-bounded and rejects unsafe member paths, source
version changes and corrupt transfers. Existing data is preserved and verified.
