# RetailAction pilot

Prepared 1 November 2026. Official source: https://huggingface.co/datasets/standard-cognition/RetailAction . Revision: `49cb590723db921a4bd5a38adea92f6abd3f7a00`.

Downloaded 16 training sample directories: 32 MP4 files and 16 metadata files, approximately 23.6 MB including README, original license, attribution notice and hash manifest. The archive stream read approximately 32.5 MB then stopped; the full 26 GB training archive was not saved. This is an archive-prefix development subset, not a random or representative benchmark. Validation/test splits are untouched.

Raw files are in `datasets/retailaction-pilot/`. The original LICENSE includes attribution and commercial conditions; retain it with derivatives and review before distribution. Data remain anonymized.

Frames are sampled unevenly. The importer verifies every file hash and decodes both views, then builds manifests using original frame timestamps. Fourteen pairs passed validation; `006853` and `009027` lack timestamps and are excluded from timing replay while retained intact as raw data. The timing-filtered subset is biased. Each sample has a separate epoch; unrelated anonymized clip times are not merged into one store session.

Use the completed index `datasets/retailaction-replay-verified/index.json`. The earlier partial preparation folder is not the completed index.

```powershell
python scripts/download_retailaction_subset.py --output datasets/retailaction-pilot-new --samples 16 --max-mib 256
python -m selfwatch_v2.retailaction datasets/retailaction-pilot-new --output datasets/retailaction-replay-new
python -m selfwatch_v2 --manifest datasets/retailaction-replay-verified/000398.json --weights weights/v2/yolo26n-seg.pt --device 0 --output runs/retailaction-new --preview
```

Real GPU runs processed the same 64 frames of sample `000398` (32 per view), with confidence 0.25 and image size 640. YOLO26n segmentation produced 56 person observations; COCO17 pose produced 37; fuzzy SH17 detection produced 54 person, 21 head and 1 ear observations. These are repeated prediction counts, not true positives, accuracy or a ranking. Other predictions include skis and gloves; such outputs need manual/annotated evaluation, and gloves are not equivalent to bare hands.

Runs: `runs/retailaction-000398-seg/`, `runs/retailaction-000398-pose/`, `runs/retailaction-000398-ppe/`. Each contains provenance, timestamped observations, timings and annotated previews. The existing fuzzy environment was used read-only for GPU inference; no fuzzy data/model changes were made.

RetailAction provides take/put/touch annotations and action locations, not complete ground-truth boxes for every person/hand, full visits, product custody or POS settlement. Additional annotations and continuous calibrated multi-camera footage are needed for the later claims. No fine-tuning or payment/exit accuracy is claimed. A webcam can test live capture later, but cannot replace paired camera ground truth.
