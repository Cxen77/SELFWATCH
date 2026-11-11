"""Select a detector on the reviewed development clips before fresh qualification."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import statistics

from scripts.validate_phase2 import OrientedDetector, rates
from selfwatch_v2.contracts import FrameRef
from selfwatch_v2.overhead_data import match_boxes
from selfwatch_v2.replay import load_manifest, replay
from selfwatch_v2.yolo import YoloPerception


CANDIDATES = [('yolo26n-seg.pt',640),('yolo26n-seg.pt',960),
              ('yolo26n.pt',640),('yolo26n.pt',960),('person-domain-expanded-v1/detector.pt',640)]


def main():
    import torch
    torch.set_num_threads(2)
    output = Path('runs/phase2-closeout-v1')
    plans = [json.loads(Path(f'docs/phase2-{name}.json').read_text()) for name in ('protocol','qualification')]
    rows = [row for plan in plans for row in plan['frames']]
    rotations = {clip:angles for plan in plans for clip,angles in plan['rotations'].items()}
    rotations['001844']['rank1'] = 270
    wanted = {(row['clip'],row['camera'],row['index']):row for row in rows}
    images = {}
    for clip in rotations:
        sources = load_manifest(Path(f'datasets/retailaction-replay-verified/{clip}.json'))
        for frame,image in replay(sources):
            row = wanted.get((clip,frame.camera_id,frame.frame_index))
            if row is None:
                continue
            if sha256(image.tobytes()).hexdigest() != row['sha256']:
                raise ValueError('Image differs from development labels')
            images[row['key']] = image
    if len(images) != len(rows):
        raise ValueError('Missing labelled development frames')
    trials = []
    for weight, size in CANDIDATES:
        detector = YoloPerception(Path('weights/v2')/weight,'0',size,.25,half=True)
        detector.include_masks = False
        observations = []
        for row in rows:
            image = images[row['key']]
            frame = FrameRef(row['camera'],row['clip'],row['index'],0,0,*row['wh'],'public_retail_frame')
            observation, _ = OrientedDetector(detector,rotations[row['clip']]).predict(frame,image)
            observations.append(observation)
        ms = statistics.median(o.inference_ms for o in observations[4:])
        for threshold in (.25,.35,.45,.55,.65):
            counts = Counter()
            views = defaultdict(Counter)
            for row, observation in zip(rows, observations):
                boxes = [p.box_xyxy for p in observation.instances if p.label == 'person' and p.score >= threshold]
                matches = match_boxes(row['boxes'],boxes)
                tally = {k:v for k,v in matches.items() if k != 'matches'}
                counts.update(tally)
                views[f"{row['clip']}/{row['camera']}"].update(tally)
            metric = rates(counts)
            trials.append(dict(weights='weights/v2/'+weight,imgsz=size,confidence=threshold,half=True,
                sha256=detector.provenance['sha256'],median_detector_ms=ms,score=metric,
                worst_view_recall=min(rates(v)['recall'] for v in views.values())))
    feasible = [r for r in trials if r['score']['recall']>=.80 and r['score']['precision']>=.85
                and r['worst_view_recall']>=2/3 and r['median_detector_ms']<=75]
    chosen = max(feasible,key=lambda r:(r['score']['recall'],r['score']['precision'],-r['median_detector_ms'])) if feasible else None
    result = dict(chosen=chosen,development_trials=trials,rotations=rotations,
                  selection='Keep only candidates passing development recall, precision, worst-view recall and detector budget. Prefer recall, then precision, then lower median latency.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'detector-selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(chosen,indent=2))


if __name__ == '__main__':
    main()
