"""Score independent visibility labels and frozen whole-body predictions."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from selfwatch_v2.hand_calibration import read_cache
from selfwatch_v2.overhead_data import match_boxes
from selfwatch_v2.pose_metrics import finalize, score_person, targets
from selfwatch_v2.wholebody_benchmark import load_dataset


def detection_visibility():
    result = {}
    for seq, role in (('MOT17-02-FRCNN','dev'),('MOT17-09-FRCNN','eval')):
        root = Path('datasets/mot17-continuity-native-v1')
        gt = np.loadtxt(root/seq/'gt/gt.txt',delimiter=',')
        frames = defaultdict(list)
        for row in gt:
            if row[6] and row[7] == 1:
                frames[int(row[0])].append(row)
        groups = defaultdict(Counter)
        cache = Path(f'runs/mot17-continuity-cache-{role}-v1/perception.jsonl')
        metadata = json.loads((cache.parent/'cache.json').read_text())
        if sha256(cache.read_bytes()).hexdigest() != metadata['perception_sha256']:
            raise ValueError('Detection cache hash mismatch')
        with cache.open(encoding='utf-8') as stream:
            for index, line in enumerate(stream,1):
                observation = json.loads(line)['observation']
                if observation['frame']['frame_index'] != index-1:
                    raise ValueError('Out of order MOT17 frames')
                rows = frames[index]
                truth = [(r[2],r[3],r[2]+r[4],r[3]+r[5]) for r in rows]
                predicted = [[value*1.5 for value in p['box_xyxy']] for p in observation['instances']]
                found = {m['gt'] for m in match_boxes(truth,predicted)['matches']}
                for n,row in enumerate(rows):
                    visibility = 'low' if row[8] <= .3 else 'medium' if row[8] < .7 else 'high'
                    groups[visibility].update(targets=1, matched=int(n in found))
        result[seq] = {k:dict(v,recall=v['matched']/v['targets']) for k,v in groups.items()}
    return result


def pose_visibility(plan):
    data, manifest = load_dataset(Path('datasets/hand-calibration-evaluation-v1'))
    baseline = Path('runs/hand-cal-eval-body-v1')
    specialist = Path('runs/hand-cal-eval-refine-v1')
    calibration = json.loads(Path('runs/hand-calibration-evaluation-v1.json').read_text())
    source = calibration['evaluation']
    base_path = baseline/'predictions.jsonl.gz'
    hand_path = specialist/'refined.jsonl.gz'
    report = json.loads((specialist/'results.json').read_text())
    if (source['selection_sha256'] != manifest['selection_sha256'] or
        source['hand_predictions_sha256'] != sha256(hand_path.read_bytes()).hexdigest() or
        report['input_lock']['baseline_cache_sha256'] != sha256(base_path.read_bytes()).hexdigest()):
        raise ValueError('Pose evidence hash mismatch')
    annotations = {a['id']:a for a in data['annotations']}
    views = {image:label for label,images in plan['pose_view_labels'].items() for image in images}
    base, refined = read_cache(base_path),read_cache(hand_path)
    if [r['image_id'] for r in base] != source['image_ids'] or [r['image_id'] for r in refined] != source['image_ids']:
        raise ValueError('Pose frame order differs from calibration')
    groups = defaultdict(Counter)
    sampled = Counter()
    for frame_index,(before,after) in enumerate(zip(base,refined)):
        people = [p for p in before['detector']['instances'] if p['label'] == 'person']
        matches = {m['gt']:m['prediction'] for m in before['matches']['matches']}
        roi = after['rois'][0] if frame_index % 5 == 0 and after['rois'] else None
        if roi is not None:
            sampled['roi_calls'] += 1
        view = views.get(before['image_id'],'unreviewed')
        for index, annotation_id in enumerate(before['ground_truth_ids']):
            annotation = annotations[annotation_id]
            person = matches.get(index)
            points = people[person]['keypoints_xyc'] if person is not None else None
            body, _ = score_person(annotation,points)
            side = min(annotation['bbox'][2:])
            size = 'under64' if side < 64 else '64to127' if side < 128 else '128plus'
            for name in ('body_visible','body_occluded'):
                if name in body:
                    groups[f'{name}/all'].update(body[name])
                    groups[f'{name}/size/{size}'].update(body[name])
                    groups[f'{name}/view/{view}'].update(body[name])
            for name, target, offset, eligible, box, alpha in targets(annotation):
                if name not in ('left_hand','right_hand') or min(box[2:]) <= 0 or not eligible.any():
                    continue
                wrist = annotation['keypoints'][3*(9 if offset == 91 else 10)+2]
                wrist_group = {2:'visible',1:'occluded',0:'unlabelled'}[wrist]
                sampled['eligible'] += int(eligible.sum())
                used = (roi is not None and person == roi['person'] and offset == roi['offset'])
                chosen = after['refined'][person] if used else points
                if used:
                    result = np.asarray(chosen)[offset:offset+21][eligible]
                    accepted = result[:,2] >= calibration['frozen']['threshold']
                    error = np.linalg.norm(result[:,:2]-target[eligible,:2],axis=1)
                    sampled.update(emitted=int(accepted.sum()),correct=int((accepted & (error<=alpha*math.sqrt(box[2]*box[3]))).sum()))
                row = body[name]
                width = min(box[2:])
                hand_size = 'under16' if width < 16 else '16to31' if width < 32 else '32plus'
                groups['hand/all'].update(row)
                groups[f'hand/size/{hand_size}'].update(row)
                groups[f'hand/view/{view}'].update(row)
                groups[f'hand/wrist/{wrist_group}'].update(row)
    return dict(metrics=finalize(groups),sampled_specialist=dict(sampled,
        precision=sampled['correct']/max(1,sampled['emitted']),yield_all=sampled['correct']/sampled['eligible']),
        scope='COCO WholeBody reliable hand labels. Wrist visibility is an occlusion proxy for hand slices. Viewpoint is independently assigned from the image on a fixed 24-image subset. Sampled specialist uses one ROI every fifth image in the frozen cache as a 10 Hz scheduling simulation.')


def main():
    plan = json.loads(Path('docs/phase2-protocol.json').read_text(encoding='utf-8'))
    result = dict(detection_visibility=detection_visibility(),pose=pose_visibility(plan))
    path = Path('runs/phase2-closeout-v1/visibility.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(detection=result['detection_visibility'],sampled=result['pose']['sampled_specialist']),indent=2))


if __name__ == '__main__':
    main()
