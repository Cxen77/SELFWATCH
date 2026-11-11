"""Qualify a limited retail perception operating point on the recorded clips."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

from selfwatch_v2.hand_refinement import hand_roi
from selfwatch_v2.overhead_data import match_boxes
from selfwatch_v2.rotation import inverse_instance


class OrientedDetector:
    def __init__(self, detector, angles):
        self.detector = detector
        self.angles = angles

    def predict(self, frame, image):
        import cv2
        angle = self.angles[frame.camera_id]
        if angle == 0:
            return self.detector.predict(frame, image)
        started = time.perf_counter()
        rotation = {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}
        rotated = cv2.rotate(image, rotation[angle])
        observation, _ = self.detector.predict(replace(frame, width=rotated.shape[1], height=rotated.shape[0]), rotated)
        restored = tuple(inverse_instance(person, frame, angle) for person in observation.instances)
        return replace(observation, frame=frame, instances=restored,
                       inference_ms=(time.perf_counter()-started)*1000), None


class SampledHand:
    def __init__(self, pose, hand, interval, threshold):
        self.pose = pose
        self.hand = hand
        self.interval = interval
        self.threshold = threshold
        self.last = {}
        self.calls = Counter()
        self.accepted = 0

    def refine(self, observation, image):
        observation = self.pose.refine(observation, image)
        frame = observation.frame
        key = (frame.camera_id, frame.stream_epoch)
        if frame.captured_at_s-self.last.get(key, float('-inf')) < self.interval-1e-6:
            return observation
        candidates = []
        for index, person in enumerate(observation.instances):
            if person.keypoints_xyc is None:
                continue
            for offset in (91,112):
                roi = hand_roi(person.keypoints_xyc[offset:offset+21], frame.width, frame.height)
                if roi is not None:
                    candidates.append((roi['score'], index, offset, roi['box']))
        if not candidates:
            return observation
        _, index, offset, box = max(candidates, key=lambda row:row[0])
        refined = self.hand.predict(image, box)
        self.calls[frame.camera_id] += 1
        self.last[key] = frame.captured_at_s
        person = observation.instances[index]
        points = list(person.keypoints_xyc)
        for joint, point in enumerate(refined):
            if point[2] >= self.threshold:
                points[offset+joint] = tuple(map(float, point))
                self.accepted += 1
        instances = list(observation.instances)
        instances[index] = replace(person, keypoints_xyc=tuple(points))
        return replace(observation, instances=tuple(instances))


class NoPose:
    def refine(self, observation, image):
        return observation


def rates(counts):
    tp, fp, fn = (counts[k] for k in ('true_positive','false_positive','false_negative'))
    return dict(tp=tp, fp=fp, fn=fn, recall=tp/max(1,tp+fn), precision=tp/max(1,tp+fp))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('docs/phase2-protocol.json'))
    parser.add_argument('--tracking-only', action='store_true')
    args = parser.parse_args()
    protocol_path = args.protocol
    plan = json.loads(protocol_path.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.tracking_only:
        from selfwatch_v2.gpu_runtime import activate
        activate()
    import cv2
    from selfwatch_v2.hand_refinement import HandPose
    from selfwatch_v2.identity import SharedIdentity
    from selfwatch_v2.legacy_guard import LegacyIdentityGuard
    from selfwatch_v2.pipeline import process
    from selfwatch_v2.pose_schedule import ScheduledPose
    from selfwatch_v2.replay import load_manifest, replay
    from selfwatch_v2.store import ObservationStore
    from selfwatch_v2.studio import Studio

    labels = {(row['clip'],row['camera'],row['index']):row for row in plan['frames']}
    report = dict(protocol_sha256=sha256(protocol_path.read_bytes()).hexdigest(), cases={})
    studio = Studio(Path.cwd(), pose_device='cuda', reid_graph=True)
    with (args.output/'engine.log').open('w',encoding='utf-8') as log, redirect_stdout(log):
        models = studio._load_models(not args.tracking_only)
        if 'detector' in plan:
            from selfwatch_v2.yolo import YoloPerception
            setting = plan['detector']
            models['detector'] = YoloPerception(setting['weights'],'0',setting['imgsz'],setting['confidence'],half=setting['half'])
            if models['detector'].provenance['sha256'] != setting['sha256']:
                raise ValueError('Frozen detector checkpoint differs')
            from selfwatch_v2.contracts import FrameRef
            import numpy as np
            warm = FrameRef('warmup','synthetic',0,0,0,600,600,'synthetic')
            models['detector'].predict(warm,np.zeros((600,600,3),np.uint8))
        models['detector'].include_masks = False
        report['models'] = {name:models[name].provenance for name in ('detector','appearance')}
        cases = ('native','oriented')
        if not args.tracking_only:
            hand = HandPose('weights/v2/hand/pose.onnx', 'cuda')
            report['models']['pose'] = models['pose'].provenance
            report['models']['hand'] = hand.provenance
            cases += ('sampled_hand',)
        for case in cases:
            boxes, slices, views = Counter(), defaultdict(Counter), defaultdict(Counter)
            clip_results = {}
            reviewed = set()
            for clip in plan['clips']:
                sources = load_manifest(Path(f'datasets/retailaction-replay-verified/{clip}.json'))
                trackers = {s.camera_id:LegacyIdentityGuard() for s in sources}
                encoder = models['appearance']
                encoder.states.clear()
                encoder.attention.clear()
                detector = models['detector'] if case == 'native' else OrientedDetector(models['detector'],plan['rotations'][clip])
                pose = NoPose()
                if case == 'sampled_hand':
                    pose = SampledHand(ScheduledPose(models['pose']), hand, plan['hand_interval_s'],plan['hand_cutoff'])
                counts = Counter()
                started = time.perf_counter()
                try:
                    with ObservationStore(args.output/f'{case}-{clip}.sqlite') as database:
                        for observation, image, tracks, _, timings in process(replay(sources),detector,pose,trackers,encoder,SharedIdentity(allow_overlap=True),parallel=False):
                            database.append(observation)
                            if not cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[0]:
                                raise RuntimeError('JPEG encode failed')
                            json.dumps(tracks,allow_nan=False)
                            camera = observation.frame.camera_id
                            counts[camera] += 1
                            key = (clip,camera,observation.frame.frame_index)
                            if key not in labels:
                                continue
                            row = labels[key]
                            if sha256(image.tobytes()).hexdigest() != row['sha256']:
                                raise ValueError('Labelled image differs from replay')
                            reviewed.add(key)
                            predictions = [p.box_xyxy for p in observation.instances if p.label == 'person']
                            match = match_boxes(row['boxes'],predictions,plan['iou'])
                            tally = {k:match[k] for k in ('true_positive','false_positive','false_negative')}
                            boxes.update(tally)
                            views[f'{clip}/{camera}'].update(tally)
                            slices[row['view']].update(tally)
                            slices[row['visibility']].update(tally)
                finally:
                    for tracker in trackers.values():
                        tracker.close()
                seconds = time.perf_counter()-started
                clip_results[clip] = dict(seconds=seconds,frames=dict(counts),fps={k:n/seconds for k,n in counts.items()},
                    hand_calls=dict(getattr(pose,'calls',{})), accepted_hand_points=getattr(pose,'accepted',0))
            if reviewed != set(labels):
                raise ValueError('Missing reviewed frames')
            score = rates(boxes)
            per_view = {k:rates(v) for k,v in views.items()}
            passed = (score['tp']+score['fn'] >= plan['minimum_people'] and score['precision'] >= plan['min_precision']
                      and score['recall'] >= plan['min_recall'] and all(v['recall']>=plan['min_view_recall'] for v in per_view.values())
                      and all(fps>=plan['min_fps_each_view'] for clip in clip_results.values() for fps in clip['fps'].values()))
            report['cases'][case] = dict(score=score, per_view=per_view, slices={k:rates(v) for k,v in slices.items()}, clips=clip_results,passed=passed)
            (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        studio.close()
    report['selected'] = next((case for case in plan['selection_order'] if report['cases'][case]['passed']),None)
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({case:dict(score=row['score'], passed=row['passed']) for case,row in report['cases'].items()},indent=2))


if __name__ == '__main__':
    main()
