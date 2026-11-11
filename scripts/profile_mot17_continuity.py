"""Compare the frozen policy with the app's scheduled ReID and tracking stack."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import statistics
import time

from selfwatch_v2.mot_continuity import dataset, digest, frame_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import torch
    from selfwatch_v2.continuity_guard import ContinuityGuard
    from selfwatch_v2.identity import SharedIdentity
    from selfwatch_v2.legacy_guard import LegacyIdentityGuard
    from selfwatch_v2.mot_metrics import evaluate_mot
    from selfwatch_v2.pipeline import process
    from selfwatch_v2.store import ObservationStore
    from selfwatch_v2.studio import Studio

    root = Path('datasets/mot17-continuity-native-v1')
    sequence = 'MOT17-09-FRCNN'
    setting, dataset_hash = dataset(root, sequence)
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    selected = policy['selected_case']
    if selected == 'baseline':
        raise ValueError('The development experiment did not select a candidate')
    for path, checksum in policy['tracker_sources'].items():
        if digest(path) != checksum:
            raise ValueError(f'Frozen tracking source differs: {path}')

    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(sequence=sequence, frames=setting['frames'], cases={},
                  dataset_sha256=dataset_hash, policy_sha256=digest(args.policy),
                  scope='One warmed offline replay per policy. JPEG decode, tracking, SQLite, JPEG encoding and JSON serialization timed. Camera capture and browser delivery outside the benchmark.')
    torch.set_num_threads(2)
    studio = Studio(Path.cwd(), reid_graph=True)

    class NoPose:
        def refine(self, observation, image):
            return observation

    try:
        with (args.output/'model-load.log').open('w', encoding='utf-8') as load_log, redirect_stdout(load_log):
            models = studio._load_models(False)
        detector = models['detector']
        detector.include_masks = False
        encoder = models['appearance']
        signature = policy['perception_signature']
        if (detector.provenance['sha256'] != signature['detector_sha256'] or
                encoder.provenance['sha256'] != signature['reid_sha256']):
            raise ValueError('Model weights differ from the frozen experiment')

        for case in ('baseline', selected):
            encoder.states.clear()
            encoder.attention.clear()
            tracker = LegacyIdentityGuard() if case == 'baseline' else ContinuityGuard(policy=case)
            target = args.output/'trackers'/case/'data'/f'{sequence}.txt'
            target.parent.mkdir(parents=True)
            stream = (frame_image(root, sequence, index) for index in range(1, setting['frames']+1))
            stage_times = []
            reid_batches = []
            start = time.perf_counter()
            try:
                with (args.output/f'{case}.log').open('w', encoding='utf-8') as log, redirect_stdout(log), \
                        target.open('w', encoding='utf-8') as output, ObservationStore(args.output/f'{case}.sqlite') as observations:
                    results = process(stream, detector, NoPose(), {sequence: tracker}, encoder,
                                      SharedIdentity(allow_overlap=True), parallel=False)
                    for frame_index, (observation, image, tracks, appearance, timings) in enumerate(results, 1):
                        observations.append(observation)
                        if not cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])[0]:
                            raise RuntimeError('JPEG encoding failed')
                        json.dumps(dict(observation=observation.to_dict(), tracks=tracks, timings=timings), allow_nan=False)
                        stage_times.append(timings)
                        reid_batches.append(sum(item['reason'] == 'fresh_embedding' for item in appearance))
                        for track in tracks:
                            if track.get('status') != 'observed' or track.get('visual_suppressed'):
                                continue
                            person = observation.instances[track['instance_index']]
                            x1, y1, x2, y2 = [value*1.5 for value in person.box_xyxy]
                            output.write(f'{frame_index},{track["global_id"]},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{person.score:.6f},-1,-1,-1\n')
            finally:
                tracker.close()
            duration = time.perf_counter()-start
            score = evaluate_mot(root, args.output/'trackers', sequence, setting['frames'], case)
            score.update(observations_per_s=len(stage_times)/duration, wall_seconds=duration,
                         fresh_embeddings=sum(reid_batches), predictions_sha256=digest(target),
                         mean_stage_ms={key:statistics.mean(row.get(key, 0) for row in stage_times)
                                        for key in ('detector_ms','reid_ms','cognitive_engine_ms','association_ms')})
            report['cases'][case] = score
            (args.output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(case, round(score['observations_per_s'], 2), 'FPS', score['summary'], flush=True)
    finally:
        studio.close()


if __name__ == '__main__':
    main()
