"""Three-view identity diagnostic with published WILDTRACK person IDs."""
from collections import Counter,defaultdict
from contextlib import redirect_stdout
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

from .contracts import FrameRef
from .evaluate_tracking import evaluate
from .identity_study import Trial, digest
from .overhead_data import contained, match_boxes

REFERENCE='official:wildtrack'
SCOPE=('One outdoor WILDTRACK development segment at 2 Hz, using published projected boxes '
       'and online IDs. Full-view IDF1 charges unmatched predictions even outside the labelled '
       'ground area; it is a diagnostic, not the official WILDTRACK metric or retail accuracy.')


def marks_for_camera(people,camera,width,height):
    marks=[]
    for person in people:
        views=[v for v in person['views'] if v['viewNum']==camera-1]
        if len(views)!=1:
            raise ValueError('Expected one published view entry per person')
        view=views[0];box=[view[k] for k in ('xmin','ymin','xmax','ymax')]
        if box==[-1]*4:
            continue
        x1,y1,x2,y2=box
        box=[max(0,x1),max(0,y1),min(width,x2),min(height,y2)]
        if box[0]<box[2] and box[1]<box[3]:
            marks.append(dict(label='person',kind='box',source=REFERENCE,
                subject=str(person['personID']),xyxy=box))
    if len({m['subject'] for m in marks})!=len(marks):
        raise ValueError('Duplicate person in the published camera labels')
    return marks


def pair_scores(annotations,traces):
    """Use synchronized pairs; missing observations cannot inflate coverage."""
    seen=defaultdict(list)
    matched={row['observation_id']:{m['subject']:m['global_id'] for m in row['matches']} for row in traces}
    for row in annotations:
        seen[row['frame']['captured_at_s']].append(row)
    totals=Counter()
    for rows in seen.values():
        for a,b in combinations(rows,2):
            left=matched[a['observation_id']];right=matched[b['observation_id']]
            shared={m['subject'] for m in a['marks']} & {m['subject'] for m in b['marks']}
            both=left.keys() & right.keys()
            totals.update(same_person_possible=len(shared),same_person_both_matched=len(both),
                same_person_shared_id=sum(left[s]==right[s] for s in both),
                different_person_pairs=len(left)*len(right)-len(both),
                different_person_shared_id=sum(x!=y and gid==other
                    for x,gid in left.items() for y,other in right.items()))
    return dict(totals)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/wildtrack-pilot-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from .appearance import AppearanceEncoder
    from .yolo import YoloPerception
    from scripts.download_wildtrack_pilot import CALIBRATION_FILES, IDENTITY_CONTINUITY_HOLDOUT, IDENTITY_HOLDOUT
    torch.set_num_threads(2)
    manifest_path=args.dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    cameras=manifest['cameras'];frames=manifest['frames']
    development=frames==list(range(1700,1790,5))
    evaluation=frames==list(range(1300,1360,5))
    holdout=frames==list(IDENTITY_HOLDOUT)
    continuity_holdout=frames==list(IDENTITY_CONTINUITY_HOLDOUT)
    if cameras!=[1,3,5] or not (development or evaluation or holdout or continuity_holdout):
        raise ValueError('Unexpected camera or timestamp selection')
    expected={f'annotations_positions/{f:08d}.json' for f in frames}
    expected.update(f'Image_subsets/C{c}/{f:08d}.png' for c in cameras for f in frames)
    if evaluation or holdout or continuity_holdout:
        expected.update(CALIBRATION_FILES)
    files=manifest['files']
    if len(files)!=len(expected) or {r['path'] for r in files}!=expected:
        raise ValueError('Missing, duplicate or unexpected pilot members')
    for row in files:
        path=contained(args.dataset,row['path'])
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('Pilot member hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    detector=YoloPerception('weights/v2/yolo26n-seg.pt','0',640,.25)
    detector.include_masks=False
    encoder=AppearanceEncoder('weights/osnet/osnet_x1_0_msmt17.pth')
    sources=[SimpleNamespace(camera_id=f'C{c}') for c in cameras]
    trials={key:Trial(key,sources) for key in ('baseline','legacy','weak_births')}
    run_id=f'wildtrack-pilot:{digest(manifest_path)[:16]}'
    predictions=[];annotations=[];counts=Counter();vectors={}
    try:
        with (args.output/'engine.log').open('w',encoding='utf-8') as log:
            for f in frames:
                people=json.loads((args.dataset/f'annotations_positions/{f:08d}.json').read_text())
                for camera in cameras:
                    path=args.dataset/f'Image_subsets/C{camera}/{f:08d}.png'
                    image=cv2.imread(str(path))
                    if image is None or image.shape[:2]!=(1080,1920):
                        raise ValueError('Unexpected WILDTRACK image geometry')
                    frame=FrameRef(f'C{camera}','wildtrack-pilot',f,f/10,0,1920,1080,
                        str(path.resolve()),'wildtrack_synchronized_image_index/10')
                    with redirect_stdout(log):
                        observation,_=detector.predict(frame,image)
                        features,diagnostics,_=encoder.extract_all(observation,image)
                        cases={key:trial.step(observation,image,features)['tracks'] for key,trial in trials.items()}
                    feature_keys={str(i):f'f{len(predictions)}_i{i}' for i in features}
                    vectors.update({feature_keys[str(i)]:np.asarray(v,dtype=np.float32).copy() for i,v in features.items()})
                    cases['camera_local']=[dict(t,global_id=json.dumps([frame.camera_id,frame.stream_epoch,t['local_id']]))
                        for t in cases['legacy'] if t.get('global_id') is not None]
                    marks=marks_for_camera(people,camera,frame.width,frame.height)
                    predicted=[p.box_xyxy for p in observation.instances if p.label=='person']
                    matched=match_boxes([m['xyxy'] for m in marks],predicted,.5)
                    counts.update(observations=1,gt_boxes=len(marks),person_detections=len(predicted),
                                  detector_matches=len(matched['matches']))
                    predictions.append(dict(observation.to_dict(),observation_id=frame.observation_id,tracks_by_case=cases,
                        feature_keys=feature_keys,feature_diagnostics=diagnostics))
                    annotations.append(dict(observation_id=frame.observation_id,frame=predictions[-1]['frame'],
                        reviewed=True,exhaustive_labels=['person'],marks=marks))
                print(f'WILDTRACK frame {f}: {len(predictions)}/{len(frames)*len(cameras)} camera observations',flush=True)
    finally:
        for trial in trials.values():
            trial.close()
    data=dict(run_id=run_id,frames=predictions,tracking_cases={
        'baseline':'Motion/IoU + shared identity','legacy':'Original cognitive tracking + shared identity',
        'weak_births':'Tentative births + shared identity','camera_local':'Same legacy tracks; camera-scoped IDs'},
        identity_by_case={key:trial.identity.snapshot() for key,trial in trials.items()})
    labels=dict(run_id=run_id,source=REFERENCE,scope='Published people within annotated ground region; boxes clipped to image.',frames=annotations)
    for name,payload in (('predictions',data),('annotations',labels)):
        (args.output/f'{name}.json').write_text(json.dumps(payload,allow_nan=False),encoding='utf-8')
    np.savez_compressed(args.output/'features.npz',**vectors)
    scores={}
    for case in data['tracking_cases']:
        try:
            result=evaluate(data,labels,case=case,reference_source=REFERENCE)
        except ValueError as error:
            if not str(error).startswith('Two visible predictions own the same global ID'):
                raise
            scores[case]=dict(error=str(error))
            continue
        scores[case]=dict(result,**pair_scores(annotations,result['frame_matches']))
    report=dict(scope=SCOPE,dataset_manifest_sha256=digest(manifest_path),source=manifest['source'],
        dataset_partition=('development' if development else 'identity_continuity_holdout' if continuity_holdout
                           else 'identity_holdout' if holdout else 'evaluation'),
        detector=detector.provenance,reid=encoder.provenance,counts=dict(counts),scores=scores,
        feature_policy='Same detector and eligible OSNet crops across policies; refresh scheduling disabled in this fixed-input diagnostic.',
        predictions_sha256=digest(args.output/'predictions.json'),annotations_sha256=digest(args.output/'annotations.json'),
        features_sha256=digest(args.output/'features.npz'))
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({case:{k:v for k,v in row.items() if k in ('IDF1','error','same_person_possible',
        'same_person_both_matched','same_person_shared_id','different_person_shared_id')} for case,row in scores.items()},indent=2))


if __name__=='__main__':
    main()
