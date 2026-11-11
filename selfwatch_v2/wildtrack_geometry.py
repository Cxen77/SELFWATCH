"""Fit geometry gates on a development segment, then score the frozen policy."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from scripts.download_wildtrack_pilot import ARCHIVE, CALIBRATION_CAMERAS, CALIBRATION_FILES
from .evaluate_tracking import evaluate
from .ground_link import GroundCamera, GroundLinker
from .identity_study import digest
from .overhead_data import contained
from .wildtrack_tracking import REFERENCE, marks_for_camera, pair_scores

DEVELOPMENT=tuple(range(1700,1790,5))
EVALUATION=tuple(range(1300,1360,5))
RADII=(.35,.50,.75,1.)
MARGINS=(.15,.25,.35)
COSINES=(.35,.45,.55,.65)
MIN_PRECISION=.95
MIN_CORRECT=20


def load_cameras(root):
    manifest=root/'calibration-manifest.json'
    data=json.loads(manifest.read_text())
    records=data['files']
    if data['etag']!=ARCHIVE[1] or len(records)!=len(CALIBRATION_FILES) or {r['path'] for r in records}!=set(CALIBRATION_FILES):
        raise ValueError('Unexpected camera calibration selection')
    for record in records:
        path=contained(root,record['path'])
        if digest(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:
            raise ValueError('Camera calibration hash mismatch')
    cameras={}
    for camera,name in CALIBRATION_CAMERAS.items():
        intr=ET.parse(root/f'calibrations/intrinsic_zero/intr_{name}.xml').getroot()
        ext=ET.parse(root/f'calibrations/extrinsic/extr_{name}.xml').getroot()
        intrinsic=np.array(intr.findtext('camera_matrix/data').split(),dtype=float).reshape(3,3)
        if np.any(np.array(intr.findtext('distortion_coefficients/data').split(),dtype=float)!=0):
            raise ValueError('Expected undistorted camera intrinsics')
        rotation=cv2.Rodrigues(np.array(ext.findtext('rvec').split(),dtype=float))[0]
        translation=np.array(ext.findtext('tvec').split(),dtype=float)/100
        image_from_floor=intrinsic@np.column_stack((rotation[:,:2],translation))
        cameras[f'C{camera}']=GroundCamera(image_from_floor)
    return cameras,digest(manifest)


def source_consistency(cameras,dataset):
    """Verify published projected boxes agree with the archive's ground grid."""
    manifest=json.loads((dataset/'manifest.json').read_text())
    if tuple(manifest['frames'])!=DEVELOPMENT or manifest['etag']!=ARCHIVE[1]:
        raise ValueError('Calibration consistency check requires the development archive')
    hashes={row['path']:row['sha256'] for row in manifest['files']}
    errors=defaultdict(list)
    for index in DEVELOPMENT:
        relative=f'annotations_positions/{index:08d}.json'
        path=dataset/relative
        if digest(path)!=hashes[relative]:
            raise ValueError('Development labels changed')
        rows=json.loads(path.read_text())
        positions={str(r['personID']):r['positionID'] for r in rows}
        for name,camera in cameras.items():
            for mark in marks_for_camera(rows,int(name[1:]),camera.width,camera.height):
                point=camera.foot(mark['xyxy'])
                if point is None:
                    continue
                position=positions[mark['subject']]
                if not 0<=position<480*1440:
                    raise ValueError('Invalid published grid index')
                world=(-3+.025*(position%480),-9+.025*(position//480))
                errors[name].append(float(np.linalg.norm(point-world)))
    result={}
    for name in cameras:
        values=errors[name]
        if len(values)<20 or np.median(values)>.4:
            raise ValueError(f'{name}: published boxes and camera projection disagree')
        result[name]=dict(boxes=len(values),median_m=float(np.median(values)),p95_m=float(np.percentile(values,95)))
    return result


def load_run(root,expected):
    result=json.loads((root/'results.json').read_text())
    for file,key in (('predictions.json','predictions_sha256'),('annotations.json','annotations_sha256'),
                     ('features.npz','features_sha256')):
        if digest(root/file)!=result[key]:
            raise ValueError('Frozen tracking run changed')
    data=json.loads((root/'predictions.json').read_text())
    labels=json.loads((root/'annotations.json').read_text())
    wanted={(i,f'C{c}') for i in expected for c in CALIBRATION_CAMERAS}
    got={(f['frame']['frame_index'],f['frame']['camera_id']) for f in data['frames']}
    if got!=wanted or len(data['frames'])!=len(wanted) or len(labels['frames'])!=len(wanted):
        raise ValueError('Expected complete, non-overlapping synchronized segment')
    if any(f['frame']['captured_at_s']!=f['frame']['frame_index']/10 for f in data['frames']):
        raise ValueError('Unexpected capture timestamps')
    return data,labels,result


def visible(track):
    return (track.get('global_id') is not None and track.get('status','observed')=='observed'
            and not track.get('visual_suppressed'))


def prepare(data,cameras,vectors):
    buckets=defaultdict(list)
    for row in data['frames']:
        frame=row['frame'];name=frame['camera_id'];camera=cameras[name]
        if (frame['width'],frame['height'])!=(camera.width,camera.height):
            raise ValueError('Frame geometry differs from calibration')
        points={};features={}
        for track in row['tracks_by_case']['camera_local']:
            if not visible(track):
                continue
            key=(name,frame['stream_epoch'],track['local_id'])
            if key in points:
                raise ValueError('Duplicate observed tracklet in a camera')
            xy=camera.foot(row['instances'][track['instance_index']]['box_xyxy'])
            # Restrict links to the documented labelled ground region.
            points[key]=xy if xy is not None and -3<=xy[0]<9 and -9<=xy[1]<27 else None
            feature_key=row['feature_keys'].get(str(track['instance_index']))
            if feature_key is not None:
                vector=np.array(vectors[feature_key],dtype=float)
                length=np.linalg.norm(vector)
                if vector.ndim!=1 or not np.isfinite(length) or length<1e-8:
                    raise ValueError('Invalid cached appearance embedding')
                features[key]=vector/length
        buckets[frame['captured_at_s']].append((row,points,features))
    return sorted(buckets.items())


def run_policy(data,labels,bundles,settings):
    linker=GroundLinker(**settings) if settings is not None else None
    frames=[]
    for timestamp,rows in bundles:
        views={row['frame']['camera_id']:points for row,points,_ in rows}
        features={key:vector for _,_,items in rows for key,vector in items.items()}
        identities=linker.update(timestamp,views,features) if linker else {}
        for row,_,_ in rows:
            frame=row['frame']
            tracks=[dict(t,global_id=identities.get((frame['camera_id'],frame['stream_epoch'],t['local_id']),t['global_id']))
                    for t in row['tracks_by_case']['camera_local'] if visible(t)]
            frames.append(dict(row,tracks_by_case=dict(row['tracks_by_case'],geometry=tracks)))
    predictions=dict(data,frames=frames,tracking_cases=dict(data['tracking_cases'],geometry='Calibrated floor, two sightings'))
    result=evaluate(predictions,labels,case='geometry',reference_source=REFERENCE)
    result.update(pair_scores(labels['frames'],result['frame_matches']))
    return result,linker.events if linker else [],predictions


def summary(result):
    names=('IDF1','same_person_possible','same_person_both_matched',
           'same_person_shared_id','different_person_shared_id')
    row={k:result[k] for k in names}
    good=row['same_person_shared_id'];wrong=row['different_person_shared_id']
    row['matched_pair_precision']=good/(good+wrong) if good+wrong else None
    return row


def freeze(output,payload):
    with output.open('x',encoding='utf-8') as target:
        json.dump(payload,target,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('fit','evaluate'))
    for name in ('run','calibration','policy','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--development-dataset',type=Path,default=Path('datasets/wildtrack-pilot-v1'))
    args=parser.parse_args()
    cameras,calibration_sha=load_cameras(args.calibration)
    code={name:digest(Path(__file__).with_name(name)) for name in ('wildtrack_geometry.py','ground_link.py','evaluate_tracking.py')}
    data,labels,baseline=load_run(args.run,DEVELOPMENT if args.mode=='fit' else EVALUATION)
    with np.load(args.run/'features.npz',allow_pickle=False) as vectors:
        bundles=prepare(data,cameras,vectors)
    if args.mode=='fit':
        consistency=source_consistency(cameras,args.development_dataset)
        if digest(args.development_dataset/'manifest.json')!=baseline['dataset_manifest_sha256']:
            raise ValueError('Development predictions and calibration check use different data')
        rows=[]
        grid=[dict(radius=radius,margin=margin) for radius,margin in product(RADII,MARGINS)]
        grid.extend(dict(radius=radius,margin=.25,min_cosine=cosine) for radius,cosine in product(RADII,COSINES))
        for settings in grid:
            score,_,_=run_policy(data,labels,bundles,settings)
            rows.append(dict(settings=settings,**summary(score)))
        viable=[row for row in rows if row['same_person_shared_id']>=MIN_CORRECT
                and (row['matched_pair_precision'] or 0)>=MIN_PRECISION]
        selected=max(viable,key=lambda r:(r['same_person_shared_id'],r['matched_pair_precision'],
            -r['settings']['radius'],r['settings']['margin'])) if viable else None
        # Retain the strongest development diagnostic for each family even if
        # the deployment gate rejected it. These are explicitly shadow arms.
        shadows={}
        for name,hybrid in (('geometry_only',False),('geometry_with_appearance',True)):
            candidates=[row for row in rows if ('min_cosine' in row['settings'])==hybrid
                        and row['same_person_shared_id']>=MIN_CORRECT]
            shadows[name]=max(candidates,key=lambda r:(r['matched_pair_precision'],r['same_person_shared_id'],
                -r['settings']['radius'],r['settings']['margin'])) if candidates else None
        policy=dict(settings=selected['settings'] if selected else None,calibration_sha256=calibration_sha,
            source_sha256=code,development_predictions_sha256=baseline['predictions_sha256'],
            development_labels_sha256=baseline['annotations_sha256'],model_sha256=baseline['detector']['sha256'],
            reid_sha256=baseline['reid']['sha256'],minimum_correct=MIN_CORRECT,minimum_precision=MIN_PRECISION,
            score_grid=rows,chosen=selected,development_shadow_arms=shadows,source_consistency_m=consistency)
        freeze(args.policy,policy)
        result=dict(development=selected,shadow_arms=shadows,source_consistency_m=consistency,policy_sha256=digest(args.policy))
    else:
        policy=json.loads(args.policy.read_text())
        if (policy['calibration_sha256'],policy['source_sha256'],policy['model_sha256'],policy['reid_sha256'])!=(
            calibration_sha,code,baseline['detector']['sha256'],baseline['reid']['sha256']):
            raise ValueError('Frozen calibration, models or evaluation code changed')
        score,events,predictions=run_policy(data,labels,bundles,policy['settings'])
        shadow_results={}
        for name,arm in policy['development_shadow_arms'].items():
            if arm:
                arm_score,arm_events,_=run_policy(data,labels,bundles,arm['settings'])
                shadow_results[name]=dict(settings=arm['settings'],**summary(arm_score),link_events=len(arm_events))
        stats=summary(score)
        result=dict(scope='Separate time segment of the same outdoor scene, with fixed local tracks and projected reference boxes. Not independent retail or live validation.',
            policy_sha256=digest(args.policy),settings=policy['settings'],detector_counts=baseline['counts'],
            appearance=summary(baseline['scores']['legacy']),camera_local=summary(baseline['scores']['camera_local']),
            geometry=stats,geometry_events=events,shadow_arms=shadow_results,
            passed_evaluation=(stats['matched_pair_precision'] or 0)>=MIN_PRECISION and stats['same_person_shared_id']>=MIN_CORRECT,
            evaluation_predictions_sha256=baseline['predictions_sha256'],evaluation_labels_sha256=baseline['annotations_sha256'])
        freeze(args.output.with_name(args.output.stem+'-predictions.json'),predictions)
    freeze(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='geometry_events'},indent=2))


if __name__=='__main__':
    main()
