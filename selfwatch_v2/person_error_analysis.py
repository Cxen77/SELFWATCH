"""Paired detector diagnostics from saved evidence; no new model inference."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.finish_person_pilot import compare
from .overhead_data import contained


def size_bucket(box,width,height):
    x1,y1,x2,y2=box
    area=(x2-x1)*(y2-y1)/(width*height)
    return 'below_0.1pct' if area<.001 else '0.1_to_1pct' if area<.01 else 'at_least_1pct'


def detected_targets(frame,case):
    result=frame['cases'][case];matches=result['matches']
    targets=[m['gt'] for m in matches];predictions=[m['prediction'] for m in matches]
    if (len(set(targets))!=len(targets) or len(set(predictions))!=len(predictions)
        or any(type(i) is not int or not 0<=i<len(frame['ground_truth']) for i in targets)
        or any(type(i) is not int or not 0<=i<len(result['predictions']) for i in predictions)
        or any(not .5<=m['iou']<=1 for m in matches)
        or len(matches)!=result['true_positive']
        or len(frame['ground_truth'])-len(matches)!=result['false_negative']
        or len(result['predictions'])-len(matches)!=result['false_positive']):
        raise ValueError('Saved detection matches are inconsistent')
    return set(targets)


def summarize_pair(baseline,adapted,case='native_640'):
    compare(baseline,adapted)
    scenes=defaultdict(lambda:defaultdict(Counter));sizes=defaultdict(lambda:defaultdict(Counter));images=[]
    for before,after in zip(baseline['frames'],adapted['frames']):
        if before['partition'] not in ('development','validation'):raise ValueError('Forbidden diagnostic partition')
        found={name:detected_targets(frame,case) for name,frame in (('pretrained',before),('adapted',after))}
        role=before['partition'];scene=before['source_group']
        for name,frame in (('pretrained',before),('adapted',after)):
            scenes[(role,scene)][name].update({k:frame['cases'][case][k] for k in
                ('true_positive','false_positive','false_negative')})
            scenes[(role,scene)][name]['images']+=1
            for index,target in enumerate(frame['ground_truth']):
                bucket=size_bucket(target['box_xyxy'],frame['width'],frame['height'])
                sizes[(role,bucket)][name]['targets']+=1
                sizes[(role,bucket)][name]['recalled']+=int(index in found[name])
        images.append(dict(image_id=before['image_id'],source_group=scene,partition=role,
            rescued=len(found['adapted']-found['pretrained']),lost=len(found['pretrained']-found['adapted']),
            persistent_misses=len(before['ground_truth'])-len(found['pretrained']|found['adapted']),
            adapted_misses=after['cases'][case]['false_negative']))
    scene_rows=[]
    for (role,scene),models in sorted(scenes.items()):
        for stats in models.values():
            tp,fp,fn=(stats[k] for k in ('true_positive','false_positive','false_negative'))
            stats['recall']=tp/max(1,tp+fn);stats['precision']=tp/max(1,tp+fp)
            stats['F1']=2*tp/max(1,2*tp+fp+fn)
        scene_rows.append(dict(partition=role,scene=scene,models=dict(models)))
    size_rows=[]
    for (role,bucket),models in sorted(sizes.items()):
        for stats in models.values():stats['recall']=stats['recalled']/stats['targets']
        size_rows.append(dict(partition=role,bucket=bucket,models=dict(models)))
    return dict(case=case,scenes=scene_rows,sizes=size_rows,images=images,
        size_protocol='Clipped enclosing GT box area divided by full input image area; fixed bins <0.1%, 0.1%-1%, >=1%. Not occlusion labels.',
        limitations='Single-seed diagnostic after validation-based checkpoint selection; not untouched test performance or tracking/hand accuracy.')


def draw_panel(root,frame,case,title,ground_truth=False):
    image=cv2.imread(str(contained(root,frame['image'])))
    if image is None or sha256(image.tobytes()).hexdigest()!=frame['processed_pixel_sha256']:
        raise ValueError('Diagnostic image differs from scored pixels')
    h,w=image.shape[:2];scale=min(384/w,216/h)
    resized=cv2.resize(image,(round(w*scale),round(h*scale)))
    panel=np.full((256,384,3),245,np.uint8);panel[:resized.shape[0],:resized.shape[1]]=resized
    boxes=[g['box_xyxy'] for g in frame['ground_truth']] if ground_truth else [p['box_xyxy'] for p in frame['cases'][case]['predictions']]
    matched={m['prediction'] for m in frame['cases'][case]['matches']}
    for index,box in enumerate(boxes):
        x1,y1,x2,y2=(round(v*scale) for v in box)
        color=(0,170,0) if ground_truth or index in matched else (0,0,230)
        cv2.rectangle(panel,(x1,y1),(x2,y2),color,1)
    cv2.putText(panel,title,(5,232),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1,cv2.LINE_AA)
    cv2.putText(panel,frame['image_id'],(5,249),cv2.FONT_HERSHEY_SIMPLEX,.32,(20,20,20),1,cv2.LINE_AA)
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();roots={name:args.comparison/name for name in ('pretrained','adapted')}
    reports={name:json.loads((root/'data.json').read_text()) for name,root in roots.items()}
    if any(report.get('source_changed_during_run') is not False for report in reports.values()):
        raise ValueError('Benchmark code stability not established')
    result=summarize_pair(reports['pretrained'],reports['adapted'])
    args.output.mkdir(parents=True,exist_ok=False)
    # One declared worst remaining-miss image per validation scene, before viewing figures.
    candidates=[r for r in result['images'] if r['partition']=='validation']
    selected=[min([r for r in candidates if r['source_group']==group],
        key=lambda r:(-r['adapted_misses'],r['image_id'])) for group in sorted({r['source_group'] for r in candidates})]
    panels=[]
    for row in selected:
        frames={name:next(f for f in report['frames'] if f['image_id']==row['image_id']) for name,report in reports.items()}
        parts=[draw_panel(roots['pretrained'],frames['pretrained'],result['case'],'Ground truth / green',True)]
        for name in ('pretrained','adapted'):
            stats=frames[name]['cases'][result['case']]
            title=f'{name}: TP {stats["true_positive"]} / FP {stats["false_positive"]} / FN {stats["false_negative"]}'
            parts.append(draw_panel(roots[name],frames[name],result['case'],title))
        panels.append(np.concatenate(parts,axis=1))
    figure=args.output/'remaining-misses.png'
    if not cv2.imwrite(str(figure),np.concatenate(panels,axis=0)):raise RuntimeError('Cannot write diagnostic figure')
    result['figure_selection']=selected
    result['sources']={name:sha256((root/'data.json').read_bytes()).hexdigest() for name,root in roots.items()}
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Person-detector error analysis','',result['limitations'],'',result['size_protocol'],'',
        '| Validation scene | Model | TP | FP | FN | Recall | F1 |','|---|---|---:|---:|---:|---:|---:|']
    for row in result['scenes']:
        if row['partition']!='validation':continue
        for model,stats in row['models'].items():lines.append(f'| {row["scene"]} | {model} | {stats["true_positive"]} | {stats["false_positive"]} | {stats["false_negative"]} | {stats["recall"]:.3f} | {stats["F1"]:.3f} |')
    lines+=['','| Validation target area | Model | Recalled | Targets | Recall |','|---|---|---:|---:|---:|']
    for row in result['sizes']:
        if row['partition']!='validation':continue
        for model,stats in row['models'].items():lines.append(f'| {row["bucket"]} | {model} | {stats["recalled"]} | {stats["targets"]} | {stats["recall"]:.3f} |')
    lines+=['','Figure: one image with most adapted-model misses per validation scene (ties: image ID). Green predictions match GT at IoU >=0.5; red predictions are unmatched. No final-test image is read.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),scenes=result['scenes'],sizes=result['sizes']),indent=2))


if __name__=='__main__':main()
