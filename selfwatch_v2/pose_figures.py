"""Export diagnostic figures from saved results, without running inference."""
import argparse
import gzip
import json
from pathlib import Path
import cv2
import numpy as np


def read_cache(path):
    with gzip.open(path,'rt',encoding='utf-8') as source:return {row['image_id']:row for row in map(json.loads,source)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    whole=Path('runs/wholebody-validation-v2');hand=Path('runs/hand-refinement-validation-gpu-v1')
    w=json.loads((whole/'results.json').read_text());h=json.loads((hand/'results.json').read_text())
    device=json.loads(Path('runs/hand-device-comparison.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),gridspec_kw={'width_ratios':[1,1.8,1]})
    groups=[(['GT person\nboxes','Detector\nboxes'],[100*w['metrics'][case]['body_visible']['PCK'] for case in ('oracle_crops','detector_pipeline')]),
        (['Whole-body\nGT boxes','Whole-body\npipeline','Hand model\nGT boxes','Hand ROI\npipeline'],[100*h['metrics'][case]['hands']['PCK'] for case in ('wholebody_oracle','wholebody_pipeline','hand_oracle','roi_refined_pipeline')]),
        (['CPU\n2 threads','CUDA'],[device['cpu']['mean'],device['cuda']['mean']])]
    titles=['Visible-body PCK (%)','Reliable-hand PCK (%)','Added hand inference (ms/image)']
    subtitles=['5% of sqrt(person box area)','10% of sqrt(hand box area)','Same cached inputs; one run/device']
    for index,(ax,(labels,values)) in enumerate(zip(axes,groups)):
        bars=ax.bar(labels,values,color=['#9ba8b2','#246d91','#7b8d9d','#227c5b'][:len(values)],width=.65)
        ax.set_title(titles[index],fontsize=11,pad=25);ax.text(.5,1.02,subtitles[index],ha='center',transform=ax.transAxes,fontsize=8)
        ax.set_ylim(0,100 if index<2 else max(values)*1.25)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9);ax.tick_params(axis='x',labelsize=8)
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.text(.5,.025,'64 public COCO-WholeBody validation images. PCK diagnostics, not official AP, unseen retail accuracy or live FPS.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(args.output/'summary.png',dpi=160)
    fig.savefig(args.output/'summary.svg');plt.close(fig)
    dataset=Path('datasets/coco-wholebody-pilot-v2');selection=json.loads((dataset/'selection.json').read_text())
    annotations={a['id']:a for a in selection['annotations']};images={i['id']:i for i in selection['images']}
    before=read_cache(whole/'predictions.jsonl.gz');after=read_cache(hand/'refined.jsonl.gz');candidates=[]
    for image_id,row in before.items():
        people=[p for p in row['detector']['instances'] if p['label']=='person']
        for match in row['matches']['matches']:
            person=match['prediction'];annotation=annotations[row['ground_truth_ids'][match['gt']]]
            if people[person]['keypoints_xyc'] is None:continue
            for side,offset in (('lefthand',91),('righthand',112)):
                box=annotation[side+'_box'];points=np.array(annotation[side+'_kpts']).reshape(21,3)
                if annotation[side+'_valid'] and np.any(points[:,2]>0) and min(box[2:])>0:
                    candidates.append(dict(image_id=image_id,annotation_id=annotation['id'],person=person,
                        side=side,offset=offset,area=box[2]*box[3],box=box))
    selected=[];seen=set()
    for row in sorted(candidates,key=lambda r:(-r['area'],r['image_id'],r['annotation_id'],r['side'])):
        if row['image_id'] in seen:continue
        selected.append(row);seen.add(row['image_id'])
        if len(selected)==3:break
    sheet=np.full((len(selected)*290,960,3),245,np.uint8)
    edges=[(0,base) for base in (1,5,9,13,17)]+[(base+j,base+j+1) for base in (1,5,9,13,17) for j in range(3)]
    for number,row in enumerate(selected):
        image_id=row['image_id'];image=cv2.imread(str(dataset/'images'/images[image_id]['file_name']))
        x,y,width,height=row['box'];cx=x+width/2;cy=y+height/2;radius=max(width,height)*.9
        x1=max(0,int(cx-radius));y1=max(0,int(cy-radius));x2=min(image.shape[1],int(cx+radius)+1);y2=min(image.shape[0],int(cy+radius)+1)
        crop=image[y1:y2,x1:x2];scale=min(320/crop.shape[1],245/crop.shape[0])
        background=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
        gt=np.array(annotations[row['annotation_id']][row['side']+'_kpts']).reshape(21,3)
        people=[p for p in before[image_id]['detector']['instances'] if p['label']=='person']
        old=np.array(people[row['person']]['keypoints_xyc'])[row['offset']:row['offset']+21]
        new=np.array(after[image_id]['refined'][row['person']])[row['offset']:row['offset']+21]
        for col,(title,points,is_gt) in enumerate((('Ground truth',gt,True),('Whole-body prediction',old,False),('Predicted-ROI refinement',new,False))):
            tile=np.full((290,320,3),245,np.uint8);tile[:background.shape[0],:background.shape[1]]=background
            valid=points[:,2]>0 if is_gt else np.ones(21,dtype=bool)
            coords=(points[:,:2]-[x1,y1])*scale;color=(40,210,40) if is_gt else (210,50,190)
            for a,b in edges:
                if valid[a] and valid[b]:cv2.line(tile,tuple(coords[a].astype(int)),tuple(coords[b].astype(int)),color,1,cv2.LINE_AA)
            for point,ok in zip(coords,valid):
                if ok:cv2.circle(tile,tuple(point.astype(int)),2,color,-1)
            cv2.putText(tile,title,(5,262),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1,cv2.LINE_AA)
            cv2.putText(tile,f'{image_id:012d} / {row["side"]}',(5,281),cv2.FONT_HERSHEY_SIMPLEX,.38,(20,20,20),1,cv2.LINE_AA)
            sheet[number*290:(number+1)*290,col*320:(col+1)*320]=tile
    if not cv2.imwrite(str(args.output/'hand_examples.png'),sheet):raise RuntimeError('Cannot write review figure')
    (args.output/'figure_notes.json').write_text(json.dumps(dict(selection='Three largest valid GT hand areas among matched/posed people, at most one per image; selected without looking at improvement.',
        selected=selected,zoom='GT-centered zoom for review only; NOT used to construct inference ROIs. Out-of-crop predictions may be invisible in the figure but are scored in full-image coordinates.',
        score='All predicted points drawn to inspect geometry; drawing is not a visibility claim.'),indent=2),encoding='utf-8')
    print(args.output.resolve())


if __name__=='__main__':main()
