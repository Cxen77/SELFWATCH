"""Verify exported pixels/labels and render deterministic development target sheets."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import cv2
import numpy as np
from .overhead_data import contained


def audit(dataset,output):
    manifest=json.loads((dataset/'export_manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    records=[];counts=Counter()
    for item in manifest['items']:
        role=item['partition']
        if role not in ('development','validation'):raise ValueError('Forbidden export partition')
        image_path=contained(dataset,item['image']);label_path=contained(dataset,item['labels'])
        image=cv2.imread(str(image_path))
        if image is None:raise ValueError('Exported image cannot be decoded')
        if sha256(image.tobytes()).hexdigest()!=item['pixel_sha256']:
            raise ValueError('Exported pixels differ from the preprocessing manifest')
        if sha256(label_path.read_bytes()).hexdigest()!=item['label_sha256']:
            raise ValueError('Exported labels changed')
        labels=[]
        for line in label_path.read_text().splitlines():
            if not line.strip():continue
            values=list(map(float,line.split()))
            if (len(values)!=5 or values[0]!=0 or not all(np.isfinite(values))
                or not all(0<=v<=1 for v in values[1:]) or min(values[3:])<=0):
                raise ValueError('Invalid normalized person target')
            _,x,y,w,h=values
            if min(x-w/2,y-h/2)<-1e-8 or max(x+w/2,y+h/2)>1+1e-8:
                raise ValueError('Person target exceeds original image bounds')
            labels.append(values[1:])
        if len(labels)!=item['people']:raise ValueError('Target count differs from manifest')
        counts[role+'_images']+=1;counts[role+'_people']+=len(labels)
        records.append(dict(image_id=item['image_id'],source_group=item['source_group'],partition=role,
            labels=labels,image=item['image'],min_area=min((w*h for _,_,w,h in labels),default=1),
            people=len(labels)))
    if dict(counts)!=manifest['counts']:raise ValueError('Export counts differ from manifest')
    groups=sorted({r['source_group'] for r in records if r['partition']=='development'})
    sheets={}
    for criterion in ('smallest_target','most_people'):
        chosen=[]
        for group in groups:
            candidates=[r for r in records if r['source_group']==group and r['people']]
            key=(lambda r:(r['min_area'],r['image_id'])) if criterion=='smallest_target' else (lambda r:(-r['people'],r['image_id']))
            chosen.append(min(candidates,key=key))
        sheet=np.full((((len(chosen)+3)//4)*240,4*320,3),245,np.uint8)
        for index,row in enumerate(chosen):
            image=cv2.imread(str(contained(dataset,row['image'])));height,width=image.shape[:2]
            tile=cv2.resize(image,(320,180))
            for x,y,w,h in row['labels']:
                cv2.rectangle(tile,(round((x-w/2)*320),round((y-h/2)*180)),
                    (round((x+w/2)*320),round((y+h/2)*180)),(30,230,30),1)
            dx=(index%4)*320;dy=(index//4)*240
            sheet[dy:dy+180,dx:dx+320]=tile
            for offset,text in enumerate((row['source_group'],f'{row["people"]} people / min area {row["min_area"]:.5f}',row['image_id'])):
                cv2.putText(sheet,text,(dx+5,dy+196+offset*16),cv2.FONT_HERSHEY_SIMPLEX,.38,(25,25,25),1,cv2.LINE_AA)
        name=criterion+'.png'
        if not cv2.imwrite(str(output/name),sheet):raise RuntimeError('Cannot save target sheet')
        sheets[criterion]=dict(path=name,image_ids=[r['image_id'] for r in chosen])
    report=dict(counts=dict(counts),sheets=sheets,
        export_manifest_sha256=sha256((dataset/'export_manifest.json').read_bytes()).hexdigest(),
        protocol='All exported pixel/label hashes, normalized geometry and counts verified; sheets use development only.',
        limitations='Sheets reveal target geometry and size, not visibility/occlusion ground truth or detector accuracy.',
        size_counts={role:dict(below_1pct=sum(w*h<.01 for r in records if r['partition']==role for _,_,w,h in r['labels']),
            below_point1pct=sum(w*h<.001 for r in records if r['partition']==role for _,_,w,h in r['labels']))
            for role in ('development','validation')})
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.dataset,args.output)


if __name__=='__main__':main()
