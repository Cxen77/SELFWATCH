"""Export person boxes for a separate detector experiment; never touch final test.

WEPDTOF supplies rotated person boxes, not segmentation masks or hand labels.
The export uses exactly the benchmark's ROI masking and enclosing-box protocol.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import cv2
from .overhead_data import load_selection,load_image


def label_line(box,width,height):
    x1,y1,x2,y2=box
    values=((x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height)
    if not all(0<=value<=1 for value in values) or min(values[2:])<=0:
        raise ValueError('Person target lies outside normalized image geometry')
    return '0 '+' '.join(f'{value:.9f}' for value in values)


def export_sources(dataset,validation_dataset=None):
    """Validate lineage and partitions before decoding any training pixels."""
    dataset=Path(dataset)
    selected,roles=load_selection(dataset)
    sources={'development':(dataset,selected),'validation':(dataset,selected)}
    if validation_dataset is not None:
        validation_dataset=Path(validation_dataset)
        parent,parent_roles=load_selection(validation_dataset)
        parent_hash=sha256((validation_dataset/'selection.json').read_bytes()).hexdigest()
        if (selected.get('parent_selection_sha256')!=parent_hash or roles!=parent_roles
            or selected['source']!=parent['source']):
            raise ValueError('Development extension does not match the frozen validation parent')
        if any(roles[i['source_group']]!='development' for i in selected['items']):
            raise ValueError('Extension contains a non-development scene')
        sources['validation']=(validation_dataset,parent)
    plan=[];seen=set()
    for role,(root,selection) in sources.items():
        items=[i for i in selection['items'] if roles[i['source_group']]==role]
        if not items:raise ValueError(f'No {role} images; supply the frozen validation parent')
        for item in items:
            name=item['image_id']
            if not isinstance(name,str) or Path(name).name!=name or any(c in name for c in '/\\:'):
                raise ValueError('Unsafe source image identifier')
            if name in seen:raise ValueError('Duplicate image across exported partitions')
            seen.add(name);plan.append((role,root,item))
    return selected,roles,sources,plan


def prepare(dataset,output,validation_dataset=None):
    selected,roles,sources,plan=export_sources(dataset,validation_dataset)
    dataset=Path(dataset);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[];counts=Counter()
    for role in ('development','validation'):
        (output/'images'/role).mkdir(parents=True)
        (output/'labels'/role).mkdir(parents=True)
    for role,root,item in plan:
        image,_,gt,excluded=load_image(root,item)
        height,width=image.shape[:2];name=item['image_id']
        image_path=output/'images'/role/(name+'.png')
        label_path=output/'labels'/role/(name+'.txt')
        if not cv2.imwrite(str(image_path),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
            raise RuntimeError('Cannot export training pixels')
        label_path.write_text('\n'.join(label_line(g['box_xyxy'],width,height) for g in gt)+'\n',encoding='utf-8')
        rows.append(dict(image_id=name,source_group=item['source_group'],partition=role,
            image=str(image_path.relative_to(output)),labels=str(label_path.relative_to(output)),
            source_sha256=item['image_sha256'],pixel_sha256=sha256(image.tobytes()).hexdigest(),
            label_sha256=sha256(label_path.read_bytes()).hexdigest(),people=len(gt),excluded=excluded,
            source_root=str(root.resolve()),width=width,height=height))
        counts[role+'_images']+=1;counts[role+'_people']+=len(gt)
    (output/'data.yaml').write_text(json.dumps(dict(path=output.resolve().as_posix(),
        train='images/development',val='images/validation',names={0:'person'}),indent=2),encoding='utf-8')
    manifest=dict(schema_version=2,dataset='WEPDTOF',source=selected['source'],roles=roles,
        source_selection_sha256=sha256((dataset/'selection.json').read_bytes()).hexdigest(),
        partition_sources={role:dict(root=str(root.resolve()),
            selection_sha256=sha256((root/'selection.json').read_bytes()).hexdigest())
            for role,(root,_) in sources.items()},
        counts=dict(counts),items=rows,task='detect',
        protocol='ROI-neutral114; clipped rotated targets converted to enclosing axis-aligned person boxes',
        warning='Pilot only. No pixel-mask, hand, object, ReID or depth supervision. Final test not exported.')
    (output/'export_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    shutil.copyfile(dataset/'NOTICE.txt',output/'NOTICE.txt')
    # A bounded recipe, not a trained checkpoint or claimed research result.
    recipe=dict(task='detect',data=str((output/'data.yaml').resolve()),epochs=20,imgsz=640,batch=4,
        device=0,workers=0,cache=False,amp=True,seed=42,deterministic=True,patience=5,
        degrees=180.,fliplr=.5,flipud=.5,mosaic=.5,close_mosaic=5,
        plots=False,save=True,project='runs/overhead-training',name='person-domain-pilot')
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),counts=dict(counts),trained=False),indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/wepdtof-pilot'))
    parser.add_argument('--validation-dataset',type=Path,
        help='Frozen parent pilot when --dataset is a development-only extension')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.dataset,args.output,args.validation_dataset)


if __name__=='__main__':main()
