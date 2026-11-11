"""Freeze new image-disjoint development and evaluation subsets before inference."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

from .download_coco_wholebody import fetch
from selfwatch_v2.wholebody_benchmark import load_dataset, file_hash


def main():
    original=Path('datasets/coco-wholebody-pilot-v2')
    old,manifest=load_dataset(original)
    annotation=original/'coco_wholebody_val_v1.0.json'
    digest=file_hash(annotation)
    if digest!=old['annotation_sha256']:
        raise ValueError('Source annotations changed')
    source=json.loads(annotation.read_text(encoding='utf-8'))
    excluded={item['id'] for item in old['images']}
    annotations=defaultdict(list)
    for item in source['annotations']:
        annotations[item['image_id']].append(item)
    eligible=sorted((item for item in source['images'] if item['id'] not in excluded
        and any(a['category_id']==1 and not a.get('iscrowd')
            and any(a[s+'_valid'] and min(a[s+'_box'][2:])>0 and any(a[s+'_kpts'][2::3])
                    for s in ('lefthand','righthand')) for a in annotations[item['id']])),
        key=lambda item: sha256(f"selfwatch-hand-cal-v1:{item['id']}".encode()).hexdigest())
    if len(eligible)<96:
        raise ValueError('Insufficient unused hand-labelled images')
    seen_hashes={row['sha256'] for row in manifest['files'] if row['path'].startswith('images/')}
    selections={}
    for partition,items in (('development',eligible[:48]),('evaluation',eligible[48:96])):
        root=Path('datasets')/f'hand-calibration-{partition}-v1'
        root.mkdir(parents=True,exist_ok=True)
        selection=dict(schema_version=1,dataset=old['dataset'],role=old['role'],partition=partition,
            annotation_sha256=digest,excluded_selection_sha256=manifest['selection_sha256'],
            source=old['source'],images=items,annotations=[a for i in items for a in annotations[i['id']]],
            categories=source['categories'],licenses=source.get('licenses',[]),
            sampling='First 96 unused images with a valid labelled hand, sorted by salted SHA-256 of image ID; first 48 development, next 48 evaluation.')
        payload=json.dumps(selection,indent=2).encode()
        path=root/'selection.json'
        if path.exists() and path.read_bytes()!=payload:
            raise ValueError('Existing split differs from frozen selection')
        path.write_bytes(payload)
        shutil.copyfile(original/'NOTICE.txt',root/'NOTICE.txt')
        selections[partition]=(root,selection,sha256(payload).hexdigest())
    for partition,(root,selection,digest) in selections.items():
        files=[]
        for index,item in enumerate(selection['images']):
            name=f"{item['id']:012d}.jpg"
            if name!=item['file_name']:
                raise ValueError('Unexpected COCO filename')
            url=f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}'
            record=fetch(url,root/'images'/name,8*1024**2)
            if record['sha256'] in seen_hashes:
                raise ValueError('Exact image duplicate across selected data')
            seen_hashes.add(record['sha256'])
            files.append(dict(record,path='images/'+name,url=url,image_id=item['id']))
            if (index+1)%8==0:
                print(f'{partition}: {index+1}/48 images',flush=True)
        (root/'download_manifest.json').write_text(json.dumps(dict(status='complete',partition=partition,
            selection_sha256=digest,files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
