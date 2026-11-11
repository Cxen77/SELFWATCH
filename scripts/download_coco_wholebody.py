"""Bounded official COCO-WholeBody validation diagnostic; never used for training."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
import urllib.request

ANNOTATIONS='https://drive.usercontent.google.com/download?id=1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb&export=download&confirm=t'
SOURCE='https://github.com/jin-s13/COCO-WholeBody'


def fetch(url,target,limit):
    if target.exists():return dict(path=target.name,bytes=target.stat().st_size,sha256=sha256(target.read_bytes()).hexdigest(),reused=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.part')
    if temporary.exists():raise ValueError('Incomplete transfer preserved; use a new output directory or inspect the partial file')
    digest=sha256();size=0
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SELFWATCH-research/0.1'}),timeout=45) as response:
        if 'text/html' in response.headers.get('Content-Type',''):raise ValueError('Download returned a web page rather than dataset bytes')
        with temporary.open('xb') as output:
            while True:
                block=response.read(min(1024**2,limit-size+1))
                if not block:break
                size+=len(block)
                if size>limit:raise ValueError('Bounded download payload exceeded')
                output.write(block);digest.update(block)
    temporary.rename(target)
    return dict(path=target.name,bytes=size,sha256=digest.hexdigest(),reused=False)


def choose(data,count):
    eligible={a['image_id'] for a in data['annotations'] if a['category_id']==1 and not a.get('iscrowd') and any(a['keypoints'][2::3])}
    images=sorted((i for i in data['images'] if i['id'] in eligible),key=lambda i:i['id'])
    if len(images)<count:raise ValueError('Not enough eligible validation images')
    chosen=[images[((2*i+1)*len(images))//(2*count)] for i in range(count)]
    ids={i['id'] for i in chosen}
    return dict(images=chosen,annotations=[a for a in data['annotations'] if a['image_id'] in ids],
        categories=data['categories'],licenses=data.get('licenses',[]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('datasets/coco-wholebody-pilot'))
    parser.add_argument('--images',type=int,default=64);args=parser.parse_args()
    if not 8<=args.images<=128:raise ValueError('Use a bounded diagnostic subset, 8–128 images')
    args.output.mkdir(parents=True,exist_ok=True)
    status=dict(status='downloading_annotations',source=SOURCE,annotation_url=ANNOTATIONS,requested_images=args.images,files=[])
    def persist():
        (args.output/'download_manifest.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    persist()
    try:
        print('Downloading author-linked validation annotations (160 MiB limit; source is 132,970,317 bytes)',flush=True)
        annotation=args.output/'coco_wholebody_val_v1.0.json'
        record=fetch(ANNOTATIONS,annotation,160*1024**2);status['files'].append(record);persist()
        data=json.loads(annotation.read_text(encoding='utf-8'));selected=choose(data,args.images)
        selected.update(schema_version=1,dataset='COCO-WholeBody-v1.0-validation',role='public_validation_diagnostic_only',
            sampling='Uniform midpoint strata over sorted image IDs with a noncrowd person having body labels; frozen before inference',
            annotation_sha256=record['sha256'],source=SOURCE)
        selection=args.output/'selection.json';payload=json.dumps(selected,indent=2)
        if selection.exists() and selection.read_text(encoding='utf-8')!=payload:raise ValueError('Frozen selection differs; preserve this output')
        selection.write_text(payload,encoding='utf-8')
        (args.output/'NOTICE.txt').write_text('COCO-WholeBody v1.0: Sheng Jin et al., ECCV 2020.\nAuthor source: '+SOURCE+'\nAuthors restrict annotations to research/noncommercial use and describe CC BY 4.0. Original COCO image rights remain per-image; licenses and URLs are retained in selection.json.\nThis is public validation for diagnostics, not a new retail test set and never training data.\n',encoding='utf-8')
        status.update(status='downloading_images',selection_sha256=sha256(selection.read_bytes()).hexdigest());persist()
        payload_bytes=0
        for index,item in enumerate(selected['images']):
            name=item['file_name']
            if name!=f'{item["id"]:012d}.jpg':raise ValueError('Unexpected COCO image filename')
            # The image alias currently fails TLS hostname validation. Use its
            # public S3 bucket over verified HTTPS; never disable certificate checks.
            url=f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}'
            target=args.output/'images'/name;record=fetch(url,target,8*1024**2)
            payload_bytes+=record['bytes']
            if payload_bytes>256*1024**2:raise ValueError('Image pool exceeded 256 MiB budget')
            record.update(path='images/'+name,url=url,image_id=item['id']);status['files'].append(record);persist()
            if (index+1)%8==0:print(f'{index+1}/{len(selected["images"])} images saved',flush=True)
        status['status']='complete';status['image_bytes']=payload_bytes;persist()
        print('Whole-body diagnostic download complete',flush=True)
    except Exception as error:
        status.update(status='failed',error=str(error));persist();raise


if __name__=='__main__':main()
