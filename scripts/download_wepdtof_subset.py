"""Inspect/download a bounded WEPDTOF subset from the official public ZIP.

Uses checked HTTP ranges, never saves/extracts the full 5.4 GB archive. A catalog
is prepared first so image and annotation correspondence can be audited.
"""
import argparse
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path,PurePosixPath
import urllib.request
import zipfile
from selfwatch_v2.http_range import HTTPRangeReader

URL='https://www.bu.edu/vip/files/WEPDTOF.zip'
PAGE='https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/download/'


def safe_target(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or any(':' in p or '\\' in p or p.endswith((' ','.')) for p in relative.parts):
        raise ValueError('Unsafe archive member path')
    reserved={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
    if any(p.split('.')[0].upper() in reserved for p in relative.parts):raise ValueError('Reserved Windows archive path')
    target=root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Archive path escaped output')
    return target


def save_member(archive,member,root):
    if member.file_size>32*1024**2 or member.flag_bits&1 or (member.external_attr>>16)&0o170000==0o120000:
        raise ValueError('Unsupported or oversized archive member')
    target=safe_target(root,member.filename)
    data=archive.read(member)  # zipfile verifies the member CRC.
    if len(data)!=member.file_size:raise ValueError('Incomplete archive member')
    if target.exists() and target.read_bytes()!=data:raise ValueError('Existing data differs; preserving local file')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.write_bytes(data)
    return dict(path=str(target.relative_to(root)),size=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=member.CRC)


def make_selection(root,count):
    if count<1:raise ValueError('Positive frames per sequence required')
    annotations=sorted((root/'WEPDTOF/annotations').glob('*.json'))
    group=lambda name:name[:-2] if name in ('jewelry_store_2','large_office_2') else name
    groups=sorted({group(p.stem) for p in annotations},key=lambda s:hashlib.sha256(('SELFWATCH-overhead-scenes-v1/'+s).encode()).hexdigest())
    development=set(groups[:len(groups)//2])
    records=[]
    for path in annotations:
        data=json.loads(path.read_text(encoding='utf-8'))
        images=sorted(data['images'],key=lambda f:f['file_name'])
        by_image={}
        for annotation in data['annotations']:by_image.setdefault(annotation['image_id'],[]).append(annotation)
        size=min(count,len(images))
        positions=[((2*i+1)*len(images))//(2*size) for i in range(size)]
        for position in positions:
            image=images[position]
            if PurePosixPath(image['file_name']).name!=image['file_name']:raise ValueError('Unexpected annotated image path')
            records.append(dict(sequence=path.stem,source_group=group(path.stem),
                split='development' if group(path.stem) in development else 'held_out',
                image_id=image['id'],image_info=image,
                image_path=f'WEPDTOF/frames/{path.stem}/{image["file_name"]}',
                roi_path=f'WEPDTOF/ROI_maps/{path.stem}.png',
                annotation_path=str(path.relative_to(root)),annotations=by_image.get(image['id'],[])))
    return dict(schema_version=1,dataset='WEPDTOF',source=URL,status='partial',
        sampling='Uniform midpoint strata over sorted source image filenames, before model inference',
        split_policy='SHA256 scene-group ordering; first half development, remaining held out; paired video segments stay together',
        group_order=groups,items=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('datasets/wepdtof-pilot'))
    parser.add_argument('--max-mib',type=int,default=384)
    parser.add_argument('--catalog-only',action='store_true')
    parser.add_argument('--metadata',action='store_true',help='Download annotations and ROI maps only')
    parser.add_argument('--frames-per-sequence',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():
        previous=json.loads((args.output/'catalog.json').read_text(encoding='utf-8'))
        if previous['source']!=URL:raise ValueError('Output belongs to another source')
    args.output.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(PAGE,timeout=30) as response:
        (args.output/'SOURCE.html').write_bytes(response.read(1024**2))
    (args.output/'NOTICE.txt').write_text(
        'WEPDTOF — Boston University VIP Laboratory.\nSource: '+PAGE+'\n'
        'The source makes frames/annotations available for non-commercial research use only under Creative Commons Attribution 4.0.\n'
        'Cite Tezcan et al., WEPDTOF, WACV 2022, pp. 503–512. Preserve the source notice; no broader use rights are asserted.\n',encoding='utf-8')
    with HTTPRangeReader(URL,max_bytes=args.max_mib*1024**2) as stream,zipfile.ZipFile(stream) as archive:
        records=[dict(name=m.filename,size=m.file_size,compressed_size=m.compress_size,crc32=m.CRC,
                      offset=m.header_offset) for m in archive.infolist() if not m.is_dir()]
        manifest=dict(source=URL,source_page=PAGE,archive_bytes=stream.size,etag=stream.etag,
            modified=stream.modified,downloaded_at=datetime.now(timezone.utc).isoformat(),
            status='catalogued',archive_payload_bytes_read=stream.bytes_read,http_requests=stream.requests,
            selection='Catalog only; no model-based frame selection',members=records)
        files=[]
        if args.metadata or args.frames_per_sequence:
            for member in archive.infolist():
                if ((member.filename.startswith('WEPDTOF/annotations/') and member.filename.endswith('.json')) or
                    (member.filename.startswith('WEPDTOF/ROI_maps/') and member.filename.endswith('.png'))):
                    files.append(save_member(archive,member,args.output))
            (args.output/'metadata_manifest.json').write_text(json.dumps(dict(source=URL,etag=stream.etag,
                status='metadata_complete',files=files,archive_payload_bytes_read=stream.bytes_read),indent=2),encoding='utf-8')
        if args.frames_per_sequence:
            selection=make_selection(args.output,args.frames_per_sequence)
            (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
            for index,item in enumerate(selection['items']):
                record=save_member(archive,archive.getinfo(item['image_path']),args.output)
                item['image_sha256']=record['sha256'];files.append(record)
                if (index+1)%args.frames_per_sequence==0:
                    print(f'Fetched {index+1}/{len(selection["items"])} images; archive payload {stream.bytes_read/1024**2:.1f} MiB',flush=True)
            selection['status']='complete'
            (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
            (args.output/'download_manifest.json').write_text(json.dumps(dict(source=URL,etag=stream.etag,
                status='complete',files=files,archive_payload_bytes_read=stream.bytes_read,
                archive_bytes=stream.size,http_requests=stream.requests),indent=2),encoding='utf-8')
        manifest.update(archive_payload_bytes_read=stream.bytes_read,http_requests=stream.requests)
        (args.output/'catalog.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        annotations=[m for m in records if m['name'].lower().endswith(('.json','.txt'))]
        print(json.dumps(dict(archive_bytes=stream.size,payload_bytes=stream.bytes_read,
            members=len(records),annotations=annotations[:30],examples=records[:8]),indent=2),flush=True)


if __name__=='__main__':main()
