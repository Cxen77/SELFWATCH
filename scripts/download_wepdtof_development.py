"""Expand only frozen development scenes; validation/final-test images stay untouched."""
import argparse
from hashlib import sha256
import json
from pathlib import Path,PurePosixPath
import shutil
import zipfile
import zlib
from selfwatch_v2.http_range import HTTPRangeReader
from selfwatch_v2.overhead_data import load_selection
from .download_wepdtof_subset import URL,safe_target,save_member


def make_extension(pilot,count):
    if count<8:raise ValueError('Keep at least the original eight source samples')
    selected,roles=load_selection(pilot)
    records=[]
    sequences=sorted({i['sequence'] for i in selected['items'] if roles[i['source_group']]=='development'})
    for sequence in sequences:
        original=[i for i in selected['items'] if i['sequence']==sequence]
        source=json.loads((pilot/original[0]['annotation_path']).read_text(encoding='utf-8'))
        images=sorted(source['images'],key=lambda i:i['file_name']);size=min(count,len(images))
        wanted={images[((2*i+1)*len(images))//(2*size)]['id'] for i in range(size)}
        wanted.update(i['image_id'] for i in original)
        by_image={}
        for annotation in source['annotations']:by_image.setdefault(annotation['image_id'],[]).append(annotation)
        for image in images:
            if image['id'] not in wanted:continue
            if PurePosixPath(image['file_name']).name!=image['file_name']:raise ValueError('Invalid source image name')
            records.append(dict(sequence=sequence,source_group=original[0]['source_group'],split='development',
                image_id=image['id'],image_info=image,image_path=f'WEPDTOF/frames/{sequence}/{image["file_name"]}',
                roi_path=original[0]['roi_path'],annotation_path=original[0]['annotation_path'],
                annotations=by_image.get(image['id'],[])))
    return dict(schema_version=1,dataset='WEPDTOF-development-extension',source=URL,status='partial',
        sampling=f'Uniform midpoint strata, {count} per development sequence, union original development samples; no model-based selection',
        parent_selection_sha256=sha256((pilot/'selection.json').read_bytes()).hexdigest(),
        group_order=selected['group_order'],split_policy='Only the seven frozen development groups; no validation or final-test images',items=records)


def local_record(root,name,member,expected_sha=None):
    path=safe_target(root,name)
    if not path.exists():return None
    payload=path.read_bytes();digest=sha256(payload).hexdigest()
    if (len(payload)!=member.file_size or zlib.crc32(payload)!=member.CRC
        or expected_sha and digest!=expected_sha):raise ValueError('Existing extension file is corrupt; preserving it for inspection')
    return dict(path=str(path.relative_to(root)),size=len(payload),sha256=digest,crc32=member.CRC)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=Path('datasets/wepdtof-pilot'))
    parser.add_argument('--output',type=Path,default=Path('datasets/wepdtof-development-expanded'))
    parser.add_argument('--frames-per-sequence',type=int,default=32)
    parser.add_argument('--max-mib',type=int,default=512);args=parser.parse_args()
    if not 8<=args.frames_per_sequence<=128 or not 8<=args.max_mib<=768:raise ValueError('Invalid bounded expansion limits')
    selection=make_extension(args.pilot,args.frames_per_sequence)
    catalog=json.loads((args.pilot/'catalog.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=True)
    old={}
    if (args.output/'selection.json').exists():
        previous=json.loads((args.output/'selection.json').read_text(encoding='utf-8'))
        if (previous['parent_selection_sha256']!=selection['parent_selection_sha256']
            or [i['image_path'] for i in previous['items']]!=[i['image_path'] for i in selection['items']]):
            raise ValueError('Output belongs to a different locked selection')
    if (args.output/'download_manifest.json').exists():
        previous=json.loads((args.output/'download_manifest.json').read_text(encoding='utf-8'))
        old={record['path'].replace('\\','/'):record['sha256'] for record in previous['files']}
    members={record['name']:record for record in catalog['members']}
    expected=sum(members[i['image_path']]['compressed_size'] for i in selection['items'])
    print(f'Locked {len(selection["items"])} development-only images; compressed source members {expected/1024**2:.1f} MiB',flush=True)
    records=[];shutil.copyfile(args.pilot/'NOTICE.txt',args.output/'NOTICE.txt')
    with HTTPRangeReader(URL,max_bytes=args.max_mib*1024**2) as stream,zipfile.ZipFile(stream) as archive:
        if stream.etag!=catalog['etag'] or stream.size!=catalog['archive_bytes']:
            raise ValueError('Official archive differs from the frozen pilot')
        metadata=sorted({str(i[key]).replace('\\','/') for i in selection['items'] for key in ('roi_path','annotation_path')})
        for name in metadata:
            source=args.pilot/name;target=safe_target(args.output,name)
            if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Existing metadata differs')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            records.append(local_record(args.output,name,archive.getinfo(name)))
        def persist(status):
            selection['status']=status
            (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
            (args.output/'download_manifest.json').write_text(json.dumps(dict(status=status,source=URL,
                etag=stream.etag,archive_bytes=stream.size,archive_payload_bytes_read=stream.bytes_read,
                parent_selection_sha256=selection['parent_selection_sha256'],files=records),indent=2),encoding='utf-8')
        try:
            for index,item in enumerate(sorted(selection['items'],key=lambda i:members[i['image_path']]['offset'])):
                name=item['image_path'];member=archive.getinfo(name);target=safe_target(args.output,name)
                # Reuse already-downloaded pilot members without network payload.
                source=safe_target(args.pilot,name)
                if not target.exists() and source.exists():
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                record=local_record(args.output,name,member,old.get(name)) or save_member(archive,member,args.output)
                records.append(record);item['image_sha256']=record['sha256']
                if (index+1)%16==0:
                    persist('partial');print(f'{index+1}/{len(selection["items"])} verified; archive payload {stream.bytes_read/1024**2:.1f} MiB',flush=True)
            persist('complete');print('Development extension complete',flush=True)
        except Exception:
            persist('partial');raise


if __name__=='__main__':main()
