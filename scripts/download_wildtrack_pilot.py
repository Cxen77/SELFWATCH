"""Fetch bounded, synchronized WILDTRACK frames and original labels."""
from hashlib import sha256
import json
from pathlib import Path
import struct
import zipfile
import zlib

from selfwatch_v2.http_range import HTTPRangeReader

URL='https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip'
SOURCE='https://www.epfl.ch/labs/cvlab/data/data-wildtrack/'
CAMERAS=(1,3,5)
FRAMES=tuple(range(1700,1790,5))
ROOT=Path('datasets/wildtrack-pilot-v1')
PREFIX='Wildtrack_dataset/'
ARCHIVE=(6807496358,'"5b86d551-195c226a6"')
CAP=8*1024**2
CALIBRATION_CAMERAS={1:'CVLab1',3:'CVLab3',5:'IDIAP1'}
CALIBRATION_FILES=('README.txt',*(path for name in CALIBRATION_CAMERAS.values() for path in
    (f'calibrations/extrinsic/extr_{name}.xml',f'calibrations/intrinsic_zero/intr_{name}.xml')))


def member(reader,info,offsets=None):
    # The pinned archive has a 4 GiB ZIP offset ambiguity in Python's reader.
    # Match the exact local filename, bound decompression, and check central CRC.
    if not 0<info.file_size<=CAP or not 0<info.compress_size<=CAP:
        raise ValueError('Member exceeds the pilot bound')
    candidates = offsets if offsets is not None else (info.header_offset-2**32,info.header_offset)
    for offset in candidates:
        if not 0<=offset<=reader.size-30:
            continue
        reader.seek(offset)
        header=reader.read(30)
        if header[:4]!=b'PK\x03\x04':
            continue
        flags,method=struct.unpack_from('<HH',header,6)
        name_len,extra_len=struct.unpack_from('<HH',header,26)
        if name_len!=len(info.filename) or reader.read(name_len)!=info.filename.encode():
            continue
        start=offset+30+name_len+extra_len
        if flags&1 or method!=info.compress_type or start+info.compress_size>reader.size:
            raise ValueError('Invalid local member metadata')
        reader.seek(start)
        payload=reader.read(info.compress_size)
        if method==zipfile.ZIP_STORED:
            data=payload
        elif method==zipfile.ZIP_DEFLATED:
            decoder=zlib.decompressobj(-15)
            data=decoder.decompress(payload,info.file_size+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('Deflate stream exceeded the central member bound')
        else:
            raise ValueError('Unexpected compression method')
        if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
            raise ValueError('Member length or CRC mismatch')
        return data
    raise ValueError('No matching local header for '+info.filename)


def download(root,frames,calibration=False):
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json'
    if manifest.exists():
        raise FileExistsError('Frozen pilot already exists')
    records=[]
    with HTTPRangeReader(URL,max_bytes=224*1024**2,block_size=4*1024**2) as reader, zipfile.ZipFile(reader) as archive:
        if (reader.size,reader.etag)!=ARCHIVE:
            raise ValueError('WILDTRACK archive differs from the inspected revision')
        paths=list(CALIBRATION_FILES) if calibration else []
        paths.extend(f'annotations_positions/{f:08d}.json' for f in frames)
        paths.extend(f'Image_subsets/C{c}/{f:08d}.png' for c in CAMERAS for f in frames)
        for index,path in enumerate(paths):
            info=archive.getinfo(PREFIX+path)
            target=root/path
            target.parent.mkdir(parents=True,exist_ok=True)
            data=target.read_bytes() if target.exists() else member(reader,info)
            if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
                raise ValueError('Local member differs from the pinned source')
            if not target.exists():
                target.write_bytes(data)
            records.append(dict(path=path,bytes=len(data),sha256=sha256(data).hexdigest(),archive_crc32=info.CRC))
            if calibration and index+1==len(CALIBRATION_FILES):
                (root/'calibration-manifest.json').write_text(json.dumps(dict(etag=reader.etag,files=records),indent=2),encoding='utf-8')
                print('Calibration and source README saved',flush=True)
            if (index+1)%6==0:
                print(f'{index+1}/{len(paths)} files; {reader.bytes_read/1024**2:.1f} MiB transferred',flush=True)
        data=dict(schema_version=1,dataset='WILDTRACK',source=SOURCE,url=URL,etag=reader.etag,
            archive_bytes=reader.size,transfer_bytes=reader.bytes_read,cameras=CAMERAS,frames=frames,
            capture_time='Annotated frame index / 10; annotations every five frames (2 Hz).',
            scope=('Separate temporal evaluation segment from the same outdoor recording.' if calibration else
                   'Single outdoor development pilot, fixed C1/C3/C5 and consecutive labelled timestamps selected before model inference.'),files=records)
    manifest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('WILDTRACK pilot complete',flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-evaluation',action='store_true')
    args=parser.parse_args()
    if args.geometry_evaluation:
        download(Path('datasets/wildtrack-geometry-evaluation-v1'),tuple(range(1300,1360,5)),calibration=True)
    else:
        download(ROOT,FRAMES)


if __name__=='__main__':
    main()
