"""Retain two complete, frozen MOT17 sequences from the official JPEG ZIP."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import zipfile
import zlib
from selfwatch_v2.http_range import HTTPRangeReader
from .download_wildtrack_pilot import member

ROOT=Path('datasets/mot17-continuity-native-v1')
URL='https://motchallenge.net/data/MOT17.zip'
ARCHIVE=(5860214001, '"15d4bc4f1-5b6c01991f807"')
SEQUENCES={'MOT17-02-FRCNN':dict(role='development',frames=600,fps=30),
           'MOT17-09-FRCNN':dict(role='evaluation',frames=525,fps=30)}


def load_catalog(path):
    if path.exists():
        catalog = json.loads(path.read_text(encoding='utf-8'))
    else:
        with HTTPRangeReader(URL, max_bytes=32*1024**2, block_size=4*1024**2) as stream:
            if (stream.size, stream.etag) != ARCHIVE:
                raise ValueError('Official archive revision changed')
            with zipfile.ZipFile(stream) as archive:
                members = [dict(name=info.filename, size=info.file_size,
                                compressed_size=info.compress_size, offset=info.header_offset, crc32=info.CRC)
                           for info in archive.infolist()
                           if any(info.filename.startswith(f'MOT17/train/{seq}/') for seq in SEQUENCES)]
            catalog = dict(source=URL, archive_bytes=stream.size, etag=stream.etag,
                           modified=stream.modified, members=members)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    if catalog['source'] != URL or (catalog['archive_bytes'], catalog['etag']) != ARCHIVE:
        raise ValueError('Unexpected MOT17 archive catalog')
    return catalog


def download_sequence(seq,setting,catalog):
    prefix=f'MOT17/train/{seq}/'
    rows=[r for r in catalog['members'] if r['name'].startswith(prefix) and
          (re.fullmatch(r'img1/\d{6}\.jpg',r['name'][len(prefix):]) or r['name'][len(prefix):] in ('gt/gt.txt','seqinfo.ini'))]
    expected={f'img1/{n:06d}.jpg' for n in range(1,setting['frames']+1)}|{'gt/gt.txt','seqinfo.ini'}
    if {r['name'][len(prefix):] for r in rows}!=expected:raise ValueError('Incomplete original JPEG sequence catalog')
    records=[]
    with HTTPRangeReader(catalog['source'],max_bytes=256*1024**2,block_size=8*1024**2) as stream:
        if (stream.size,stream.etag)!=(catalog['archive_bytes'],catalog['etag']):raise ValueError('Source archive revision changed')
        for index,row in enumerate(sorted(rows,key=lambda r:r['offset'])):
            relative=seq+'/'+row['name'][len(prefix):];path=ROOT/relative
            if path.exists():
                payload=path.read_bytes()
                if len(payload)!=row['size'] or zlib.crc32(payload)!=row['crc32']:raise ValueError('Existing member differs')
            else:
                info=zipfile.ZipInfo(row['name']);info.file_size=row['size'];info.compress_size=row['compressed_size']
                info.header_offset=row['offset'];info.CRC=row['crc32']
                found=False
                for offset in (row['offset'],row['offset']-2**32):
                    if not 0<=offset<=stream.size-30:continue
                    stream.seek(offset);header=stream.read(30)
                    if header[:4]!=b'PK\x03\x04':continue
                    name_len=struct.unpack_from('<H',header,26)[0]
                    if name_len!=len(row['name']) or stream.read(name_len)!=row['name'].encode():continue
                    info.compress_type=struct.unpack_from('<H',header,8)[0];found=True;break
                if not found:raise ValueError('No exact local member header')
                payload=member(stream,info,offsets=(offset,))
                path.parent.mkdir(parents=True,exist_ok=True)
                partial=path.with_suffix(path.suffix+'.partial')
                partial.write_bytes(payload)
                partial.replace(path)
            records.append(dict(path=relative,bytes=len(payload),sha256=sha256(payload).hexdigest(),crc32=row['crc32']))
            if (index+1)%100==0:print(seq,index+1,'/',len(rows),'transferred',round(stream.bytes_read/1024**2,1),'MiB',flush=True)
        return records,dict(sequence=seq,bytes=stream.bytes_read,requests=stream.requests)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():raise ValueError('Completed sequence dataset already frozen')
    catalog_path=Path('runs/mot17-archive-catalog.json');catalog=load_catalog(catalog_path)
    selection=dict(schema_version=1,sequences=SEQUENCES,source=catalog['source'],
        source_page='https://motchallenge.net/data/MOT17/',archive_bytes=catalog['archive_bytes'],etag=catalog['etag'],
        catalog_sha256=sha256(catalog_path.read_bytes()).hexdigest(),
        protocol='Complete native JPEG sequences, independent official GT; select on 02 before scoring 09. No video preview frames or public detector boxes used.')
    frozen=ROOT/'selection.json'
    if frozen.exists() and json.loads(frozen.read_text())!=selection:raise ValueError('Selection changed')
    frozen.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(download_sequence,seq,setting,catalog) for seq,setting in SEQUENCES.items()]
        results=[job.result() for job in jobs]
    (ROOT/'manifest.json').write_text(json.dumps(dict(selection=selection,
        files=[row for records,_ in results for row in records],transfers=[transfer for _,transfer in results],status='complete'),indent=2),encoding='utf-8')
    print('Native MOT17 sequences complete',flush=True)

if __name__=='__main__':main()
