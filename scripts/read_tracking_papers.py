"""Retain primary research PDFs and searchable text without executing repository code."""
import hashlib
import json
from pathlib import Path
import urllib.request
from pypdf import PdfReader

PAPERS = [
    ('CVPR2025-TrackTrack', 'Focusing on Tracks for Online Multi-Object Tracking', 2025,
     'https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf'),
    ('2603.06034', 'Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking', 2026, 'https://arxiv.org/pdf/2603.06034v2'),
    ('2610.22306', 'ARC-SORT: Adaptive, Robust, Compensated Multi-Object Tracking with Interacting Multiple Models', 2026, 'https://arxiv.org/pdf/2610.22306v1'),
    ('2609.18582', 'On-the-Fly Homographies Calibration for Multi-Camera Tracking', 2026, 'https://arxiv.org/pdf/2609.18582v1'),
    ('2408.13003', 'BoostTrack++: using tracklet information to detect more objects in multiple object tracking', 2025, 'https://arxiv.org/pdf/2408.13003v2'),
    ('2302.11813', 'Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification', 2023, 'https://arxiv.org/pdf/2302.11813'),
]

def main():
    root=Path('C:/Users/Dev/Desktop/SELFWATCH_RESEARCH')
    manifest_path=root/'paper_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    text_dir=root/'tracking-20261108';text_dir.mkdir(exist_ok=True)
    for identifier,title,year,url in PAPERS:
        path=root/'papers'/f'{identifier}_tracking.pdf'
        if not path.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'SELFWATCH-research/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:
                data=response.read(40*1024**2+1)
            if not data.startswith(b'%PDF') or len(data)>40*1024**2:
                raise ValueError('Unexpected PDF response')
            path.write_bytes(data)
        reader=PdfReader(path)
        text_path=text_dir/f'{identifier}.txt'
        text_path.write_text('\n'.join(f'\nPAGE {i+1}\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages)),encoding='utf-8',errors='replace')
        record=dict(id=identifier,title=title,year=year,source=url,path=str(path.resolve()),
            bytes=path.stat().st_size,pages=len(reader.pages),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            status='verified',downloaded_date='2026-11-08',extracted_text=str(text_path.resolve()))
        existing=next((p for p in manifest['papers'] if p['id']==identifier),None)
        if existing is None:manifest['papers'].append(record)
        elif existing['sha256']!=record['sha256']:raise ValueError('Existing source differs; preserve original revision')
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
