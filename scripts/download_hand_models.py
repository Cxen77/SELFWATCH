"""Download the official RTMLib hand models as bounded research candidates."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from .download_coco_wholebody import fetch

PREFIX='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
MODELS={
    'pose':('rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',(256,256)),
    'detector':('rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',(320,320))}


def main():
    root=Path('weights/v2/hand');root.mkdir(parents=True,exist_ok=True);records={}
    for name,(filename,size) in MODELS.items():
        print(f'Downloading official hand {name}',flush=True)
        archive=root/(name+'.zip');record=fetch(PREFIX+filename,archive,128*1024**2)
        with zipfile.ZipFile(archive) as source:
            members=[m for m in source.infolist() if m.filename.endswith('.onnx')]
            if len(members)!=1 or members[0].file_size>256*1024**2:raise ValueError('Unexpected bounded model archive')
            payload=source.read(members[0]) # CRC verified; extract no executable files or archive paths.
        target=root/(name+'.onnx')
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Existing candidate differs; preserve it for inspection')
        if not target.exists():target.write_bytes(payload)
        records[name]=dict(source_url=PREFIX+filename,archive=record,model_path=str(target),
            model_bytes=len(payload),model_sha256=sha256(payload).hexdigest(),input_wh=size,
            model_list='https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/hand.py',
            status='Research candidate, not enabled in the portal')
        (root/'provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(f'Saved {name}: {len(payload):,} bytes',flush=True)


if __name__=='__main__':main()
