"""Fetch the separate person-training baseline from its official release, SHA-verified."""
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

URL='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'
SHA256='9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'
SIZE=5544453


def main():
    target=Path('weights/v2/yolo26n.pt');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():payload=target.read_bytes()
    else:
        with urllib.request.urlopen(URL,timeout=60) as response:payload=response.read(SIZE+1)
    if len(payload)!=SIZE or sha256(payload).hexdigest()!=SHA256:
        raise ValueError('Baseline differs from the official release asset size/digest')
    if not target.exists():target.write_bytes(payload)
    manifest=dict(path=str(target),source=URL,bytes=SIZE,sha256=SHA256,
        integrity_source='GitHub official assets v8.4.0 release API digest',
        purpose='Separate detection-only domain adaptation; existing segmentation weights unchanged')
    target.with_suffix('.source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
