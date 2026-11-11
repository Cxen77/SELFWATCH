"""Isolated official ORT CUDA-12 runtime; keeps the CPU environment intact."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    version='1.26.0' # Newer default PyPI wheels use CUDA 13; this laptop's torch is CUDA 12.8.
    root=Path(__file__).resolve().parents[1];base=root/'.runtime';base.mkdir(exist_ok=True)
    target=base/('ort-gpu-'+version)
    if target.exists():raise RuntimeError('Isolated runtime already exists; verify it rather than overwrite it')
    with urllib.request.urlopen(f'https://pypi.org/pypi/onnxruntime-gpu/{version}/json',timeout=30) as response:
        metadata=json.load(response)
    tag=f'cp{sys.version_info.major}{sys.version_info.minor}'
    candidates=[r for r in metadata['urls'] if f'{tag}-{tag}-win_amd64.whl' in r['filename']]
    if len(candidates)!=1:raise RuntimeError('No unique official wheel for this Python/Windows environment')
    record=candidates[0]
    if record['size']>384*1024**2 or not record['url'].startswith('https://files.pythonhosted.org/'):
        raise RuntimeError('Unexpected wheel size or source')
    wheel=base/record['filename'];digest=sha256();count=0
    print(f'Downloading official {record["filename"]}: {record["size"]/1024**2:.1f} MiB',flush=True)
    with urllib.request.urlopen(record['url'],timeout=30) as response,wheel.open('xb') as output:
        while True:
            block=response.read(1024**2)
            if not block:break
            count+=len(block)
            if count>record['size']:raise RuntimeError('Download exceeded declared size')
            digest.update(block);output.write(block)
    if count!=record['size'] or digest.hexdigest()!=record['digests']['sha256']:
        raise RuntimeError('Wheel integrity failure; installation not attempted')
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheel),
                    '--disable-pip-version-check'],check=True)
    manifest=dict(distribution='onnxruntime-gpu',version=version,source=record['url'],
        bytes=count,sha256=digest.hexdigest(),target=str(target),cuda_major=12,
        note='Official PyPI wheel, isolated from the normal CPU onnxruntime installation; execution still requires validation')
    (base/'pose-gpu-runtime.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
