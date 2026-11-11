"""Archive changed source/docs with checksums and selected experiment evidence."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--date',required=True)
    parser.add_argument('--max-source-mib',type=int,default=2)
    parser.add_argument('--evidence',type=Path,nargs='*',default=[])
    args=parser.parse_args();root=Path.cwd().resolve()
    if not 1<=args.max_source_mib<=32:raise ValueError('Source limit must be 1-32 MiB')
    args.output.mkdir(parents=True,exist_ok=False)
    paths=subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','-z']).decode().split('\0')
    allowed={'.py','.ps1','.md','.js','.css','.html','.json','.txt'}
    files=[]
    for name in sorted(set(paths)):
        path=(root/name).resolve()
        if not name or not path.is_file() or not path.is_relative_to(root):continue
        if path.suffix not in allowed and path.name!='.gitignore':continue
        if path.stat().st_size>args.max_source_mib*1024**2:raise ValueError(f'Unexpected large source file: {name}')
        files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
            sha256=sha256(path.read_bytes()).hexdigest()))
    archive=args.output/'source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        for item in files:output.write(root/item['path'],item['path'])
    with zipfile.ZipFile(archive) as saved:
        if saved.testzip() is not None:raise ValueError('Source archive CRC mismatch')
        for item in files:
            if sha256(saved.read(item['path'])).hexdigest()!=item['sha256']:
                raise ValueError('Source archive checksum mismatch')
            if sha256((root/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Source changed during checkpoint')
    model_files=[]
    for path in sorted((root/'weights').rglob('*')):
        if path.is_file() and path.suffix in ('.pt','.pth','.onnx'):
            model_files.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,
                sha256=sha256(path.read_bytes()).hexdigest()))
    evidence=[]
    if args.evidence:(args.output/'evidence').mkdir()
    for index,path in enumerate(args.evidence):
        target=args.output/'evidence'/f'{index:02d}-{path.name}'
        shutil.copyfile(path,target)
        evidence.append(dict(source=str(path.resolve()),path=str(target.relative_to(args.output)),
            bytes=target.stat().st_size,sha256=sha256(target.read_bytes()).hexdigest()))
    manifest=dict(date=args.date,base_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        branch=subprocess.check_output(['git','branch','--show-current']).decode().strip(),
        scope='Changed source/docs relative to preserved baseline; excludes datasets, binary weights and private camera images',
        max_source_mib=args.max_source_mib,
        files=files,models=model_files,evidence=evidence,
        archive=dict(path='source.zip',bytes=archive.stat().st_size,sha256=sha256(archive.read_bytes()).hexdigest()))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),source_files=len(files),archive=manifest['archive']),indent=2))


if __name__=='__main__':main()
