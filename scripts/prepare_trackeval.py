"""Isolate official TrackEval with explicit NumPy scalar-name compatibility only."""
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    root=Path('.runtime/TrackEval');target=Path('.runtime/TrackEval-numpy2')
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip()
    target.mkdir(exist_ok=True);records=[]
    for source in sorted((root/'trackeval').rglob('*.py')):
        relative=source.relative_to(root);path=target/relative;path.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_text(encoding='utf-8')
        code=re.sub(r'\bnp\.int\b','np.int64',raw)
        code=re.sub(r'\bnp\.float\b','np.float64',code)
        code=re.sub(r'\bnp\.bool\b','np.bool_',code)
        path.write_text(code,encoding='utf-8')
        records.append(dict(path=relative.as_posix(),source_sha256=sha256(source.read_bytes()).hexdigest(),
            sha256=sha256(path.read_bytes()).hexdigest(),dtype_names_changed=raw!=code))
    shutil.copyfile(root/'LICENSE',target/'LICENSE')
    (target/'provenance.json').write_text(json.dumps(dict(repository='https://github.com/JonathonLuiten/TrackEval',
        revision=revision,changes='Only replace removed NumPy np.int/np.float/np.bool scalar aliases with int64/float64/bool_. No metric equations or preprocessing changes.',files=records),indent=2),encoding='utf-8')
    print('Prepared',len(records),'Python files; NumPy alias substitutions in',sum(r['dtype_names_changed'] for r in records))

if __name__=='__main__':main()
