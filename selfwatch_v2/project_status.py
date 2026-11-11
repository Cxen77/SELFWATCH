"""Derive the site checklist from the one human-maintained research log."""
from pathlib import Path
import re


def read_status(path):
    text=Path(path).read_text(encoding='utf-8')
    def field(pattern):
        match=re.search(pattern,text,re.M)
        if not match:raise ValueError('Master project log is missing a status field')
        return match[1]
    current=int(field(r'^\*\*Current stage: (\d+)\*\*$'))
    updated=field(r'^\*\*Updated: (.+)\*\*$')
    summary=field(r'^\*\*Status summary:\*\* (.+)$')
    stages=[];gates=[]
    for line in text.splitlines():
        if not line.startswith('|'):continue
        cells=[cell.strip() for cell in line.strip('|').split('|')]
        if len(cells)!=4:continue
        first,name,status,evidence=cells
        if first.isdigit():stages.append(dict(number=int(first),name=name,status=status,evidence=evidence))
        elif first in 'ABCDEF' and len(first)==1:
            gates.append(dict(name=first+' / '+name,status=status,condition=evidence))
    if [s['number'] for s in stages]!=list(range(12)) or len(gates)!=6 or not 0<=current<12:
        raise ValueError('Master log needs stages 0–11 and six decision gates')
    return dict(updated=updated,current_stage=current,summary=summary,stages=stages,gates=gates,
        master_log='/project-log.md')
