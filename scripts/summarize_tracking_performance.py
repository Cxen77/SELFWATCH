"""Regenerate the measured quality-preserving execution / optional-work table."""
import hashlib
import json
from pathlib import Path
import statistics


def main():
    output=Path('runs/tracking-performance-20261108');output.mkdir(exist_ok=True)
    eager_path=Path('runs/crowded-tracking-eager-control-20261108.json')
    graph_path=Path('runs/crowded-tracking-graph-20261108.json')
    eager=json.loads(eager_path.read_text());graph=json.loads(graph_path.read_text())
    signatures={t['identity_signature_sha256'] for t in eager['trials']+graph['trials']}
    if len(signatures)!=1:raise ValueError('Tracking outputs disagree; inspect before making a preservation claim')
    groups=[('Original detailed',eager,'detailed'),('Graph detailed',graph,'detailed'),
            ('Graph balanced',graph,'balanced'),('Original tracking',eager,'fast'),('Graph tracking',graph,'fast')]
    rows=[]
    for name,data,mode in groups:
        values=[t['observations_per_s'] for t in data['trials'] if t['mode']==mode]
        rows.append(dict(name=name,trials=values,mean_fps=statistics.mean(values),
            optional_landmarks=mode!='fast',mask_outlines=mode=='detailed'))
    report=dict(rows=rows,same_identity_signature=next(iter(signatures)),
        scope='One public camera, 12 predecoded 1280x720 observations, 8-17 people, original 2Hz timestamps; includes SQLite/JPEG/JSON; excludes browser/capture and review export. Two graph trials, one isolated eager control per mode. Not deployment capacity or accuracy validation.',
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (eager_path,graph_path)])
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.4,4.8),layout='constrained')
    for index,row in enumerate(rows):
        ax.barh(index,row['mean_fps'],color=['#64748b','#2563eb','#0891b2','#64748b','#15803d'][index],height=.62)
        ax.scatter(row['trials'],[index]*len(row['trials']),color='#111827',s=20,zorder=3)
        ax.text(max(row['trials'])+.16,index,f"{row['mean_fps']:.2f}",va='center',fontsize=10)
    ax.set_yticks(range(len(rows)),[r['name'] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,12)
    ax.set_xlabel('Processed observations per second (offline)')
    ax.set_title('SELFWATCH: crowded person-tracking throughput',loc='left',fontweight='bold')
    ax.spines[['top','right']].set_visible(False);ax.xaxis.grid(alpha=.18);ax.set_axisbelow(True)
    fig.supxlabel('WILDTRACK C1: 12 frames, 8-17 detections. Dots show trials; all modes retain ReID and original memory.\nBalanced reduces optional poses; Tracking omits poses/mask outlines. Capture/browser costs excluded.',fontsize=9)
    fig.savefig(output/'throughput.png',dpi=180);fig.savefig(output/'throughput.svg');plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
