"""Audit standard ID-switch locations and visibility-conditioned recovery.

The recovery diagnostic is additional to official TrackEval metrics. It never
changes predictions and always includes missed pre/post observations in counts.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .mot_metrics import load_trackeval


def clear_trace(data):
    """Mirror pinned CLEAR matching solely to localize errors; verify its count."""
    previous=np.full(data['num_gt_ids'],np.nan);last_step=previous.copy()
    matches=[];switches=[];eps=np.finfo(float).eps
    for step,(gt,pred) in enumerate(zip(data['gt_ids'],data['tracker_ids'])):
        current={}
        # Preserve the pinned evaluator's exact early-continue behavior.
        if len(gt)==0 or len(pred)==0:matches.append(current);continue
        sim=data['similarity_scores'][step]
        score=1000*(pred[None,:]==last_step[gt[:,None]])+sim
        score[sim<.5-eps]=0
        rows,cols=linear_sum_assignment(-score);valid=score[rows,cols]>eps
        rows,cols=rows[valid],cols[valid]
        g,p=gt[rows],pred[cols]
        for subject,person in zip(g,p):
            current[int(subject)]=int(person)
            if np.isfinite(previous[subject]) and previous[subject]!=person:
                switches.append(dict(frame=step+1,subject=int(subject),old_id=int(previous[subject]),new_id=int(person)))
        previous[g]=p;last_step[:]=np.nan;last_step[g]=p;matches.append(current)
    return matches,switches


def episodes(annotations,length):
    kept=annotations[(annotations[:,6]!=0)&(annotations[:,7]==1)]
    subjects=sorted({int(subject) for subject in kept[:,1]})
    visibility={i:np.full(length,np.nan) for i in range(len(subjects))}
    mapping={original:dense for dense,original in enumerate(subjects)}
    for row in kept:visibility[mapping[int(row[1])]][int(row[0])-1]=row[8]
    result=[]
    for subject,values in visibility.items():
        start=None
        for index in range(length+1):
            low=index<length and np.isfinite(values[index]) and values[index]<=.3
            if low and start is None:start=index
            if low or start is None:continue
            end=index-1
            if end-start+1>=3:
                prior=[t for t in range(max(0,start-15),start) if values[t]>=.7]
                later=[t for t in range(end+1,min(length,end+31)) if values[t]>=.7]
                if prior and later:
                    result.append(dict(subject=subject,original_subject=subjects[subject],low_start=start,
                        low_end=end,pre_clean=prior,post_clean=later[0]))
            start=None
    return result,visibility


def recovery(event,matches,visibility,length):
    subject=event['subject'];pre=[t for t in event['pre_clean'] if subject in matches[t]]
    if not pre:return dict(status='no_pre_occlusion_identity',delay_s=None)
    expected=matches[pre[-1]][subject];first=event['post_clean']
    for step in range(first+1,min(length,first+31)):
        if (visibility[subject][step-1]>=.7 and visibility[subject][step]>=.7 and
            matches[step-1].get(subject)==expected and matches[step].get(subject)==expected):
            return dict(status='same_id_recovered',delay_s=(step-first)/30,pre_id=expected,confirmed_frame=step+1)
    return dict(status='not_recovered_within_1s',delay_s=None,pre_id=expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('datasets/mot17-continuity-native-v1'))
    args=parser.parse_args();report=json.loads((args.study/'results.json').read_text())
    seq=report['sequence'];length=report['frames'];trackeval,_=load_trackeval()
    labels=np.loadtxt(args.dataset/seq/'gt/gt.txt',delimiter=',',ndmin=2)
    events,visibility=episodes(labels,length);results={}
    for case,source in report['results'].items():
        with redirect_stdout(io.StringIO()):
            dataset=trackeval.datasets.MotChallenge2DBox(dict(GT_FOLDER=str(args.dataset),
                TRACKERS_FOLDER=str(args.study/'trackers'),TRACKERS_TO_EVAL=[case],SEQ_INFO={seq:length},
                BENCHMARK='MOT17',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,DO_PREPROC=True,PRINT_CONFIG=False))
            raw=dataset.get_raw_seq_data(case,seq);data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
        matches,switches=clear_trace(data)
        if len(switches)!=source['summary']['IDSW']:raise ValueError('Diagnostic trace disagrees with official CLEAR IDSW')
        rows=[dict(event,recovery=recovery(event,matches,visibility,length)) for event in events]
        successful=[r['recovery']['delay_s'] for r in rows if r['recovery']['status']=='same_id_recovered']
        results[case]=dict(verified_IDSW=len(switches),switch_events=switches,recovery_events=rows,
            eligible_gt_episodes=len(rows),same_id_recovered=len(successful),
            conditional_mean_recovery_delay_s=float(np.mean(successful)) if successful else None,
            no_pre_identity=sum(r['recovery']['status']=='no_pre_occlusion_identity' for r in rows))
    output=dict(sequence=seq,role=report['role'],results=results,
        protocol='GT-only episodes: visibility <=0.3 for >=3 consecutive frames; clean >=0.7 within 15 frames before and 30 after. Recover the pre-episode matched ID on two consecutive clean frames within 1 second after first clean return. Missing pre-ID counts as failure; delay mean is conditional on success. Single-camera diagnostic, not a standard benchmark metric.')
    (args.study/'failure-analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({c:{k:v for k,v in r.items() if k not in ('switch_events','recovery_events')} for c,r in results.items()},indent=2))

if __name__=='__main__':main()
