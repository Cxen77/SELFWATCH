"""Run pinned TrackEval MOT preprocessing and standard HOTA/CLEAR/Identity."""
from contextlib import redirect_stdout
import io
from hashlib import sha256
import json
from pathlib import Path
import sys


def load_trackeval():
    root=Path(__file__).resolve().parents[1]/'.runtime/TrackEval-numpy2'
    manifest=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    if manifest['revision']!='12c8791b303e0a0b50f753af204249e622d0281a':raise ValueError('Unexpected TrackEval revision')
    for row in manifest['files']:
        if sha256((root/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('TrackEval source checksum mismatch')
    sys.path.insert(0,str(root)) if str(root) not in sys.path else None
    with redirect_stdout(io.StringIO()):import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(root.resolve()):raise ValueError('Different TrackEval package already loaded')
    return trackeval,manifest


def evaluate_mot(gt_root,tracker_root,sequence,frames,case):
    import numpy as np
    trackeval,provenance=load_trackeval()
    config=dict(GT_FOLDER=str(gt_root),TRACKERS_FOLDER=str(tracker_root),TRACKERS_TO_EVAL=[case],
        SEQ_INFO={sequence:frames},BENCHMARK='MOT17',SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,
        DO_PREPROC=True,PRINT_CONFIG=False)
    with redirect_stdout(io.StringIO()):
        dataset=trackeval.datasets.MotChallenge2DBox(config)
        raw=dataset.get_raw_seq_data(case,sequence)
        data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
        metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False)),
                 trackeval.metrics.Identity(dict(PRINT_CONFIG=False))]
        results={metric.get_name():metric.eval_sequence(data) for metric in metrics}
    def plain(value):
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        return value
    results=plain(results)
    summary={key:results['CLEAR'][key] for key in ('IDSW','Frag','MOTA','CLR_Re','CLR_Pr','CLR_TP','CLR_FN','CLR_FP')}
    summary.update(IDF1=results['Identity']['IDF1'],HOTA=float(np.mean(results['HOTA']['HOTA'])),
        AssA=float(np.mean(results['HOTA']['AssA'])),DetA=float(np.mean(results['HOTA']['DetA'])),
        gt_detections=data['num_gt_dets'],predicted_detections=data['num_tracker_dets'])
    return dict(summary=summary,metrics=results,preprocessing='Official MOT17 distractor/zero-marked handling',
                evaluator_revision=provenance['revision'])
