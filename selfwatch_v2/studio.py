"""Local source workspace: bounded uploads, browser cameras, one GPU owner.

Camera frames arrive over loopback only. Live capture keeps one pending frame
per view (the original CameraStream policy); replay drops no observations.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
from hashlib import sha256
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import itertools
import json
import math
import mimetypes
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import parse_qs,unquote,urlsplit
from types import SimpleNamespace

from .contracts import FrameRef


ACTIVE=('loading','running','paused','stopping')
VIDEO_EXTENSIONS={'.mp4','.avi','.mov','.mkv','.webm','.m4v'}


def contained(root,relative):
    root=root.resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Path is outside the selected run')
    return path


def probe_video(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        ok,image=cap.read();fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or image is None or min(image.shape[:2])<16 or max(image.shape[:2])>8192:
            raise ValueError('This file is not a supported, decodable video')
        if not math.isfinite(fps) or not .1<=fps<=240:
            raise ValueError('Video has no usable FPS metadata; convert it to a constant-frame-rate MP4 first')
        return dict(width=image.shape[1],height=image.shape[0],fps=fps,frame_count=max(0,count),
            timing='Container FPS estimate; not verified camera synchronization or VFR timing')
    finally:cap.release()


class LiveMailbox:
    """One latest packet per camera, delivered in server-receipt order."""
    def __init__(self,cameras,epoch,idle_s=90.):
        self.cameras=set(cameras);self.epoch=epoch;self.idle_s=idle_s
        self.condition=threading.Condition();self.pending={};self.indices={c:0 for c in cameras}
        self.dropped=0;self.received=0;self.last_input=time.monotonic();self.closed=False
        self.last_capture=float('-inf')

    def put(self,camera,image):
        if camera not in self.cameras:raise ValueError('Camera is not in this session')
        with self.condition:
            if self.closed:raise ValueError('Camera session has stopped')
            # Windows Python 3.11 monotonic() can have coarse ticks. Use the
            # high-resolution counter and enforce strict session-wide order.
            now=max(time.perf_counter(),self.last_capture+1e-6);self.last_capture=now
            height,width=image.shape[:2];index=self.indices[camera]
            frame=FrameRef(camera,self.epoch,index,now,time.time(),width,height,'browser-camera',
                'server_receive_monotonic; not hardware capture time')
            self.indices[camera]+=1;self.received+=1
            if camera in self.pending:self.dropped+=1
            self.pending[camera]=(frame,image);self.last_input=time.monotonic();self.condition.notify()

    def stop(self):
        with self.condition:self.closed=True;self.condition.notify_all()

    def __iter__(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    if time.monotonic()-self.last_input>self.idle_s:self.closed=True;break
                    self.condition.wait(timeout=.25)
                if self.closed:return
                key=min(self.pending,key=lambda key:self.pending[key][0].captured_at_s)
                packet=self.pending.pop(key)
            yield packet


class Studio:
    def __init__(self,root,device='0',worker=None,pose_device='cpu',reid_graph=True):
        self.root=Path(root).resolve();self.device=device;self.pose_device=pose_device
        self.reid_graph=reid_graph
        self.upload_dir=self.root/'runs'/'studio-uploads';self.upload_dir.mkdir(parents=True,exist_ok=True)
        self.uploads={};self.lock=threading.RLock();self.job=None;self.future=None
        self.instance_id=secrets.token_hex(6);self.state_encode_lock=threading.Lock()
        self.cached_state_revision=None;self.cached_state_payload=None
        self.probing=False
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='selfwatch-gpu-owner')
        self.models=None;self.worker=worker or self._work
        for path in sorted(self.upload_dir.glob('*.json'))[-100:]:
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                media=contained(self.upload_dir,record['file'])
                if re.fullmatch('[a-f0-9]{24}',record['id']) and media.is_file():self.uploads[record['id']]=record
            except (ValueError,KeyError,OSError):continue

    def register_upload(self,path,name,digest):
        metadata=probe_video(path)
        identifier=path.stem
        record=dict(id=identifier,name=Path(name.replace('\\','/')).name[:160],file=path.name,
            bytes=path.stat().st_size,sha256=digest,**metadata)
        (self.upload_dir/(identifier+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        with self.lock:self.uploads[identifier]=record
        return record

    def state(self):
        with self.lock:
            if not self.job:return dict(status='idle',uploads=list(self.uploads.values()),latest={},cameras=[],processed=0)
            j=self.job
            now=time.perf_counter() if j['status'] in ACTIVE else j.get('finished_perf',time.perf_counter())
            recent=[tick for tick in j['output_ticks'] if tick[1]>=now-5.]
            def rate(events):
                if len(events)<3 or events[-1]-events[0]<1.:return None
                return round((len(events)-1)/(events[-1]-events[0]),2)
            times=[t for _,t in recent]
            fps=dict(total=rate(times),per_camera={camera:rate([t for c,t in recent if c==camera]) for camera in j['cameras']},
                meaning='Processed observations per second over recent output intervals, up to 5 seconds; not display redraw FPS')
            stages={key:round(sum(r.get(key,0.) for r in j['recent_timings'])/len(j['recent_timings']),1)
                    for key in ('detector_ms','pose_ms','reid_ms','cognitive_engine_ms','association_ms','persistence_ms','publish_ms') } if j['recent_timings'] else {}
            return dict(status=j['status'],run_id=j['id'],kind=j['kind'],processed=j['processed'],
                limit=j['limit'],cameras=j['cameras'],latest=dict(j['latest']),error=j.get('error'),
                elapsed_s=round(time.monotonic()-j['started'],1),uploads=list(self.uploads.values()),
                dropped_live_frames=j['mailbox'].dropped if j.get('mailbox') else j.get('native_dropped',0),
                review_url=f'/runs/{j["id"]}/' if j.get('saved_frames',0) else None,
                cross_camera_links=j.get('links',0),mode_note=j['timing_note'],fps=fps,stage_mean_ms=stages,
                performance_mode=j['performance'],detailed_pose=j['pose'],frame_age_ms=j.get('frame_age_ms'),pose_device=self.pose_device)

    def state_response(self,since=None):
        """Serialize each published revision once, shared by all open tabs."""
        with self.state_encode_lock:
            with self.lock:
                job=self.job
                fields=(self.instance_id,job['id'] if job else 'idle',job['processed'] if job else 0,
                    job['status'] if job else 'idle',len(self.uploads),bool(job and job.get('finished_perf')),
                    int(time.perf_counter()) if job and job['status'] in ACTIVE else 0)
                revision=':'.join(map(str,fields))
            if since==revision:return 204,b'',revision
            if revision!=self.cached_state_revision:
                self.cached_state_payload=json.dumps(self.state(),allow_nan=False).encode()
                self.cached_state_revision=revision
            return 200,self.cached_state_payload,revision

    def start(self,spec):
        from .replay import Source,load_manifest
        if not isinstance(spec,dict):raise ValueError('Expected a session configuration object')
        kind=spec.get('kind');limit=int(spec.get('max_frames',600))
        if kind not in ('videos','browser','local','sample') or not 1<=limit<=3000:
            raise ValueError('Choose a source and a frame limit between 1 and 3000')
        mode=spec.get('rotation','off')
        if mode not in ('off','weak','memory','always'):raise ValueError('Invalid person-recovery mode')
        performance=spec.get('performance','fast')
        if performance not in ('detailed','balanced','fast'):raise ValueError('Invalid performance mode')
        with self.lock:
            if self.probing:raise RuntimeError('Camera detection is running; wait for it to finish')
            if self.future and not self.future.done():raise RuntimeError('Stop the current session before starting another')
            run_id='studio-'+secrets.token_hex(8);sources=[]
            if kind=='sample':
                sources=load_manifest(self.root/'datasets/retailaction-replay-verified/000398.json')
                cameras=[s.camera_id for s in sources];labels={c:c for c in cameras}
                note='Recorded dataset timestamps; views are not assumed hardware-synchronized'
            elif kind=='videos':
                identifiers=spec.get('upload_ids',[])
                if not 1<=len(identifiers)<=3 or len(set(identifiers))!=len(identifiers):raise ValueError('Select one to three different uploaded videos')
                offsets=spec.get('offsets_s',[0.]*len(identifiers))
                if len(offsets)!=len(identifiers):raise ValueError('One offset is required per view')
                labels={}
                for index,identifier in enumerate(identifiers):
                    if identifier not in self.uploads:raise ValueError('Uploaded video is unavailable')
                    record=self.uploads[identifier];camera=f'camera-{index+1}';labels[camera]=record['name']
                    sources.append(Source(camera,contained(self.upload_dir,record['file']),run_id,record['fps'],float(offsets[index])))
                cameras=list(labels);note='Playback timestamps estimated from container FPS + entered offsets; zero offsets do not establish synchronization'
            elif kind=='local':
                indices=spec.get('camera_indices',[])
                if (not 1<=len(indices)<=3 or any(type(i) is not int or not 0<=i<=3 for i in indices)
                    or len(set(indices))!=len(indices)):raise ValueError('Choose one to three local camera indices, 0–3')
                cameras=[f'camera-{i}' for i in indices];labels={c:c for c in cameras}
                note='Local camera dequeue time; no measured hardware capture clock or synchronization'
            else:
                cameras=spec.get('cameras',[])
                if (not 1<=len(cameras)<=3 or len(set(cameras))!=len(cameras)
                    or any(not isinstance(c,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,40}',c) for c in cameras)):
                    raise ValueError('Choose one to three distinct browser cameras')
                labels={c:c for c in cameras};note='Server receipt time; no hardware camera clock, metric depth or calibrated synchronization'
            output=self.root/'runs'/run_id;output.mkdir();(output/'images').mkdir()
            job=dict(id=run_id,kind=kind,limit=limit,cameras=cameras,labels=labels,sources=sources,
                status='loading',processed=0,latest={},started=time.monotonic(),output=output,
                stop=threading.Event(),pause=threading.Event(),controls=[],pose=bool(spec.get('pose',True)),rotation=mode,timing_note=note,
                performance=performance,output_ticks=deque(maxlen=600),recent_timings=deque(maxlen=30),
                camera_indices=spec.get('camera_indices',[]) if kind=='local' else [],
                mailbox=LiveMailbox(cameras,run_id) if kind=='browser' else None)
            if performance=='fast':job['pose']=False
            self.job=job
            self.future=self.executor.submit(self._run,job)
            return dict(run_id=run_id,status='loading',cameras=cameras)

    def _run(self,job):
        try:self.worker(job)
        except Exception as exc:
            (job['output']/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            with self.lock:job.update(status='failed',error=str(exc))
        finally:
            if job['mailbox']:job['mailbox'].stop()
            with self.lock:job['finished_perf']=time.perf_counter()

    def stop(self):
        with self.lock:
            if self.job and self.job['status'] in ACTIVE:
                self.job['status']='stopping';self.job['stop'].set()
                self.job['pause'].clear()
                if self.job['mailbox']:self.job['mailbox'].stop()
        return dict(status='stopping')

    def pause(self,paused):
        if type(paused) is not bool:raise ValueError('Paused must be true or false')
        with self.lock:
            job=self.job
            if not job or job['status'] not in ('running','paused'):raise ValueError('No running analysis to pause or resume')
            if paused:job['pause'].set()
            else:job['pause'].clear()
            job['status']='paused' if paused else 'running'
            job['controls'].append(dict(action=job['status'],received_at_s=time.time(),processed=job['processed']))
            return dict(status=job['status'])

    def find_cameras(self):
        with self.lock:
            if self.probing or self.future and not self.future.done():raise RuntimeError('Stop analysis or wait for camera detection to finish')
            self.probing=True
        found=[]
        try:
            for index in range(4):
                try:
                    result=subprocess.run([sys.executable,'-m','selfwatch_v2.camera_probe','--index',str(index)],
                        cwd=str(Path(__file__).resolve().parents[1]),capture_output=True,text=True,timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                    if result.returncode==0:
                        camera=json.loads(result.stdout)
                        if camera.get('available'):found.append(camera)
                except (subprocess.TimeoutExpired,ValueError):continue
        finally:
            with self.lock:self.probing=False
        return dict(cameras=found)

    def close(self):
        self.stop();self.executor.shutdown(wait=True,cancel_futures=True)

    def push_frame(self,run_id,camera,payload):
        import cv2
        import numpy as np
        from PIL import Image
        with self.lock:
            job=self.job
            if not job or run_id!=job['id'] or job['kind']!='browser' or job['status'] not in ('loading','running','paused'):
                raise ValueError('This camera session is no longer active')
            mailbox=job['mailbox']
        with Image.open(io.BytesIO(payload)) as header:
            if header.format!='JPEG' or min(header.size)<16 or header.width>1280 or header.height>1280 or header.width*header.height>1280*720:
                raise ValueError('Camera frames must be JPEG, at most 1280 × 720 pixels')
        image=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Cannot decode camera frame')
        mailbox.put(camera,image)
        return dict(queued=True,dropped=mailbox.dropped)

    def _load_models(self,with_pose):
        import torch
        from .yolo import YoloPerception
        from .appearance import AppearanceEncoder
        from .wholebody import WholeBody
        torch.set_num_threads(2)
        if self.models is None:
            device='cpu' if self.device=='cpu' else f'cuda:{self.device}' if self.device.isdecimal() else self.device
            self.models=dict(detector=YoloPerception(self.root/'weights/v2/yolo26n-seg.pt',self.device),
                appearance=AppearanceEncoder(self.root/'weights/osnet/osnet_x1_0_msmt17.pth',device=device,batch_buckets=True),pose=None,
                detector_warmed=False,pose_warmed=False,reid_warmed=False,reid_graph_warmed=False)
        if with_pose and self.models['pose'] is None:
            self.models['pose']=WholeBody(self.root/'weights/v2/rtmw/wholebody133.onnx',self.pose_device,max_people=4)
        # Keep lazy kernel/allocator startup in the explicit loading phase.
        # Synthetic warmup is never stored, tracked or published as evidence.
        import numpy as np
        if not self.models['detector_warmed']:
            for width,height in ((600,600),(1280,720)):
                frame=FrameRef('warmup','synthetic',0,0.,time.time(),width,height,'synthetic_model_warmup')
                self.models['detector'].predict(frame,np.zeros((height,width,3),dtype=np.uint8))
            self.models['detector_warmed']=True
        if with_pose and not self.models['pose_warmed']:
            image=np.zeros((600,600,3),dtype=np.uint8)
            for _ in range(3):self.models['pose'].model(image,bboxes=[(100,50,400,550)])
            self.models['pose_warmed']=True
        if not self.models['reid_warmed']:
            extractor=self.models['appearance'].extractor
            crop=np.zeros((extractor.input_height,extractor.input_width,3),dtype=np.uint8)
            for batch_size in (1,2,4,8,16):extractor.extract_batch([crop]*batch_size)
            self.models['reid_warmed']=True
        if self.reid_graph and not self.models['reid_graph_warmed']:
            extractor=self.models['appearance'].extractor
            if extractor._device.type=='cuda':
                from .cuda_reid import GraphedReID
                extractor._model=GraphedReID(extractor._model,extractor._device,extractor._half,extractor._reid_stream)
                self.models['appearance'].provenance['cuda_graph_batch_sizes']=sorted(extractor._model.graphs)
            self.models['reid_graph_warmed']=True
        # Apply the CPU budget after optional libraries have initialized too.
        torch.set_num_threads(2)
        return self.models

    def _native_stream(self,job):
        """Reuse the old project's real capture implementation, not its detector."""
        from multicam.camera_stream import CameraStream
        opened=[]
        try:
            for index in job['camera_indices']:
                import cv2
                camera=CameraStream(index,index,SimpleNamespace(close=lambda:None),None,None,label=f'Local camera {index}',
                    capture_backend=cv2.CAP_DSHOW if sys.platform=='win32' else None,capture_size=(1280,720))
                opened.append(camera)
                if not camera.open():raise ValueError(f'Cannot open local camera {index}. Close other camera apps or select another source.')
            indices={c.camera_id:0 for c in opened}
            while not job['stop'].is_set():
                emitted=False
                for camera in opened:
                    alive,image=camera.read_frame(timeout=.001)
                    if not alive:raise RuntimeError(f'Local camera {camera.camera_id} disconnected')
                    if image is None:continue
                    height,width=image.shape[:2];index=indices[camera.camera_id];indices[camera.camera_id]+=1
                    with self.lock:job['native_dropped']=sum(c.dropped_frames for c in opened)
                    yield FrameRef(f'camera-{camera.camera_id}',job['id'],index,time.perf_counter(),time.time(),
                        width,height,f'localcamera:{camera.camera_id}','server_dequeue_perf_counter; not hardware capture time'),image
                    emitted=True
                if not emitted:job['stop'].wait(.005)
        finally:
            for camera in opened:camera.close()

    def _work(self,job):
        import cv2
        from .legacy_guard import LegacyIdentityGuard
        from .identity import SharedIdentity
        from .pipeline import process
        from .replay import replay
        from .store import ObservationStore
        from .demo import write_review_assets
        from .rotation import RotationPersonPerception
        models=self._load_models(job['pose']);detector=copy.copy(models['detector'])
        detector.include_masks=job['performance']=='detailed'
        detector.provenance=dict(detector.provenance,output_mask_polygons=detector.include_masks)
        if job['stop'].is_set():job['status']='stopped';return
        if job['rotation']!='off':detector=RotationPersonPerception(detector,mode=job['rotation'])
        # Scheduler state is session-local; checkpoint/model allocations are reused.
        appearance=models['appearance'];appearance.states.clear();appearance.attention.clear()
        class NoPose:
            provenance=dict(providers=['disabled'],schema=None)
            def refine(self,observation,image):return observation
        pose=models['pose'] if job['pose'] else NoPose()
        if job['pose'] and job['performance']=='balanced':
            from .pose_schedule import ScheduledPose
            pose=ScheduledPose(pose,interval_s=.5,max_people=2)
        identity=SharedIdentity(allow_overlap=True)
        trackers={c:LegacyIdentityGuard(log_dir=job['output']/'cognitive_logs') for c in job['cameras']}
        raw=iter(job['mailbox']) if job['mailbox'] else self._native_stream(job) if job['kind']=='local' else replay(job['sources'])
        def stream():
            for frame,image in itertools.islice(raw,job['limit']):
                if job['stop'].is_set():return
                was_paused=job['pause'].is_set()
                while job['pause'].is_set() and not job['stop'].is_set():job['stop'].wait(.05)
                if job['stop'].is_set():return
                if was_paused and job['kind'] in ('local','browser'):continue
                h,w=image.shape[:2]
                if max(h,w)>1280:
                    factor=1280/max(h,w);image=cv2.resize(image,(round(w*factor),round(h*factor)))
                    frame=replace(frame,width=image.shape[1],height=image.shape[0])
                yield frame,image
        run_config=dict(kind=job['kind'],cameras=job['cameras'],labels=job['labels'],max_frames=job['limit'],
            detector=detector.provenance,pose=pose.provenance,reid=appearance.provenance,
            timing_note=job['timing_note'],max_source_side=1280,live_capture_policy='latest frame per view',
            replay_policy='no observation dropping',cpu_gpu_overlap=self.pose_device=='cpu',retail_alerts=False,
            performance=job['performance'],detailed_pose=job['pose'],mask_polygons=job['performance']=='detailed',
            pose_schedule=pose.provenance.get('scheduling'),
            model_warmup='Synthetic detector shapes and three pose crops; excluded from stored observations and timed processing',
            uploaded_sources=[dict(camera_id=s.camera_id,path=str(s.path),fps=s.fps,offset_s=s.offset_s) for s in job['sources']])
        (job['output']/'run.json').write_text(json.dumps(run_config,indent=2),encoding='utf-8')
        records=[];iterator=None;start=time.perf_counter()
        with self.lock:job['status']='paused' if job['pause'].is_set() else 'running'
        try:
            with ObservationStore(job['output']/'observations.sqlite') as store:
                iterator=process(stream(),detector,pose,trackers,appearance,identity,parallel=self.pose_device=='cpu')
                for observation,image,tracks,diagnostics,timings in iterator:
                    persistence_started=time.perf_counter()
                    store.append(observation);filename=f'images/{len(records):06d}.jpg'
                    if not cv2.imwrite(str(job['output']/filename),image,[cv2.IMWRITE_JPEG_QUALITY,85]):raise RuntimeError('Cannot save run image')
                    record=dict(observation.to_dict(),image=filename,tracks=tracks,appearance_diagnostics=diagnostics,
                        observation_id=observation.frame.observation_id,identity_memory=timings.pop('identity_memory',[]),
                        identity_audit=timings.pop('identity_audit',{}),perception_diagnostics=timings.pop('perception_diagnostics',None),timings=timings)
                    records.append(record)
                    timings['persistence_ms']=(time.perf_counter()-persistence_started)*1000
                    # The live UI needs current evidence, not the full saved
                    # audit/diagnostics on every polling request. Durable
                    # records retain the complete research payload.
                    latest={key:record[key] for key in ('frame','instances','tracks','identity_memory')}
                    latest.update(image_url=f'/runs/{job["id"]}/{filename}',revision=len(records))
                    publish_started=time.perf_counter()
                    with self.lock:
                        job['latest'][observation.frame.camera_id]=latest;job['processed']=len(records)
                        job['links']=sum(e.get('association_kind')=='cross_camera' for e in identity.events)
                        job['output_ticks'].append((observation.frame.camera_id,time.perf_counter()))
                        timings['publish_ms']=(time.perf_counter()-publish_started)*1000
                        job['recent_timings'].append(dict(timings))
                        if job['kind'] in ('browser','local'):job['frame_age_ms']=round((time.perf_counter()-observation.frame.captured_at_s)*1000,1)
        finally:
            if iterator:iterator.close()
            if hasattr(raw,'close'):raw.close()
            for tracker in trackers.values():tracker.close()
        if records:
            data=dict(schema_version=1,name='SELFWATCH camera workspace',run_id=job['id'],sample=job['id'],
                cameras=job['cameras'],frames=records,pose=pose.provenance,identity=identity.snapshot(),variants={},
                tracker_engine='legacy',timeline_semantics=job['timing_note'],
                identity_semantics='Experimental visual associations; no identity accuracy claim')
            (job['output']/'data.json').write_text(json.dumps(data,allow_nan=False),encoding='utf-8');write_review_assets(job['output'])
            (job['output']/'summary.json').write_text(json.dumps(dict(frames=len(records),wall_seconds=time.perf_counter()-start,
                accuracy='Not measured',cross_camera_links=job.get('links',0),retail_alerts=False),indent=2),encoding='utf-8')
            (job['output']/'controls.json').write_text(json.dumps(job['controls'],indent=2),encoding='utf-8')
        with self.lock:
            job['saved_frames']=len(records);job['status']='stopped' if job['stop'].is_set() else 'completed'


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,studio):
        self.studio=studio;self.token=secrets.token_urlsafe(32);self.upload_slots=threading.BoundedSemaphore(2)
        super().__init__(address,Handler)


class RedirectServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,target_port):
        self.target_port=target_port
        super().__init__(address,RedirectHandler)


class RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url=urlsplit(self.path)
        path=url.path if url.path.startswith('/') else '/'
        location=f'http://127.0.0.1:{self.server.target_port}'+path+('?' + url.query if url.query else '')
        self.send_response(308);self.send_header('Location',location);self.send_header('Content-Length','0')
        self.send_header('Cache-Control','no-store');self.end_headers()
    do_HEAD=do_GET


class Handler(BaseHTTPRequestHandler):
    server_version='SELFWATCH/2'
    def _headers(self,status,mime,length,extra=None):
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(length))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for name,value in (extra or {}).items():self.send_header(name,value)
        self.send_header('Connection','close');self.end_headers();self.close_connection=True

    def _json(self,value,status=200):
        payload=json.dumps(value,allow_nan=False).encode();self._headers(status,'application/json; charset=utf-8',len(payload));self.wfile.write(payload)

    def _validate(self,mutation=False):
        port=self.server.server_port;host=self.headers.get('Host','')
        if host not in (f'127.0.0.1:{port}',f'localhost:{port}'):raise PermissionError('Use the local SELFWATCH address')
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
            raise PermissionError('Cross-origin requests are not permitted')
        if mutation and not secrets.compare_digest(self.headers.get('X-Selfwatch-Token',''),self.server.token):
            raise PermissionError('Reload the workspace before changing a session')

    def _length(self,maximum):
        values=self.headers.get_all('Content-Length',[])
        if self.headers.get('Transfer-Encoding') or len(values)!=1:raise ValueError('A single Content-Length is required')
        length=int(values[0])
        if not 0<length<=maximum:raise ValueError(f'Request exceeds the {maximum//1024**2} MiB limit or is empty')
        self.connection.settimeout(30)
        return length

    def _read(self,maximum):
        length=self._length(maximum);payload=self.rfile.read(length)
        if len(payload)!=length:raise ValueError('Incomplete request')
        return payload

    def do_GET(self):
        try:
            self._validate();url=urlsplit(self.path);route=unquote(url.path);studio=self.server.studio
            if route=='/api/bootstrap':return self._json(dict(token=self.server.token,max_upload_mib=512,max_cameras=3,pose_device=studio.pose_device))
            if route=='/api/state':
                since=parse_qs(url.query).get('since',[None])[0]
                status,payload,revision=studio.state_response(since)
                self._headers(status,'application/json; charset=utf-8',len(payload),{'X-Selfwatch-Revision':revision})
                self.wfile.write(payload);return
            if route=='/project-status.json':
                from .project_status import read_status
                return self._json(read_status(studio.root/'docs/PROJECT_LOG.md'))
            assets=Path(__file__).parent/'studio_assets'
            if route in ('/','/studio.js','/studio.css','/plan.js','/portal.js','/portal.css'):
                path=assets/('portal.html' if route=='/' else route[1:])
            elif route=='/plan/':path=assets/'plan.html'
            elif route=='/execution-plan.md':path=studio.root.parent/'SELFWATCH_RESEARCH'/'EXECUTION_PLAN_V2.md'
            elif route=='/project-log.md':path=studio.root/'docs/PROJECT_LOG.md'
            elif route.startswith('/runs/'):
                match=re.fullmatch(r'/runs/(studio-[a-f0-9]{16})/(.*)',route)
                if not match:raise FileNotFoundError()
                root=studio.root/'runs'/match[1]
                if not (root/'run.json').is_file():raise FileNotFoundError()
                path=contained(root,match[2] or 'index.html')
            elif route.startswith('/research/'):
                match=re.fullmatch(r'/research/(combined|identity|overhead)/(.*)',route)
                if not match:raise FileNotFoundError()
                roots=dict(combined='combined-cognitive-v2-000398',identity='identity-study-pilot-v3',overhead='overhead-detector-pilot-v1')
                path=contained(studio.root/'runs'/roots[match[1]],match[2] or 'index.html')
            else:raise FileNotFoundError()
            if not path.is_file() or path.suffix.lower() not in ('.html','.js','.css','.json','.jpg','.png','.md'):
                raise FileNotFoundError()
            payload=path.read_bytes();mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self._headers(200,mime,len(payload));self.wfile.write(payload)
        except PermissionError as exc:self._json(dict(error=str(exc)),403)
        except (ValueError,FileNotFoundError):self._json(dict(error='Resource not found'),404)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        try:
            self._validate(mutation=True);url=urlsplit(self.path);query=parse_qs(url.query);studio=self.server.studio
            if url.path=='/api/upload':return self._upload(query)
            if url.path=='/api/frame':
                return self._json(studio.push_frame(query.get('run_id',[''])[0],query.get('camera',[''])[0],self._read(2*1024**2)))
            if url.path=='/api/start':return self._json(studio.start(json.loads(self._read(16384))))
            if url.path=='/api/pause':return self._json(studio.pause(json.loads(self._read(16384))['paused']))
            if url.path=='/api/cameras':self._read(16384);return self._json(studio.find_cameras())
            if url.path=='/api/stop':self._read(16384);return self._json(studio.stop())
            self._json(dict(error='Route not found'),404)
        except PermissionError as exc:self._json(dict(error=str(exc)),403)
        except RuntimeError as exc:self._json(dict(error=str(exc)),409)
        except (BrokenPipeError,ConnectionResetError):pass
        except (ValueError,KeyError,TypeError,OSError) as exc:self._json(dict(error=str(exc)),400)
        except Exception:
            traceback.print_exc();self._json(dict(error='The local server could not process this request'),500)

    def _upload(self,query):
        studio=self.server.studio;name=query.get('name',[''])[0];suffix=Path(name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:raise ValueError('Select MP4, AVI, MOV, MKV, M4V or WebM video')
        length=self._length(512*1024**2)
        if length+1024**3>shutil.disk_usage(studio.upload_dir).free:raise ValueError('Insufficient disk space for this upload')
        if sum(p.stat().st_size for p in studio.upload_dir.iterdir() if p.is_file())+length>5*1024**3:
            raise ValueError('Local upload storage reached 5 GiB. Archive uploads before adding more.')
        if not self.server.upload_slots.acquire(blocking=False):raise RuntimeError('Two uploads are already active; wait for one to finish')
        target=studio.upload_dir/(secrets.token_hex(12)+suffix);complete=False
        try:
            digest=sha256();remaining=length
            with target.open('xb') as output:
                while remaining:
                    block=self.rfile.read(min(1024**2,remaining))
                    if not block:raise ValueError('Upload ended early')
                    output.write(block);digest.update(block);remaining-=len(block)
            record=studio.register_upload(target,name,digest.hexdigest());complete=True
            self._json(record,201)
        finally:
            if not complete:target.unlink(missing_ok=True)
            self.server.upload_slots.release()


def main():
    import sys
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766);parser.add_argument('--device',default='0')
    parser.add_argument('--alias-port',type=int,default=8765)
    parser.add_argument('--pose-device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--no-reid-graph',action='store_true',help='Use original OSNet execution for comparison/troubleshooting')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    pose_device='cpu'
    if args.pose_device=='cuda':
        from .gpu_runtime import activate
        activate(args.root);pose_device='cuda'
    elif args.pose_device=='auto' and args.device!='cpu':
        marker=args.root/'.runtime/pose-gpu-validated.json'
        if marker.is_file():
            try:
                validation=json.loads(marker.read_text());metadata=json.loads((args.root/'.runtime/pose-gpu-runtime.json').read_text())
                if validation.get('validated') and validation['wheel_sha256']==metadata['sha256']:
                    from .gpu_runtime import activate
                    activate(args.root)
                    import torch
                    if torch.cuda.mem_get_info()[0]>=3*1024**3:pose_device='cuda'
            except Exception as exc:print(f'GPU pose unavailable; using CPU: {exc}',flush=True)
    studio=Studio(args.root,args.device,pose_device=pose_device,reid_graph=not args.no_reid_graph);server=Server(('127.0.0.1',args.port),studio)
    alias=None
    if args.alias_port and args.alias_port!=args.port:
        alias=RedirectServer(('127.0.0.1',args.alias_port),args.port)
        threading.Thread(target=alias.serve_forever,daemon=True,name='selfwatch-old-address').start()
    print(f'SELFWATCH workspace: http://127.0.0.1:{args.port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        studio.close();server.server_close()
        if alias:alias.shutdown();alias.server_close()


if __name__=='__main__':main()
