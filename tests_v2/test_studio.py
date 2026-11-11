"""Real HTTP/upload/session boundaries, no test GPU or physical camera needed."""
from contextlib import redirect_stdout
import http.client
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
from selfwatch_v2.studio import LiveMailbox,Studio,Server,RedirectServer


class MailboxTests(unittest.TestCase):
    def test_latest_frame_per_view_and_receive_order_without_queue_growth(self):
        box=LiveMailbox(['A','B'],'session');image=np.zeros((48,64,3),dtype=np.uint8)
        box.put('A',image);box.put('B',image);box.put('A',image)
        self.assertEqual(len(box.pending),2);self.assertEqual(box.dropped,1)
        iterator=iter(box);first=next(iterator);second=next(iterator)
        self.assertEqual([first[0].camera_id,second[0].camera_id],['B','A'])
        self.assertEqual(second[0].frame_index,1)
        self.assertIn('not hardware',second[0].timestamp_method)
        box.stop();self.assertEqual(list(iterator),[])
        with self.assertRaises(ValueError):box.put('A',image)

    def test_stopped_idle_camera_does_not_leave_worker_blocked(self):
        box=LiveMailbox(['A'],'e',idle_s=.01);box.last_input-=1
        self.assertEqual(list(box),[])
        with self.assertRaises(ValueError):box.put('other',np.zeros((2,2,3)))

    def test_coincident_clock_values_still_produce_strict_capture_order(self):
        box=LiveMailbox(['A'],'session');image=np.zeros((48,64,3),dtype=np.uint8)
        with patch('selfwatch_v2.studio.time.perf_counter',return_value=100.):
            box.put('A',image);first=next(iter(box))[0];box.put('A',image);second=next(iter(box))[0]
        self.assertGreater(second.captured_at_s,first.captured_at_s)


class StudioHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name)
        def worker(job):
            job['status']='running';job['stop'].wait(timeout=5);job['status']='stopped'
        self.studio=Studio(self.root,worker=worker);self.server=Server(('127.0.0.1',0),self.studio)
        self.server_thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.server_thread.start()
        self.port=self.server.server_port

    def tearDown(self):
        self.studio.stop()
        if self.studio.future:self.studio.future.result(timeout=2)
        self.studio.close()
        self.server.shutdown();self.server.server_close();self.server_thread.join(timeout=2);self.temp.cleanup()

    def request(self,path,body=None,token=True,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        options=dict(headers or {})
        if token:options['X-Selfwatch-Token']=self.server.token
        connection.request('GET' if body is None else 'POST',path,body,options)
        response=connection.getresponse();payload=response.read();status=response.status;connection.close()
        return status,json.loads(payload)

    def test_cross_origin_or_missing_token_cannot_start_gpu_or_open_camera(self):
        body=json.dumps(dict(kind='browser',cameras=['A']))
        self.assertEqual(self.request('/api/start',body,token=False)[0],403)
        self.assertEqual(self.request('/api/start',body,headers={'Origin':'https://example.invalid'})[0],403)
        self.assertIsNone(self.studio.future)
        self.assertEqual(self.request('/api/bootstrap')[0],200)

    def test_only_one_worker_and_old_session_cannot_supply_new_camera_frames(self):
        body=json.dumps(dict(kind='browser',cameras=['A'],max_frames=5))
        status,job=self.request('/api/start',body);self.assertEqual(status,200)
        self.assertEqual(self.request('/api/start',body)[0],409)
        image=np.zeros((48,64,3),dtype=np.uint8);_,jpeg=cv2.imencode('.jpg',image)
        self.assertEqual(self.request('/api/frame?run_id=stale&camera=A',jpeg.tobytes())[0],400)
        self.assertEqual(self.request('/api/frame?run_id='+job['run_id']+'&camera=A',jpeg.tobytes())[0],200)
        self.assertEqual(self.request('/api/stop','{}')[0],200)
        self.studio.future.result(timeout=2)
        self.assertEqual(self.studio.state()['status'],'stopped')

    def test_real_video_upload_and_bad_payload_cleanup(self):
        video=self.root/'fixture.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10.,(64,48))
        for value in (0,64,128):writer.write(np.full((48,64,3),value,dtype=np.uint8))
        writer.release();self.assertTrue(video.is_file())
        status,record=self.request('/api/upload?name=store.avi',video.read_bytes());self.assertEqual(status,201)
        self.assertEqual((record['width'],record['height']),(64,48));self.assertEqual(record['name'],'store.avi')
        before=set(self.studio.upload_dir.iterdir())
        self.assertEqual(self.request('/api/upload?name=broken.mp4',b'not a video')[0],400)
        self.assertEqual(set(self.studio.upload_dir.iterdir()),before)
        self.assertEqual(self.request('/api/upload?name=bad.exe',b'bad')[0],400)
        self.assertEqual(self.request('/api/start',json.dumps(dict(kind='videos',upload_ids=[record['id']],max_frames=3)))[0],200)

    def test_paths_and_oversize_camera_headers_are_rejected(self):
        self.assertEqual(self.request('/runs/studio-0000000000000000/../../app.py')[0],404)
        self.assertEqual(self.request('/api/start','[]')[0],400)
        self.assertEqual(self.request('/api/start',json.dumps(dict(kind='browser',cameras=['../x'])))[0],400)
        _,job=self.request('/api/start',json.dumps(dict(kind='browser',cameras=['A'])))
        image=np.zeros((721,1280,3),dtype=np.uint8);_,jpeg=cv2.imencode('.jpg',image)
        self.assertEqual(self.request('/api/frame?run_id='+job['run_id']+'&camera=A',jpeg.tobytes())[0],400)

    def test_processing_fps_uses_observations_and_keeps_per_view_rates(self):
        self.studio.start(dict(kind='browser',cameras=['A','B']))
        with self.studio.lock:
            for n in range(9):self.studio.job['output_ticks'].append(('A' if n%2==0 else 'B',96+n*.5))
        with patch('selfwatch_v2.studio.time.perf_counter',return_value=100.):state=self.studio.state()
        self.assertEqual(state['fps']['total'],2.)
        self.assertEqual(state['fps']['per_camera'],dict(A=1.,B=1.))

    def test_tracking_default_keeps_landmark_work_optional(self):
        self.studio.start(dict(kind='browser',cameras=['A']))
        self.assertEqual(self.studio.job['performance'],'fast')
        self.assertFalse(self.studio.job['pose'])

    def test_balanced_mode_is_accepted_with_landmarks(self):
        self.studio.start(dict(kind='browser',cameras=['A'],performance='balanced',pose=True))
        self.assertEqual(self.studio.job['performance'],'balanced')
        self.assertTrue(self.studio.job['pose'])

    def test_native_capture_reuses_original_stream_and_releases_on_close(self):
        opened=[]
        class Camera:
            def __init__(self,camera_id,*args,**kwargs):self.camera_id=camera_id;self.dropped_frames=0;self.closed=False;opened.append(self)
            def open(self):return True
            def read_frame(self,timeout):return True,np.zeros((48,64,3),dtype=np.uint8)
            def close(self):self.closed=True
        job=dict(camera_indices=[0,1],id='fixture',stop=threading.Event())
        with patch('multicam.camera_stream.CameraStream',Camera):
            stream=self.studio._native_stream(job);a=next(stream);b=next(stream);stream.close()
        self.assertEqual([a[0].camera_id,b[0].camera_id],['camera-0','camera-1'])
        self.assertTrue(all(c.closed for c in opened))

    def test_failed_native_open_closes_the_already_opened_views(self):
        opened=[]
        class Camera:
            def __init__(self,camera_id,*args,**kwargs):self.camera_id=camera_id;self.closed=False;opened.append(self)
            def open(self):return self.camera_id==0
            def close(self):self.closed=True
        job=dict(camera_indices=[0,1],id='fixture',stop=threading.Event())
        with patch('multicam.camera_stream.CameraStream',Camera):
            with self.assertRaisesRegex(ValueError,'Cannot open'):next(self.studio._native_stream(job))
        self.assertTrue(all(c.closed for c in opened))

    def test_sessions_reuse_one_persistent_worker_thread(self):
        owners=[]
        def worker(job):owners.append(threading.get_ident());job['status']='completed'
        self.studio.worker=worker
        for _ in range(2):
            self.studio.start(dict(kind='browser',cameras=['A']))
            self.studio.future.result(timeout=2)
        self.assertEqual(len(owners),2);self.assertEqual(owners[0],owners[1])

    def test_state_revision_avoids_resending_same_frames_and_invalidates_on_new_work(self):
        first,payload,revision=self.studio.state_response()
        self.assertEqual(first,200);self.assertEqual(json.loads(payload)['status'],'idle')
        self.assertEqual(self.studio.state_response(revision),(204,b'',revision))
        self.studio.start(dict(kind='browser',cameras=['A']))
        status,_,updated=self.studio.state_response(revision)
        self.assertEqual(status,200);self.assertNotEqual(revision,updated)

    def test_http_state_delta_has_empty_body_for_unchanged_revision(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        connection.request('GET','/api/state');response=connection.getresponse();response.read()
        revision=response.getheader('X-Selfwatch-Revision');connection.close()
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        connection.request('GET','/api/state?since='+revision)
        response=connection.getresponse();self.assertEqual(response.status,204);self.assertEqual(response.read(),b'');connection.close()

    def test_pause_resume_are_real_session_controls_and_stop_clears_pause(self):
        self.studio.start(dict(kind='browser',cameras=['A']))
        # Wait for the fake inference worker to enter its running state.
        import time
        deadline=time.monotonic()+1
        while self.studio.state()['status']=='loading' and time.monotonic()<deadline:time.sleep(.001)
        status,value=self.request('/api/pause',json.dumps(dict(paused=True)))
        self.assertEqual(status,200);self.assertEqual(value['status'],'paused');self.assertTrue(self.studio.job['pause'].is_set())
        self.assertEqual(self.request('/api/pause',json.dumps(dict(paused=False)))[0],200)
        self.assertFalse(self.studio.job['pause'].is_set())
        self.studio.pause(True);self.studio.stop();self.assertFalse(self.studio.job['pause'].is_set())
        self.assertEqual(len(self.studio.job['controls']),3)

    def test_old_port_redirects_to_the_one_canonical_portal(self):
        alias=RedirectServer(('127.0.0.1',0),self.port)
        worker=threading.Thread(target=alias.serve_forever,daemon=True);worker.start()
        try:
            connection=http.client.HTTPConnection('127.0.0.1',alias.server_port,timeout=3)
            connection.request('GET','/?run=previous-demo');response=connection.getresponse();response.read()
            self.assertEqual(response.status,308)
            self.assertEqual(response.getheader('Location'),f'http://127.0.0.1:{self.port}/?run=previous-demo');connection.close()
        finally:alias.shutdown();alias.server_close();worker.join(timeout=2)

    def test_detection_lists_only_probed_cameras_and_does_not_start_a_model(self):
        def probe(command,**kwargs):
            index=int(command[-1]);return SimpleNamespace(returncode=0,stdout=json.dumps(dict(index=index,available=index==0,width=640,height=480)))
        with patch('selfwatch_v2.studio.subprocess.run',side_effect=probe):
            result=self.studio.find_cameras()
        self.assertEqual([c['index'] for c in result['cameras']],[0]);self.assertIsNone(self.studio.future)
        self.assertFalse(self.studio.probing)


if __name__=='__main__':unittest.main()
