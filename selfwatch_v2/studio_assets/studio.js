'use strict';
const $=id=>document.getElementById(id),activeStates=new Set(['loading','running','stopping']);
let token='',state={status:'idle'},cameras=[],nativeCameras=[],uploadRecords=[],views=new Map(),knownUploads='',polling=false,launching=false,lastPollAt=0,stateRevision='';
const bodyEdges=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
function message(text,error=false){$('message').textContent=text;$('message').dataset.error=error;}
async function api(path,body={}){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Selfwatch-Token':token},body:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw Error(value.error||'Request failed');return value;}
function refreshUploads(records){
  uploadRecords=records;const signature=records.map(r=>r.id).join();if(signature===knownUploads)return;knownUploads=signature;
  const checked=new Set([...$('uploads').querySelectorAll('input[type=checkbox]:checked')].map(x=>x.value));$('uploads').replaceChildren();
  for(const record of records){
    const item=document.createElement('div');item.className='upload-entry';const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.value=record.id;check.checked=checked.has(record.id)||records.length<=3;
    const text=document.createElement('span');text.textContent=record.name;label.append(check,text);item.append(label);
    const detail=document.createElement('small');detail.textContent=`${record.width} × ${record.height} / ${record.fps.toFixed(1)} FPS metadata`;item.append(detail);
    const offsetLabel=document.createElement('label');offsetLabel.textContent='Time offset (seconds)';const offset=document.createElement('input');offset.type='number';offset.value='0';offset.step='.01';offset.dataset.upload=record.id;offsetLabel.append(offset);item.append(offsetLabel);$('uploads').append(item);
  }
}
function upload(file){return new Promise((resolve,reject)=>{
  const xhr=new XMLHttpRequest();xhr.open('POST','/api/upload?name='+encodeURIComponent(file.name));xhr.setRequestHeader('X-Selfwatch-Token',token);xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.responseType='json';
  xhr.upload.onprogress=e=>{if(e.lengthComputable){$('upload-progress').value=100*e.loaded/e.total;$('upload-status').textContent=`Uploading ${file.name}: ${Math.round(100*e.loaded/e.total)}%`;}};
  xhr.onload=()=>xhr.status===201?resolve(xhr.response):reject(Error(xhr.response?.error||'Video upload failed'));
  xhr.onerror=()=>reject(Error('Upload interrupted. Check the local server and retry.'));xhr.send(file);
});}
async function uploadFiles(){
  $('files').disabled=true;$('upload-progress').hidden=false;
  try{for(const file of $('files').files){if(file.size>512*1024**2)throw Error(`${file.name} exceeds 512 MiB`);await upload(file);}
    $('upload-status').textContent='Upload complete. Select up to three videos, then start tracking.';knownUploads='';await poll();
  }catch(error){$('upload-status').textContent=error.message;}finally{$('files').disabled=false;$('files').value='';$('upload-progress').hidden=true;}
}
async function listCameras(){
  if(!navigator.mediaDevices?.getUserMedia)throw Error('Camera access is unavailable in this browser. Open this local address in Chrome or Edge.');
  const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');const selected=$('device').value;$('device').replaceChildren();
  for(let i=0;i<4;i++)$('device').add(new Option(i===0?'Laptop / local camera 0':`Local camera ${i}`,'local:'+i));$('device').add(new Option('Default browser camera',''));
  for(const [i,device] of devices.entries())$('device').add(new Option(device.label||`Camera ${i+1}`,device.deviceId));
  if([...$('device').options].some(o=>o.value===selected))$('device').value=selected;
}
function releaseCamera(camera){camera.stream.getTracks().forEach(t=>t.stop());clearInterval(camera.timer);camera.video.srcObject=null;camera.element.remove();}
function releaseCameras(){cameras.forEach(releaseCamera);cameras=[];}
async function requestCamera(constraints){
  let expired=false;let timeout;
  const request=navigator.mediaDevices.getUserMedia(constraints).then(stream=>{if(expired){stream.getTracks().forEach(t=>t.stop());throw Error('Camera permission arrived after the request expired. Try again.');}return stream;});
  try{return await Promise.race([request,new Promise((_,reject)=>{timeout=setTimeout(()=>{expired=true;reject(Error('Camera permission is still pending. Use the local laptop-camera option, or allow camera access in Chrome / Edge.'));},15000);})]);}finally{clearTimeout(timeout);}
}
function addNativeCamera(index){
  if(activeStates.has(state.status))throw Error('Stop tracking before changing sources.');
  if(cameras.length)throw Error('Use local cameras or browser cameras in one session.');
  if(nativeCameras.includes(index))throw Error('That camera is already selected.');
  if(nativeCameras.length>=3)throw Error('Three cameras are already selected.');
  const row=document.createElement('div');row.className='camera-entry';const text=document.createElement('span');text.textContent=`Local camera ${index} / ready`;const remove=document.createElement('button');remove.textContent='Remove';
  remove.onclick=()=>{if(activeStates.has(state.status)){message('Stop tracking before changing sources.',true);return;}nativeCameras=nativeCameras.filter(i=>i!==index);row.remove();};row.append(text,remove);$('camera-sources').append(row);nativeCameras.push(index);message('Local camera selected. Press Start tracking to open it.');
}
async function addCamera(defaultCamera=false){
  if(activeStates.has(state.status))throw Error('Stop tracking before changing camera sources.');
  if(cameras.length>=3)throw Error('Three cameras are already selected.');
  if(nativeCameras.length)throw Error('Use local cameras or browser cameras in one session.');
  if(!navigator.mediaDevices?.getUserMedia)throw Error('This browser does not expose camera access. Open the workspace in Chrome or Edge.');
  const selection=defaultCamera?'':$('device').value;
  message('Requesting camera access. Check the browser permission prompt.');
  const stream=await requestCamera({video:{...(selection?{deviceId:{exact:selection}}:{facingMode:'user'}),width:{ideal:1280},height:{ideal:720}},audio:false});
  const deviceId=stream.getVideoTracks()[0].getSettings().deviceId;
  if(cameras.some(c=>c.deviceId===deviceId)){stream.getTracks().forEach(t=>t.stop());throw Error('That camera is already selected.');}
  const camera={id:'camera-'+crypto.randomUUID().slice(0,8),deviceId,stream,video:document.createElement('video'),busy:false,timer:null};
  camera.video.autoplay=true;camera.video.muted=true;camera.video.playsInline=true;camera.video.srcObject=stream;camera.video.className='preview';
  camera.element=document.createElement('div');const row=document.createElement('div');row.className='camera-entry';const name=document.createElement('span');name.textContent=stream.getVideoTracks()[0].label||'Camera preview';
  const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{if(activeStates.has(state.status)){message('Stop tracking before removing a camera.',true);return;}releaseCamera(camera);cameras=cameras.filter(c=>c!==camera);};row.append(name,remove);camera.element.append(row,camera.video);$('camera-sources').append(camera.element);cameras.push(camera);
  await camera.video.play();await listCameras();message('Camera preview is on. Press Start tracking to process it.');
}
async function sendFrame(camera,runId){
  if(camera.busy||!camera.video.videoWidth||!activeStates.has(state.status)||state.run_id!==runId)return;
  camera.busy=true;
  try{
    const canvas=document.createElement('canvas'),factor=Math.min(1,1280/camera.video.videoWidth,720/camera.video.videoHeight);
    canvas.width=Math.round(camera.video.videoWidth*factor);canvas.height=Math.round(camera.video.videoHeight*factor);canvas.getContext('2d').drawImage(camera.video,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.85));if(!blob)return;
    const response=await fetch(`/api/frame?run_id=${encodeURIComponent(runId)}&camera=${camera.id}`,{method:'POST',headers:{'X-Selfwatch-Token':token,'Content-Type':'image/jpeg'},body:blob});
    if(!response.ok){const error=await response.json();throw Error(error.error);}
  }catch(error){if(activeStates.has(state.status))message(error.message,true);}finally{camera.busy=false;}
}
async function start(kind){
  if(launching)return;launching=true;
  try{
    const selected=[...$('uploads').querySelectorAll('input[type=checkbox]:checked')].map(c=>c.value);
    if(!kind){if((cameras.length||nativeCameras.length)&&selected.length)throw Error('Select videos or connected cameras for a session. Uncheck uploaded videos to use cameras.');kind=nativeCameras.length?'local':cameras.length?'browser':'videos';}
    const offsets=selected.map(id=>Number([...$('uploads').querySelectorAll('input[type=number]')].find(i=>i.dataset.upload===id).value));
    const spec={kind,max_frames:Number($('limit').value),pose:$('pose').checked,performance:$('performance').value,rotation:$('rotation').value,upload_ids:selected,offsets_s:offsets,cameras:cameras.map(c=>c.id),camera_indices:nativeCameras};
    const result=await api('/api/start',spec);views.clear();$('views').replaceChildren();$('people').replaceChildren();$('review').hidden=true;$('telemetry').hidden=true;$('run-mode').hidden=true;state={...result,kind,status:'loading'};
    message('Loading models. One GPU worker will process all views.');
    if(kind==='browser')for(const camera of cameras){camera.timer=setInterval(()=>sendFrame(camera,result.run_id),1000/Number($('capture-fps').value));sendFrame(camera,result.run_id);}
    else releaseCameras();await poll();
  }catch(error){message(error.message,true);}finally{launching=false;}
}
function makeView(camera){
  const panel=document.createElement('article');panel.className='camera';const title=document.createElement('h3'),name=document.createElement('span'),clock=document.createElement('small');name.textContent=camera;title.append(name,clock);
  const canvas=document.createElement('canvas');canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`Tracking view ${camera}`);const detail=document.createElement('p');panel.append(title,canvas,detail);$('views').append(panel);
  const view={panel,canvas,clock,detail,revision:-1,image:null,record:null,runId:state.run_id};views.set(camera,view);return view;
}
function draw(view){
  if(!view.image||!view.record)return;const c=view.canvas,f=view.record,ctx=c.getContext('2d');c.width=f.frame.width;c.height=f.frame.height;ctx.drawImage(view.image,0,0,c.width,c.height);
  const scale=Math.max(1,c.width/650),tracks=new Map(f.tracks.filter(t=>!t.visual_suppressed).map(t=>[t.instance_index,t]));ctx.lineWidth=2*scale;
  for(const [index,p] of f.instances.entries()){
    const t=tracks.get(index),color=t?.global_id!=null?`hsl(${(t.global_id*137.5+185)%360} 70% 62%)`:'#7eb9f1',b=p.box_xyxy;
    if($('boxes').checked){ctx.strokeStyle=color;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.font=`600 ${15*scale}px Segoe UI`;const label=t?.global_id!=null?`Person ${t.global_id}${t.protection_state==='FROZEN'?' / locked':''}`:p.label;
      const w=ctx.measureText(label).width+10*scale,x=Math.max(0,Math.min(b[0],c.width-w)),y=Math.max(23*scale,b[1]);ctx.fillStyle='#173044';ctx.fillRect(x,y-23*scale,w,23*scale);ctx.fillStyle=color;ctx.fillText(label,x+5*scale,y-6*scale);
      if(t){ctx.beginPath();t.trajectory.forEach((q,i)=>i?ctx.lineTo(q[1],q[2]):ctx.moveTo(q[1],q[2]));ctx.stroke();}}
    if($('masks').checked&&p.mask_polygon_xy){ctx.strokeStyle=color;ctx.beginPath();p.mask_polygon_xy.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.closePath();ctx.stroke();}
    const k=p.keypoints_xyc;if(!k)continue;
    function visible(i){return k[i]&&k[i][2]>=.45&&k[i][0]>=0&&k[i][0]<=c.width&&k[i][1]>=0&&k[i][1]<=c.height;}
    function dot(i,color){if(!visible(i))return;ctx.fillStyle=color;ctx.beginPath();ctx.arc(k[i][0],k[i][1],2*scale,0,Math.PI*2);ctx.fill();}
    function edge(a,b,color){if(!visible(a)||!visible(b))return;ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(k[a][0],k[a][1]);ctx.lineTo(k[b][0],k[b][1]);ctx.stroke();}
    if($('body').checked){bodyEdges.forEach(([a,b])=>edge(a,b,'#66d6c4'));for(let i=0;i<23;i++)dot(i,'#66d6c4');}
    if($('face').checked)for(let i=23;i<91;i++)dot(i,'#ffe28d');
    if($('hands').checked)for(const start of [91,112]){for(let i=0;i<21;i++)dot(start+i,'#ff99d5');for(let finger=0;finger<5;finger++){let previous=start;for(let joint=1;joint<=4;joint++){const current=start+1+finger*4+joint-1;edge(previous,current,'#ff99d5');previous=current;}}}
  }
}
async function updateView(camera,record){
  const view=views.get(camera)||makeView(camera);if(view.revision===record.revision&&view.runId===state.run_id)return;
  view.revision=record.revision;view.runId=state.run_id;view.record=record;view.clock.textContent=`Frame ${record.frame.frame_index}`;
  view.detail.textContent=`${record.tracks.filter(t=>!t.visual_suppressed).length} person tracks / ${record.instances.filter(p=>p.label.toLowerCase()==='person').length} person detections`;
  const image=new Image(),revision=record.revision,runId=state.run_id;image.src=record.image_url;
  try{await image.decode();if(view.revision!==revision||view.runId!==runId)return;view.image=image;draw(view);}catch(error){message('Cannot load a processed frame. Check the saved run folder.',true);}
}
function memory(latest){
  $('people').replaceChildren();const people=new Map();
  for(const [camera,record] of Object.entries(latest))for(const owner of record.identity_memory||[]){
    if(owner.global_id==null)continue;if(!people.has(owner.global_id))people.set(owner.global_id,[]);people.get(owner.global_id).push({camera,...owner});
  }
  for(const [id,records] of people){const item=document.createElement('article');item.className='person';item.dataset.held=records.every(r=>!r.observed);const title=document.createElement('strong');title.textContent=`Person ${id}`;item.append(title);
    for(const record of records){const line=document.createElement('p');line.textContent=`${record.camera}: ${record.state}${record.observed?' / observed':' / held'}${record.age_s!=null?` / ${record.age_s.toFixed(1)} s`:''}`;item.append(line);}$('people').append(item);}
  if(!people.size){const empty=document.createElement('p');empty.className='hint';empty.textContent='No person identity has been established in these views yet.';$('people').append(empty);}
}
async function poll(){
  if(polling)return;polling=true;lastPollAt=performance.now();
  try{
    const response=await fetch('/api/state'+(stateRevision?'?since='+encodeURIComponent(stateRevision):''),{cache:'no-store'});if(response.status===204)return;if(!response.ok)throw Error('Workspace server unavailable');stateRevision=response.headers.get('X-Selfwatch-Revision')||'';const next=await response.json(),previous=state;state=next;
    if(state.run_id!==previous.run_id){views.clear();$('views').replaceChildren();$('people').replaceChildren();$('review').hidden=true;$('telemetry').hidden=true;$('run-mode').hidden=true;}
    refreshUploads(state.uploads);const active=activeStates.has(state.status);$('run-state').textContent={idle:'Ready',loading:'Loading models',running:'Tracking',stopping:'Stopping',stopped:'Stopped',completed:'Complete',failed:'Needs attention'}[state.status]||state.status;$('run-state').dataset.active=active;
    $('start').disabled=active;$('sample').disabled=active;$('stop').disabled=!active||state.status==='stopping';$('laptop').disabled=active;$('add-camera').disabled=active;$('find').disabled=active;
    for(const [camera,record] of Object.entries(state.latest||{}))updateView(camera,record);
    $('empty').hidden=views.size>0;$('views').dataset.count=views.size;
    if(state.processed){$('run-mode').hidden=false;$('run-mode').textContent=state.performance_mode==='fast'?'This run: Fast mode / IDs, ReID and memory; landmarks and mask polygons skipped.':`This run: Detailed mode / ${state.detailed_pose?'whole-body landmarks on '+state.pose_device.toUpperCase():'landmarks disabled'}; masks retained.`;
      const timings=state.stage_mean_ms||{};const poseText=state.detailed_pose?`${state.pose_device==='cuda'?'GPU':'CPU'} pose ${timings.pose_ms??'—'} ms`:'pose skipped';$('telemetry').hidden=false;$('telemetry').textContent=`${state.fps?.total!=null?state.fps.total.toFixed(1)+' total processing FPS':'FPS warming up'} / ${state.processed} frames · detector ${timings.detector_ms??'—'} ms · ${poseText} · ReID ${timings.reid_ms??'—'} ms · ${state.dropped_live_frames} live frames replaced${state.frame_age_ms!=null?' · '+state.frame_age_ms+' ms server-receipt/dequeue age':''}`;memory(state.latest);
      for(const [camera,view] of views){const fps=state.fps?.per_camera?.[camera];view.clock.textContent=`${fps==null?'FPS warming up':fps.toFixed(1)+' processing FPS'} / frame ${view.record?.frame.frame_index??'—'}`;}}
    if(state.review_url){$('review').hidden=false;$('review').href=state.review_url;}
    if(state.mode_note)$('timing').textContent=state.mode_note+'. Cross-view appearance links are experimental; depth and payment verification are not enabled.';
    if(state.error)message(state.error,true);
    else if(previous.status!==state.status&&state.status==='running')message('Tracking sources. FPS counts processed observations; saved frames will be available for review.');
    else if(previous.status!==state.status&&state.status==='completed')message('Session complete. Open the saved recording to replay and review all processed frames.');
    else if(previous.status!==state.status&&state.status==='stopped')message('Tracking stopped. Camera access is off; processed observations are saved.');
    if(previous.run_id===state.run_id&&activeStates.has(previous.status)&&!active)releaseCameras();
  }catch(error){message(error.message+'. Restart scripts/start-studio.ps1 if the server stopped.',true);if(cameras.some(c=>c.timer))releaseCameras();}finally{polling=false;}
}
async function action(fn){try{await fn();}catch(error){message(error.name==='NotAllowedError'?'Camera access was denied. Allow this local site to use the camera, then try again.':error.name==='NotFoundError'?'No camera was found. Connect a camera or upload a video.':error.message,true);}}
async function init(){
  try{const response=await fetch('/api/bootstrap',{cache:'no-store'});if(!response.ok)throw Error('Cannot connect to the local workspace');const config=await response.json();token=config.token;$('pose-device').textContent=config.pose_device==='cuda'?'Pose runs on GPU with a 2 GiB arena limit. GPU calls stay serialized.':'Pose runs on CPU with two inference threads.';
    $('files').onchange=uploadFiles;$('laptop').onclick=()=>action(()=>addNativeCamera(0));$('add-camera').onclick=()=>action(()=>$('device').value.startsWith('local:')?addNativeCamera(Number($('device').value.split(':')[1])):addCamera(false));
    $('find').onclick=()=>action(async()=>{if(!navigator.mediaDevices?.getUserMedia)throw Error('Camera access is unavailable in this browser. Use a local camera or Chrome / Edge.');message('Requesting camera permission to list browser devices.');const temporary=await requestCamera({video:true,audio:false});try{await listCameras();}finally{temporary.getTracks().forEach(t=>t.stop());}message('Camera list updated. Choose a camera, then add it.');});
    $('start').onclick=()=>start();$('sample').onclick=()=>start('sample');$('stop').onclick=()=>action(async()=>{releaseCameras();await api('/api/stop');message('Stopping the worker and saving the recording.');await poll();});
    $('performance').onchange=()=>{$('pose').disabled=$('performance').value==='fast';message($('performance').value==='fast'?'Next session: Tracking keeps IDs, ReID and memory; landmarks and mask outlines are skipped.':$('performance').value==='balanced'?'Next session: Balanced keeps tracking every frame and rotates landmark work through up to two people per view every half second.':'Next session: Detailed includes whole-body pose on every frame.');};
    for(const id of ['boxes','body','hands','face','masks'])$(id).onchange=()=>views.forEach(draw);
    await poll();setInterval(()=>{if(!document.hidden||performance.now()-lastPollAt>=1000)poll();},100);
  }catch(error){message(error.message,true);}
}
window.addEventListener('pagehide',releaseCameras);
init();
