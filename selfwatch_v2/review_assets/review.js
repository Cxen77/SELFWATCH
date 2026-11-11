"use strict";
const $=id=>document.getElementById(id);
let data, times=[], cursor=0, playing=false, lastWall=null, playTime=0, views=[], marks={}, storageKey;
const bodyEdges=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
function notify(message){$('status').textContent=message;}
function key(frame){return frame.observation_id;}
function trackingCase(){return $('tracking-case')?.value||'legacy';}
function selectedIdentity(){return data.identity_by_case?.[trackingCase()]||data.identity;}
function selectedFrame(frame){
 const selected=trackingCase();if(!frame.tracks_by_case)return frame;
 return {...frame,tracks:frame.tracks_by_case[selected],
   identity_memory:frame.identity_memory_by_case?.[selected]||[],
   identity_audit:frame.identity_audit_by_case?.[selected]||{}};
}
function entry(frame){return marks[key(frame)]||(marks[key(frame)]={reviewed:false,marks:[],issues:[]});}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(marks));}catch(e){notify('Browser could not save this review. Export it now to keep your marks.');}}
function stop(){playing=false;$('play').textContent='Play';$('play').setAttribute('aria-label','Play recording');lastWall=null;}
function currentView(){return views[Number($('active').value)];}
function status(){const all=Object.values(marks), reviewed=all.filter(x=>x.reviewed).length;const v=currentView();$('review-status').textContent=`${reviewed} / ${data.frames.length} frames reviewed. ${v&&v.frame?(marks[key(v.frame)]?.marks.length||0):0} marks on active frame.`;$('reviewed').textContent=v?.frame&&marks[key(v.frame)]?.reviewed?'Mark active frame unreviewed':'Mark active frame reviewed';}
function point(e,canvas){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height,(e.clientY-r.top)*canvas.height/r.height))];}
function draw(view){
 const original=view.frame, c=view.canvas, ctx=c.getContext('2d');if(!original)return;
 const alternate=data.variants?.[$('variant').value]?.[original.observation_id];
 const f=alternate?{...alternate,observation_id:original.observation_id,tracks:[]}:original;
 if(!view.image.complete||view.image.naturalWidth===0)return;
 ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(view.image,0,0,c.width,c.height);
 const scale=Math.max(1,c.width/550), threshold=Number($('threshold').value);ctx.lineWidth=2*scale;
 const tracks=new Map(f.tracks.map(t=>[t.instance_index,t]));
 const PALETTE=['#2563eb','#0284c7','#0d9488','#4f46e5','#059669','#d97706'];
 for(const [index,p] of f.instances.entries()){
   const b=p.box_xyxy,rawTrack=tracks.get(index),t=rawTrack?.visual_suppressed?null:rawTrack;
   if($('boxes').checked){
     const color=t&&t.global_id!=null?PALETTE[t.global_id%PALETTE.length]:'#0284c7';
     ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2*scale;
     ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.font=`600 ${13*scale}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
     const identity=t&&t.global_id!=null?`Person ${t.global_id}`:p.label;
     const state=t?.protection_state==='FROZEN'?' · LOCKED':t?.protection_state==='TENTATIVE'?' · provisional':'';
     const label=`${identity}${state}`;
     const width=ctx.measureText(label).width+14*scale;
     const height=22*scale;
     const x=Math.max(0,Math.min(b[0],c.width-width)),y=Math.max(height,b[1]);
     ctx.fillStyle='rgba(9, 9, 11, 0.9)';
     if(ctx.roundRect){ctx.beginPath();ctx.roundRect(x,y-height,width,height,4*scale);ctx.fill();}else{ctx.fillRect(x,y-height,width,height);}
     ctx.fillStyle='#ffffff';ctx.fillText(label,x+6*scale,y-6*scale);
     if(t){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=1.5*scale;t.trajectory.forEach((q,i)=>i?ctx.lineTo(q[1],q[2]):ctx.moveTo(q[1],q[2]));ctx.stroke();}}
   if($('masks').checked&&p.mask_polygon_xy){ctx.beginPath();p.mask_polygon_xy.forEach((q,i)=>i?ctx.lineTo(...q):ctx.moveTo(...q));ctx.closePath();ctx.stroke();}
   const k=p.keypoints_xyc;if(!k)continue;
   function dot(i,color,r=1.8){const q=k[i];if(q&&q[2]>=threshold&&q[0]>=0&&q[0]<=c.width&&q[1]>=0&&q[1]<=c.height){ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r*scale,0,2*Math.PI);ctx.fill();}}
   function edge(a,b,color,lw=1.5){if(k[a]&&k[b]&&k[a][2]>=threshold&&k[b][2]>=threshold){ctx.strokeStyle=color;ctx.lineWidth=lw*scale;ctx.beginPath();ctx.moveTo(k[a][0],k[a][1]);ctx.lineTo(k[b][0],k[b][1]);ctx.stroke();}}
   if($('body').checked){bodyEdges.forEach(([a,b])=>edge(a,b,'rgba(56, 189, 248, 0.75)',1.8));for(let i=0;i<Math.min(23,k.length);i++)dot(i,'#ffffff',2.0);}
   if($('face').checked)for(let i=23;i<91;i++)dot(i,'rgba(226, 232, 240, 0.55)',1.2);
   if($('hands').checked)for(const start of [91,112]){for(let j=0;j<21;j++)dot(start+j,'#38bdf8',1.5);for(const base of [1,5,9,13,17]){edge(start,start+base,'rgba(56, 189, 248, 0.65)',1.2);for(let j=0;j<3;j++)edge(start+base+j,start+base+j+1,'rgba(56, 189, 248, 0.65)',1.2);}}
 }
 for(const m of marks[key(f)]?.marks||[]){ctx.strokeStyle='#d77800';ctx.fillStyle='#d77800';ctx.lineWidth=3*scale;if(m.kind==='box'){ctx.strokeRect(m.xyxy[0],m.xyxy[1],m.xyxy[2]-m.xyxy[0],m.xyxy[3]-m.xyxy[1]);}else{ctx.beginPath();ctx.arc(...m.xy,5*scale,0,Math.PI*2);ctx.stroke();}}
}
function render(){
 $('exhaustive').checked=false;
 const t=times[cursor];$('timeline').value=cursor;$('time').textContent=t.toFixed(2)+' s';$('score').textContent=Number($('threshold').value).toFixed(2);
 for(const v of views){let f=null;for(const candidate of v.frames){if(candidate.frame.captured_at_s>t+1e-8)break;f=candidate;}if(!f){v.frame=null;v.canvas.getContext('2d').clearRect(0,0,v.canvas.width,v.canvas.height);v.meta.textContent='No observation yet';continue;}
   f=selectedFrame(f);
   const alternate=data.variants?.[$('variant').value]?.[f.observation_id];
   const count=alternate?`${alternate.instances.filter(i=>i.label.toLowerCase()==='person').length} person detections (reference)`:`${f.tracks.length} observed person tracks`;
   v.meta.textContent=`Frame ${f.frame.frame_index} · ${count} · ${Math.round((t-f.frame.captured_at_s)*1000)} ms replay offset`;
   if(v.frame?.observation_id!==f.observation_id){v.frame=f;v.canvas.width=f.frame.width;v.canvas.height=f.frame.height;const image=new Image();v.image=image;image.onload=()=>{if(v.image===image)draw(v);};image.src=f.image;}else{v.frame=f;draw(v);}
 }
 const identityList=$('identity-current');if(identityList){identityList.replaceChildren();if($('variant').value){identityList.textContent='Reference detections only. Identity memory belongs to the primary YOLO + RTMW run.';}else for(const v of views){if(!v.frame)continue;const records=v.frame.identity_memory?.length?v.frame.identity_memory:v.frame.tracks.map(track=>({...track,state:track.protection_state||'ACTIVE',observed:true}));for(const record of records){const p=document.createElement('p');p.className='memory-person';p.dataset.state=record.state;
 const title=document.createElement('strong');title.textContent=record.global_id==null?`Unlinked local person ${record.local_id}`:`Person ${record.global_id}`;p.appendChild(title);
 const detail=document.createElement('span');detail.textContent=`${v.frame.frame.camera_id} · ${record.state}${record.observed?' · observed':` · not visible · last seen ${record.age_s?.toFixed(2)??'?'} s before this frame`}`;p.appendChild(detail);
 const track=v.frame.tracks.find(t=>t.local_id===record.local_id);if(track){const status=document.createElement('small');status.textContent=`Local person ${record.local_id} / track ${track.tracker_local_id??track.local_id} · ${track.identity_status}${track.visual_suppressed?' · duplicate display suppressed':''}`;if(track.candidates?.length)status.textContent+=` · candidate Person ${track.candidates[0].global_id}: cosine ${track.candidates[0].cosine.toFixed(2)}`;p.appendChild(status);}identityList.appendChild(p);}
 const audit=v.frame.identity_audit;if(audit&&Object.keys(audit).length){const p=document.createElement('p');p.className='hint';p.textContent=`${v.frame.frame.camera_id}: ${audit.phantom_count} phantoms · ${audit.warm_count} warm memories · ${audit.frozen_local_ids.length} ownership locks`;identityList.appendChild(p);}}}
 const identity=selectedIdentity();
 if(identity){
   const links=identity.links.filter(e=>e.captured_at_s<=t+1e-8),recoveries=(identity.protected_recoveries||[]).filter(e=>e.captured_at_s<=t+1e-8);
   const policy=data.tracking_cases?.[trackingCase()]||(data.tracker_engine==='legacy'?'Original phantom, warm-memory and ownership engine':'Identity memory');
   $('identity-summary').textContent=`${policy}. ${recoveries.length} protected returns · ${links.length} cross-view / appearance revisions by ${t.toFixed(2)} s.`;
   $('identity-links').replaceChildren();
   [...links.map(e=>({...e,kind:'link'})),...recoveries.map(e=>({...e,kind:'recovery'}))].sort((a,b)=>a.captured_at_s-b.captured_at_s).forEach(e=>{const p=document.createElement('p');p.textContent=e.kind==='recovery'?`${e.captured_at_s.toFixed(2)} s: ${e.camera_id}, Person ${e.global_id} returned with its held ID.`:`${e.captured_at_s.toFixed(2)} s: ${e.camera_id}, provisional Person ${e.old_global_id} linked to Person ${e.global_id}; cosine ${e.cosine.toFixed(2)}. Appearance hypothesis.`;$('identity-links').appendChild(p);});
 }
 status();
}
function tick(wall){if(!playing)return;if(lastWall===null)lastWall=wall;playTime+=(wall-lastWall)/1000*Number($('speed').value);lastWall=wall;
 while(cursor<times.length-1&&times[cursor+1]<=playTime)cursor++;
 render();if(cursor===times.length-1){stop();return;}requestAnimationFrame(tick);}
async function init(){
 const response=await fetch('data.json',{cache:'no-store'});if(!response.ok)throw new Error('Cannot load data.json');data=await response.json();if(!data.frames.length)throw new Error('No observations in this run');
 storageKey='selfwatch-review:'+data.run_id;try{marks=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(e){marks={};}
 if(data.initial_review&&!Object.values(marks).some(a=>a.reviewed||a.marks?.length||a.issues?.length)){
   marks={};data.initial_review.frames.forEach(f=>{marks[f.observation_id]={reviewed:f.reviewed,exhaustive_labels:f.exhaustive_labels||[],marks:f.marks,issues:f.issues};});persist();
 }
 times=[...new Set(data.frames.map(f=>f.frame.captured_at_s))].sort((a,b)=>a-b);$('timeline').max=times.length-1;
 $('sample').textContent=`Retail sample ${data.sample} · ${data.frames.length} observations`;$('provider').textContent=`Whole-body runtime: ${data.pose.providers.join(', ')}.`;
 if(data.identity){$('identity-note').textContent='Person IDs are shared across views when appearance evidence supports a link. Track numbers are local debug IDs. Lost people stay in memory; phantom predictions are not drawn as visible people. Initial visit IDs can be revised by a logged cross-camera association. These associations still need ground-truth validation.';}
 if(data.tracking_cases){
   $('tracking-case-label').hidden=false;$('study-note').hidden=false;
   for(const [key,name] of Object.entries(data.tracking_cases)){const option=document.createElement('option');option.value=key;option.textContent=name;$('tracking-case').appendChild(option);}
   $('tracking-case').value='legacy';$('tracking-case').onchange=()=>{stop();render();};
 }
 if(data.study_navigation){
   $('study-navigation').hidden=false;
   const home=document.createElement('a');home.href='../';home.textContent='All study results';$('study-navigation').appendChild(home);
   for(const item of data.study_navigation){const a=document.createElement('a');a.href=item.href;a.textContent=`Sample ${item.sample}`;if(item.sample===data.sample)a.setAttribute('aria-current','page');$('study-navigation').appendChild(a);}
 }
 Object.keys(data.variants||{}).forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;$('variant').appendChild(option);});$('variant').onchange=render;
 data.cameras.forEach((camera,index)=>{
   const panel=document.createElement('article');panel.className='camera';panel.innerHTML='<h2><span></span><small>Recorded view</small></h2><canvas aria-label="Camera frame and annotation canvas"></canvas><div class="frame-meta"></div>';
   panel.querySelector('h2 span').textContent=camera;$('cameras').appendChild(panel);
   const v={canvas:panel.querySelector('canvas'),meta:panel.querySelector('.frame-meta'),frames:data.frames.filter(f=>f.frame.camera_id===camera),frame:null,image:new Image()};views.push(v);
   const option=document.createElement('option');option.value=index;option.textContent=camera;$('active').appendChild(option);
   v.canvas.addEventListener('pointerdown',e=>{if($('mark').value==='inspect'||!v.frame)return;stop();$('active').value=index;$('exhaustive').checked=false;v.start=point(e,v.canvas);v.startFrame=v.frame;v.canvas.setPointerCapture(e.pointerId);});
   v.canvas.addEventListener('pointerup',e=>{if(!v.start||v.startFrame!==v.frame)return;const start=v.start,end=point(e,v.canvas),label=$('mark').value;v.start=null;if(label==='inspect')return;
     const mark={label,subject:$('subject').value.trim(),visibility:'visible',source:'manual',created_at:new Date().toISOString()};
     if(label.endsWith('wrist')){mark.kind='point';mark.xy=end;}else{if(Math.abs(start[0]-end[0])<4||Math.abs(start[1]-end[1])<4)return;mark.kind='box';mark.xyxy=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];}
     const annotation=entry(v.frame);annotation.marks.push(mark);annotation.reviewed=false;annotation.exhaustive_labels=[];persist();draw(v);status();});
   v.canvas.addEventListener('pointercancel',()=>v.start=null);
 });
 $('play').onclick=()=>{if(playing){stop();return;}if(cursor===times.length-1)cursor=0;playing=true;playTime=times[cursor];lastWall=null;$('play').textContent='Pause';$('play').setAttribute('aria-label','Pause recording');requestAnimationFrame(tick);};
 $('timeline').oninput=()=>{stop();cursor=Number($('timeline').value);render();};
 $('previous').onclick=()=>{stop();cursor=Math.max(0,cursor-1);render();};$('next').onclick=()=>{stop();cursor=Math.min(times.length-1,cursor+1);render();};
 for(const id of ['boxes','body','hands','face','masks','threshold'])$(id).oninput=render;
 $('active').onchange=()=>{$('exhaustive').checked=false;status();};
 $('reviewed').onclick=()=>{stop();const v=currentView();if(v?.frame){const annotation=entry(v.frame);annotation.reviewed=!annotation.reviewed;annotation.exhaustive_labels=annotation.reviewed&&$('exhaustive').checked?['person']:[];persist();status();}};
 $('flag').onclick=()=>{stop();const v=currentView();if(v?.frame&&$('issue').value!=='none'){entry(v.frame).issues.push($('issue').value);persist();status();notify('Observation added to active frame.');}};
 $('undo').onclick=()=>{const v=currentView();if(v?.frame){const annotation=entry(v.frame);annotation.marks.pop();annotation.reviewed=false;annotation.exhaustive_labels=[];persist();draw(v);status();}};
 $('export').onclick=()=>{const result={schema_version:1,run_id:data.run_id,sample:data.sample,annotation_source:data.initial_review?data.initial_review.annotation_source+'; interactive edits retained in individual marks':'manual review of anonymized footage',coordinate_space:'original_image_pixels',frames:data.frames.map(f=>({observation_id:f.observation_id,frame:f.frame,...(marks[key(f)]||{reviewed:false,marks:[],issues:[]})}))};const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`selfwatch-${data.sample}-review.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Review exported. Unreviewed frames are explicitly retained.');};
 const events=$('events');data.frames.forEach(f=>{const button=document.createElement('button');button.textContent=`${f.frame.captured_at_s.toFixed(2)} s / ${f.frame.camera_id} / frame ${f.frame.frame_index} / ${f.tracks.length} person tracks`;button.onclick=()=>{stop();cursor=times.indexOf(f.frame.captured_at_s);$('active').value=data.cameras.indexOf(f.frame.camera_id);render();};events.appendChild(button);});
 render();notify(data.initial_review?'Loaded six preliminary assistant person-box labels. Human adjudication is still needed.':'Ready. No frames have been automatically marked reviewed.');
}
init().catch(e=>notify('Could not open this recording: '+e.message+'. Serve this folder with the local review launcher.'));
