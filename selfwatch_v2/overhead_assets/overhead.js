'use strict';
const $=id=>document.getElementById(id);
let data,frames=[],index=0,renderToken=0;
const pct=value=>(100*value).toFixed(1)+'%';
const number=value=>value.toFixed(1);
function cell(row,text){const td=document.createElement('td');td.textContent=text;row.append(td);}
function metrics(){
  $('metrics').replaceChildren();
  for(const [name,s] of Object.entries(data.metrics[$('partition').value])){
    const row=document.createElement('tr');row.dataset.selected=name===$('candidate').value;
    for(const value of [data.cases[name],s.true_positive,s.false_positive,s.false_negative,pct(s.precision),pct(s.recall),s.F1.toFixed(3),number(s.mean_ms),number(s.p95_ms)])cell(row,value);
    $('metrics').append(row);
  }
}
function filter(rebuildScenes){
  const role=$('partition').value;
  if(rebuildScenes){
    $('scene').replaceChildren(new Option('All scenes','all'));
    for(const scene of [...new Set(data.frames.filter(f=>f.partition===role).map(f=>f.sequence))])$('scene').add(new Option(scene.replaceAll('_',' '),scene));
  }
  frames=data.frames.filter(f=>f.partition===role&&($('scene').value==='all'||f.sequence===$('scene').value));
  index=0;$('image').max=Math.max(0,frames.length-1);metrics();render();
}
function draw(canvas,image,frame,name){
  canvas.width=frame.width;canvas.height=frame.height;canvas.style.aspectRatio=`${frame.width}/${frame.height}`;
  const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
  const scale=Math.max(frame.width/650,1),result=frame.cases[name];
  ctx.lineWidth=2*scale;ctx.font=`600 ${12*scale}px Segoe UI`;
  function box(b,label,color,dashed){
    const [x1,y1,x2,y2]=b;ctx.strokeStyle=color;ctx.setLineDash(dashed?[6*scale,4*scale]:[]);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
    const w=ctx.measureText(label).width+8*scale,h=17*scale,x=Math.min(Math.max(x1,0),canvas.width-w),y=Math.max(h,y1);
    ctx.fillStyle='#173044';ctx.fillRect(x,y-h,w,h);ctx.fillStyle=color;ctx.fillText(label,x+4*scale,y-4*scale);
  }
  if($('gt').checked)frame.ground_truth.forEach((g,i)=>box(g.box_xyxy,`Annotation ${i+1}`,'#ffe15a',true));
  if($('pred').checked){const matched=new Set(result.matches.map(m=>m.prediction));result.predictions.forEach((p,i)=>box(p.box_xyxy,`${matched.has(i)?'Match':'Extra'} ${Math.round(100*p.score)}%`,matched.has(i)?'#61b5ff':'#ff8ecb',false));}
}
async function render(){
  const token=++renderToken,frame=frames[index];if(!frame)return;
  $('image').value=index;$('position').textContent=`${index+1} / ${frames.length}`;
  $('previous').disabled=index===0;$('next').disabled=index===frames.length-1;
  $('frame-title').textContent=`${frame.sequence.replaceAll('_',' ')} / source image ${frame.image_id} / ${frame.ground_truth.length} annotated people`;
  const selected=$('candidate').value;$('candidate-title').textContent=data.cases[selected];
  for(const [name,target] of [['native_640','baseline-result'],[selected,'candidate-result']]){
    const s=frame.cases[name];$(target).textContent=`${s.true_positive} matched / ${s.false_negative} missed / ${s.false_positive} extra · ${number(s.latency_ms)} ms · ${s.passes} pass${s.passes===1?'':'es'}`;
  }
  const image=new Image();image.src=frame.image;
  try{await image.decode();if(token!==renderToken)return;draw($('baseline'),image,frame,'native_640');draw($('alternative'),image,frame,selected);}
  catch(error){if(token===renderToken)$('status').textContent='Image unavailable. Check that this run’s images folder is present.';}
}
async function init(){
  try{
    const response=await fetch('data.json',{cache:'no-store'});if(!response.ok)throw Error('Result file not found');data=await response.json();
    for(const [name,label] of Object.entries(data.cases))$('candidate').add(new Option(label,name));
    $('candidate').value=data.choice.quality_winner;
    $('status').textContent=`${data.metrics.development.native_640.images} development + ${data.metrics.validation.native_640.images} validation images. Forty final-test images are excluded from inference.`;
    $('selection').textContent=`Selected on development scenes before validation: ${data.cases[data.choice.quality_winner]}. ${data.choice.budget_satisfied?'A candidate met':'No candidate met'} the ${data.choice.detector_budget_ms} ms detector P95 budget on this machine.`;
    $('protocol').textContent=`Confidence ${data.protocol.confidence}; IoU ${data.protocol.iou}. ${data.protocol.input}. ${data.protocol.box_rule}.`;
    $('partition').onchange=()=>filter(true);$('scene').onchange=()=>filter(false);
    $('candidate').onchange=()=>{metrics();render();};$('gt').onchange=render;$('pred').onchange=render;
    $('image').oninput=()=>{index=Number($('image').value);render();};
    $('previous').onclick=()=>{if(index>0){index--;render();}};$('next').onclick=()=>{if(index<frames.length-1){index++;render();}};
    $('workspace').hidden=false;filter(true);
  }catch(error){$('status').textContent=`Cannot load this comparison: ${error.message}. Serve the completed run folder using scripts/open-review.ps1.`;}
}
init();
