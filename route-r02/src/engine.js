/* SHUIMO ROUTE R02 — deterministic, prepainted 2.5D scenery.
 * No dependencies, remote services, per-frame painting or hidden fog triggers.
 * Camera position is constrained to an arc-length-parameterized authored trail.
 * All mountain UVs and silhouettes remain on the same world-space proxy.
 */
'use strict';
(() => {
const DATA = JSON.parse(document.getElementById('scene-data').textContent);
const S = DATA.scene, images = DATA.images;
const $ = id => document.getElementById(id);
const canvas = $('scene'), overlay = $('overlay'), overlayCtx = overlay.getContext('2d');
const clamp = (x,a,b)=>Math.min(b,Math.max(a,x));
const mix=(a,b,t)=>a+(b-a)*t, dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const smooth=(a,b,x)=>{const t=clamp((x-a)/(b-a),0,1);return t*t*(3-2*t);};
const rad=x=>x*Math.PI/180;
const state={s:0, yaw:0, pitch:0, playing:false, direction:1, speed:4,
  fog:false, mode:'relief', geometry:false, region:false, appearance:'auto',
  edge:'ink', compare:true, inspect:'paint', solo:false, backdrop:false};
let gl,program,loc={},mountains=[],fogs=[],ready=false,dirty=true,lastTime=0,draws=0;
let renderCount=0,uiFps=0,fpsTime=0,framesSince=0;
let rw=0,rh=0,ratio=1;
const rawTrail=u=>[-65+145*u+18*Math.sin(2*Math.PI*u),5+2.2*Math.sin(Math.PI*u)+.6*Math.sin(3*Math.PI*u),-20+110*u+14*Math.sin(Math.PI*u)];
const trail=[];let routeLength=0;
for(let i=0;i<=600;i++){
  const u=i/600,p=rawTrail(u);
  if(i)routeLength+=Math.hypot(...p.map((v,j)=>v-trail[i-1].p[j]));
  trail.push({u,p,d:routeLength});
}
function at(s){
  const d=clamp(s,0,1)*routeLength;let lo=0,hi=600;
  while(hi-lo>1){const k=(lo+hi)>>1;if(trail[k].d<d)lo=k;else hi=k;}
  const a=trail[lo],b=trail[hi],t=(d-a.d)/Math.max(b.d-a.d,1e-9);
  return {u:mix(a.u,b.u,t),p:a.p.map((v,i)=>mix(v,b.p[i],t))};
}
function camera(s=state.s,yaw=state.yaw,pitch=state.pitch){
  const t=at(s),y=-.03+.08*Math.sin(Math.PI*t.u)-.02*t.u+rad(yaw);
  const p=.125+.005*Math.sin(Math.PI*t.u)+rad(pitch);
  return {pos:t.p, right:[Math.cos(y),0,-Math.sin(y)],
    up:[-Math.sin(y)*Math.sin(p),Math.cos(p),-Math.cos(y)*Math.sin(p)],
    forward:[Math.sin(y)*Math.cos(p),Math.sin(p),Math.cos(y)*Math.cos(p)],yaw:y,pitch:p};
}
const refPos=at(0).p;
function world(p,u,v){
  const r=p.relief||0, vp=clamp(v/(p.sourceV||1),0,1);
  const carrierU=u;u=clamp(u,0,1);
  const bulge=Math.sin(Math.PI*u)*Math.sin(Math.PI*vp)*(.76+.24*Math.sin(3*Math.PI*u));
  return [p.x+(carrierU-.5)*p.w,p.base+(1-v)*p.h,p.z-r*bulge-.18*r*Math.sin(5*Math.PI*u)*Math.sin(Math.PI*vp)];
}
function flat(p,P){
  const k=(p.z-refPos[2])/(P[2]-refPos[2]);
  return [refPos[0]+(P[0]-refPos[0])*k,refPos[1]+(P[1]-refPos[1])*k,p.z];
}
function project(P,c=camera(),w=rw,h=rh){
  const d=P.map((v,i)=>v-c.pos[i]),z=dot(d,c.forward),f=1/Math.tan(rad(S.route.vertical_fov_deg)/2);
  return [w*.5+dot(d,c.right)*h*.5*f/z,h*.5-dot(d,c.up)*h*.5*f/z,z];
}
function inside(u,v,poly){
 let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){
  const a=poly[i],b=poly[j];if((a[1]>v)!=(b[1]>v)&&u<(b[0]-a[0])*(v-a[1])/(b[1]-a[1])+a[0])c=!c;
 }return c;
}
const VS=`attribute vec3 aPosition; attribute vec3 aFlat; attribute vec2 aUV;
uniform vec3 uCamera,uRight,uUp,uForward; uniform float uFlat,uF,uAspect;
varying vec2 vUV;
void main(){vec3 d=mix(aPosition,aFlat,uFlat)-uCamera;float z=dot(d,uForward);
gl_Position=vec4(dot(d,uRight)*uF/uAspect,dot(d,uUp)*uF,1.0000400008*z-.4000080002,z);vUV=aUV;}`;
const FS=`precision mediump float;varying vec2 vUV;uniform sampler2D uA,uB,uFields;
uniform float uMix,uAlpha,uRegion,uSolid,uInspect;uniform vec4 uColor;
void main(){if(uSolid>.5){gl_FragColor=vec4(uColor.rgb*uColor.a,uColor.a);return;}
vec4 a=texture2D(uA,vUV),b=texture2D(uB,vUV);vec4 t=mix(a,b,uMix);
if(uInspect>.5){
 vec4 fields=texture2D(uFields,vUV);
 float v=uInspect<1.5?fields.r:uInspect<2.5?fields.g:t.a;
 gl_FragColor=vec4(vec3(v),1.);return;
}
// All textures are premultiplied BEFORE interpolation/mipmap generation.
// No .02 alpha discard: faint original ink is part of the visible silhouette.
if(t.a<=0.)discard;
float delta=length(a.rgb-b.rgb);
t.rgb=mix(t.rgb,vec3(.70,.27,.15)*t.a,uRegion*step(.004,delta)*.48);
gl_FragColor=t*uAlpha;}`;
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
 if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function initializeGL(){
 gl=canvas.getContext('webgl',{alpha:false,antialias:true,preserveDrawingBuffer:true,powerPreference:'low-power'});
 if(!gl)throw Error('此浏览器未启用 WebGL。请用开启硬件加速的浏览器打开；也可先看随包的 MP4 录像。');
 program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,VS));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,FS));gl.linkProgram(program);
 if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
 for(const a of ['aPosition','aFlat','aUV'])loc[a]=gl.getAttribLocation(program,a);
 for(const u of ['uCamera','uRight','uUp','uForward','uFlat','uF','uAspect','uA','uB','uMix','uAlpha','uRegion','uSolid','uColor','uFields','uInspect'])loc[u]=gl.getUniformLocation(program,u);
 gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
 gl.disable(gl.CULL_FACE);gl.uniform1i(loc.uA,0);gl.uniform1i(loc.uB,1);gl.uniform1i(loc.uFields,2);
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();state.playing=false;$('error').hidden=false;$('error').textContent='图形上下文已丢失。请重新打开页面；原始资产和进度文件不受影响。';});
}
function buffer(target,array){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,array,gl.STATIC_DRAW);return b;}
const texCache=new Map();
async function texture(name){
 if(texCache.has(name))return texCache.get(name);
 const img=new Image();img.src=images[name];await img.decode();
 const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,true);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
 gl.generateMipmap(gl.TEXTURE_2D);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);
 texCache.set(name,t);return t;
}
async function mesh(p,isFog=false){
 const [nx,ny]=isFog?[1,1]:p.grid;const vert=[],ind=[],lines=[],samples=[];
 for(let j=0;j<=ny;j++)for(let i=0;i<=nx;i++){
  const domain=p.domain||[0,1,0,1],tu=i/nx,tv=j/ny;
  const u=mix(domain[0],domain[1],tu),v=mix(domain[2],domain[3],tv);
  const P=world(p,u,v),Q=flat(p,P);vert.push(...P,...Q,tu,tv);
  if(!isFog&&u>=0&&u<=1&&v>=0&&v<=(p.sourceV||1)&&inside(u,v,p.contour))samples.push({P,Q,uv:[u,v]});
 }
 for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
  const a=j*(nx+1)+i,b=a+1,c=a+nx+1,d=c+1;ind.push(a,c,b,b,c,d);
  if(i%4===0&&j%4===0)lines.push(a,b,a,c,b,d,c,d);
 }
 const variants={};
 if(!isFog)for(const [k,v] of Object.entries(p.edgeVariants))variants[k]={ta:await texture(v.A),tb:await texture(v.B)};
 return {p,isFog,variants,fields:await texture(isFog?'fog.png':p.fieldTexture),vbo:buffer(gl.ARRAY_BUFFER,new Float32Array(vert)),ibo:buffer(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(ind)),lbo:buffer(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(lines)),n:ind.length,ln:lines.length,samples,
  ta:await texture(isFog?'fog.png':p.textureA),tb:await texture(isFog?'fog.png':p.textureB)};
}
function weight(s=state.s,appearance=state.appearance){
 if(appearance==='A')return 0;if(appearance==='B')return 1;
 return smooth(S.patches.find(p=>p.id==='M02').transition[0],S.patches.find(p=>p.id==='M02').transition[1],s);
}
function bind(m,edge=state.edge){
 gl.bindBuffer(gl.ARRAY_BUFFER,m.vbo);
 for(const [name,size,offset] of [['aPosition',3,0],['aFlat',3,12],['aUV',2,24]]){
  gl.enableVertexAttribArray(loc[name]);gl.vertexAttribPointer(loc[name],size,gl.FLOAT,false,32,offset);
 }
 const t=m.isFog?m:m.variants[edge];
 gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,t.ta);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,t.tb);
 gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,m.fields);
}
function cameraUniforms(c,flatMode){
 gl.uniform3fv(loc.uCamera,c.pos);gl.uniform3fv(loc.uRight,c.right);gl.uniform3fv(loc.uUp,c.up);gl.uniform3fv(loc.uForward,c.forward);
 gl.uniform1f(loc.uF,1/Math.tan(rad(S.route.vertical_fov_deg)/2));gl.uniform1f(loc.uAspect,rw/rh);gl.uniform1f(loc.uFlat,flatMode?1:0);
}
function drawMesh(m,flatMode,line=false,edge=state.edge){
 bind(m,edge);
 gl.uniform1f(loc.uInspect,m.isFog?0:({paint:0,coverage:1,ink:2,alpha:3}[state.inspect]||0));gl.uniform1f(loc.uFlat,flatMode?1:0);
 gl.uniform1f(loc.uMix,m.p.id==='M02'?weight():0);gl.uniform1f(loc.uAlpha,m.isFog?m.p.opacity:1);
 gl.uniform1f(loc.uRegion,state.region&&m.p.id==='M02'?1:0);gl.uniform1f(loc.uSolid,line?1:0);
 if(line){gl.uniform4f(loc.uColor,.37,.42,.31,.20);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.lbo);gl.drawElements(gl.LINES,m.ln,gl.UNSIGNED_SHORT,0);}
 else{gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.ibo);gl.drawElements(gl.TRIANGLES,m.n,gl.UNSIGNED_SHORT,0);}
 draws++;
}
function drawWorld(c,flatMode,edge=state.edge){
 cameraUniforms(c,flatMode);
 // This authored scene has nonintersecting depth slabs. Sort all scenery AND
 // fog together. Transparent margins never write depth or block later fog.
 gl.depthMask(false);gl.disable(gl.DEPTH_TEST);
 let layers=[...mountains];
 if(state.solo||state.inspect!=='paint')layers=layers.filter(m=>m.p.id==='M02');
 if(state.fog&&state.inspect==='paint'&&!state.solo)layers.push(...fogs);
 layers.sort((a,b)=>b.p.z-a.p.z);
 for(const m of layers)drawMesh(m,flatMode,false,edge);
 if(state.geometry){for(const m of layers)if(!m.isFog)drawMesh(m,flatMode,true,edge);}
 gl.depthMask(true);gl.enable(gl.DEPTH_TEST);
}
function resize(){
 const r=canvas.getBoundingClientRect();ratio=Math.min(window.devicePixelRatio||1,1.5);
 const w=Math.max(2,Math.round(r.width*ratio)),h=Math.max(2,Math.round(r.height*ratio));
 if(w!==rw||h!==rh){rw=w;rh=h;canvas.width=w;canvas.height=h;overlay.width=w;overlay.height=h;dirty=true;}
}
function render(){
 if(!ready)return;resize();draws=0;gl.viewport(0,0,rw,rh);gl.disable(gl.SCISSOR_TEST);gl.depthMask(true);
 state.backdrop?gl.clearColor(.72,.15,.57,1):gl.clearColor(...S.paper,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 const c=camera();
 if(state.compare){
  gl.enable(gl.SCISSOR_TEST);gl.scissor(0,0,Math.floor(rw/2),rh);drawWorld(c,state.mode==='flat','legacy');
  gl.scissor(Math.floor(rw/2),0,rw-Math.floor(rw/2),rh);drawWorld(c,state.mode==='flat',state.edge);gl.disable(gl.SCISSOR_TEST);
 }else if(state.mode==='split'){
  gl.enable(gl.SCISSOR_TEST);gl.scissor(0,0,Math.floor(rw/2),rh);drawWorld(c,false);
  gl.scissor(Math.floor(rw/2),0,rw-Math.floor(rw/2),rh);gl.clear(gl.DEPTH_BUFFER_BIT);drawWorld(c,true);gl.disable(gl.SCISSOR_TEST);
 }else drawWorld(c,state.mode==='flat');
 annotations(c);renderCount++;dirty=false;updateUI();
}
function annotations(c){
 const ctx=overlayCtx;ctx.clearRect(0,0,rw,rh);ctx.save();ctx.scale(ratio,ratio);const w=rw/ratio,h=rh/ratio;
 if(state.compare||state.mode==='split'){
  ctx.strokeStyle='rgba(77,78,66,.38)';ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke();
  for(const [x,label] of (state.compare?[[24,'旧：手工山形裁切'],[w/2+18,edgeLabels[state.edge]]]:[[24,'起伏景片'],[w/2+18,'平面景片 · 同起点对齐']])){
   ctx.font='12px system-ui';ctx.fillStyle='rgba(245,242,232,.9)';ctx.fillRect(x-8,18,ctx.measureText(label).width+16,28);ctx.fillStyle='#464f46';ctx.fillText(label,x,37);
  }
 }
 if(state.geometry){
  for(const m of mountains){
   const p=m.p,P=world(p,...p.peak_uv),point=project(state.mode==='flat'?flat(p,P):P,c);const [x,y]=point.map(v=>v/ratio);
   if(x<12||x>w-12||y<12||y>h-12)continue;
   ctx.strokeStyle='#9d5e44';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(x,y-6);ctx.lineTo(x,y-18);ctx.stroke();
   ctx.font='11px system-ui';const text=`${p.id} ${p.name}`;const tw=ctx.measureText(text).width;
   ctx.fillStyle='rgba(245,242,232,.92)';ctx.fillRect(x-tw/2-5,y-35,tw+10,17);ctx.fillStyle='#685548';ctx.fillText(text,x-tw/2,y-23);
  }
 }
 ctx.restore();
}
function residuals(s=state.s,yaw=state.yaw,pitch=state.pitch){
 const c=camera(s,yaw,pitch),rows=[];
 for(const m of mountains){
  const e=[];for(const {P,Q} of m.samples){const a=project(P,c),b=project(Q,c);if(a[2]>1&&a[0]>=0&&a[0]<rw&&a[1]>=0&&a[1]<rh)e.push(Math.hypot(a[0]-b[0],a[1]-b[1])/ratio);}
  e.sort((a,b)=>a-b);rows.push({id:m.p.id,name:m.p.name,p95:e.length?e[Math.floor((e.length-1)*.95)]:0,max:e.length?e[e.length-1]:0,n:e.length});
 }
 return rows;
}
function map(){
 const c=$('map').getContext('2d'),w=240,h=140;c.clearRect(0,0,w,h);
 const xy=P=>[24+(P[0]+65)/150*185,116-(P[2]+20)/115*91];
 c.lineWidth=1;c.strokeStyle='#dedfd3';for(let i=0;i<4;i++){c.beginPath();c.moveTo(20,23+i*30);c.lineTo(220,23+i*30);c.stroke();}
 c.lineWidth=2;c.strokeStyle='#c4cbbc';c.beginPath();trail.forEach((t,i)=>{const [x,y]=xy(t.p);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
 c.strokeStyle='#4f6658';c.beginPath();trail.filter(t=>t.d<=state.s*routeLength).forEach((t,i)=>{const [x,y]=xy(t.p);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
 const co=camera(),[x,y]=xy(co.pos);c.fillStyle='#a56b50';c.beginPath();c.arc(x,y,4.5,0,Math.PI*2);c.fill();c.strokeStyle='#a56b50';c.beginPath();c.moveTo(x,y);c.lineTo(x+Math.sin(co.yaw)*17,y-Math.cos(co.yaw)*17);c.stroke();
 c.font='10px system-ui';c.fillStyle='#7a8276';c.fillText('起点',16,136);c.fillText('终点',205,18);
}
const edgeLabels={legacy:'旧：手工山形裁切',feather:'对照：仅羽化旧边界',ink:'新：原墨迹外缘＋内部遮挡'};
let lastMetrics=-1;
function updateUI(force=false){
 $('play').textContent=state.playing?'暂停':'开始行走';$('reverse').textContent=state.direction<0?'改为前进':'改为倒退';$('reverse').classList.toggle('selected',state.direction<0);$('reverse').setAttribute('aria-pressed',String(state.direction<0));
 $('route').value=Math.round(state.s*1000);$('distance').textContent=(state.s*routeLength).toFixed(1);
 $('total').textContent=routeLength.toFixed(0);$('progress').textContent=(state.s*100).toFixed(0)+'%';
 $('yaw-value').textContent=(state.yaw>0?'+':'')+state.yaw.toFixed(0)+'°';$('yaw').value=state.yaw;
 const w=weight();$('mix-value').textContent=(w*100).toFixed(0)+'%';$('mix-bar').style.width=(w*100)+'%';
 $('phase').textContent=state.appearance!=='auto'?'锁定缓存 '+state.appearance:w<=0?'复用 A':w>=1?'复用 B':'局部交接 A → B';
 $('phase-description').textContent=w>0&&w<1?'只替换主峰右下坡；轮廓、山脊与其他山体继续复用。':'当前画迹保持不变；每帧只更新相机投影与遮挡。';
 $('mode-label').textContent=state.mode==='relief'?'起伏景片':state.mode==='flat'?'平面景片对照':'左右对照';
 $('edge-label').textContent=state.compare?'旧 / 新边缘对照':edgeLabels[state.edge];
 $('fog-label').textContent=state.fog?'有雾':'无雾';$('step-label').textContent=state.playing?(state.direction>0?'沿路行走':'沿路倒退'):'停步观察';
 $('paint-calls').textContent='0';$('texture-count').textContent=texCache.size;
 $('triangles').textContent=mountains.reduce((a,m)=>a+m.n/3,0).toLocaleString();
 if(force||Math.abs(state.s-lastMetrics)>.009){const rows=residuals();$('error-value').textContent=Math.max(...rows.map(r=>r.p95)).toFixed(1);lastMetrics=state.s;}
 map();
}
function syncControls(){
 $('fog').checked=state.fog;$('geometry').checked=state.geometry;$('region').checked=state.region;
 $('edge').value=state.edge;$('inspect').value=state.inspect;
 for(const k of ['compare','solo','backdrop'])$(k).checked=state[k];
 $('mode').value=state.mode;$('appearance').value=state.appearance;$('speed').value=String(state.speed);
}
function change(values,immediate=true){
 for(const [k,v] of Object.entries(values)){
  if(!(k in state))throw Error('Unknown state key: '+k);
  if(typeof state[k]==='number'&&!Number.isFinite(Number(v)))throw Error(k+' must be finite');
  state[k]=v;
 }
 state.s=clamp(Number(state.s),0,1);state.yaw=clamp(Number(state.yaw),-16,16);state.pitch=clamp(Number(state.pitch),-7,7);
 if(!['flat','relief','split'].includes(state.mode))throw Error('Unknown projection mode');
 if(!['legacy','feather','ink'].includes(state.edge))throw Error('Unknown edge mode');
 if(!['paint','coverage','ink','alpha'].includes(state.inspect))throw Error('Unknown field inspection');
 if(!['auto','A','B'].includes(state.appearance))throw Error('Unknown appearance mode');
 dirty=true;lastMetrics=-1;syncControls();if(immediate)render();
}
function tick(time){
 const dt=lastTime?Math.min((time-lastTime)/1000,.05):0;lastTime=time;
 if(ready&&state.playing){
  state.s+=state.direction*state.speed*dt/routeLength;
  if(state.s>=1||state.s<=0){state.s=clamp(state.s,0,1);state.playing=false;}
  dirty=true;
 }
 if(dirty&&ready){render();framesSince++;}
 if(time-fpsTime>1500){uiFps=framesSince*1000/(time-fpsTime);framesSince=0;fpsTime=time;}
 requestAnimationFrame(tick);
}
function setupEvents(){
 $('play').onclick=()=>{if(state.s===1&&state.direction>0)state.direction=-1;if(state.s===0&&state.direction<0)state.direction=1;change({playing:!state.playing});};
 $('reverse').onclick=()=>{const d=-state.direction;const canMove=!((state.s===0&&d<0)||(state.s===1&&d>0));change({direction:d,playing:canMove});};
 $('reset').onclick=()=>change({s:0,playing:false,direction:1,yaw:0,pitch:0});
 $('route').oninput=e=>change({s:Number(e.target.value)/1000,playing:false});
 $('speed').onchange=e=>change({speed:Number(e.target.value)});
 $('yaw').oninput=e=>change({yaw:Number(e.target.value)});
 $('look-reset').onclick=()=>change({yaw:0,pitch:0});
 for(const k of ['fog','geometry','region','compare','solo','backdrop'])$(k).onchange=e=>change({[k]:e.target.checked});
 for(const k of ['mode','appearance','edge','inspect'])$(k).onchange=e=>change({[k]:e.target.value,...(k==='mode'&&e.target.value==='split'?{compare:false}:{})});
 $('full').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('stage').requestFullscreen();}catch(e){$('status-note').textContent='当前容器不允许全屏；页面内操作不受影响。';}};
 $('snapshot').onclick=()=>{render();const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=`shuimo_route_r02_${(state.s*100).toFixed(1)}.png`;a.click();};
 let drag=null;
 canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,yaw:state.yaw,pitch:state.pitch};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');});
 canvas.addEventListener('pointermove',e=>{if(drag)change({yaw:drag.yaw-(e.clientX-drag.x)*.07,pitch:drag.pitch+(e.clientY-drag.y)*.045});});
 const release=()=>{drag=null;canvas.classList.remove('dragging');};canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
 window.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('play').click();}
  if(e.code==='ArrowUp'||e.code==='KeyW'){e.preventDefault();change({s:state.s+.8/routeLength,playing:false});}
  if(e.code==='ArrowDown'||e.code==='KeyS'){e.preventDefault();change({s:state.s-.8/routeLength,playing:false});}
  if(e.code==='ArrowLeft')change({yaw:state.yaw-2});if(e.code==='ArrowRight')change({yaw:state.yaw+2});
 });
 new ResizeObserver(()=>{dirty=true;lastMetrics=-1;}).observe($('stage'));
 document.addEventListener('visibilitychange',()=>{lastTime=0;});
}
function pixelHash(){
 const a=new Uint8Array(rw*rh*4);gl.readPixels(0,0,rw,rh,gl.RGBA,gl.UNSIGNED_BYTE,a);
 let h=2166136261;for(let i=0;i<a.length;i++){h^=a[i];h=Math.imul(h,16777619);}return (h>>>0).toString(16).padStart(8,'0');
}
function coverageProbe(uv){
 const m=mountains.find(x=>x.p.id==='M02'),c=camera(.22,0,0);gl.disable(gl.SCISSOR_TEST);gl.depthMask(true);gl.clearColor(1,0,1,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 cameraUniforms(c,false);gl.disable(gl.DEPTH_TEST);gl.depthMask(false);drawMesh(m,false);gl.depthMask(true);gl.enable(gl.DEPTH_TEST);
 const P=world(m.p,uv[0],uv[1]),screen=project(P,c);const pix=new Uint8Array(4);
 gl.readPixels(Math.round(screen[0]),rh-1-Math.round(screen[1]),1,1,gl.RGBA,gl.UNSIGNED_BYTE,pix);
 render();return {uv,screen,rgba:[...pix],pass:pix[1]>185&&Math.abs(pix[0]-pix[2])<35};
}
function diagnosticProbe(id,uv,bg=[1,0,1]){
 const m=mountains.find(x=>x.p.id===id),c=camera();
 const screen=project(world(m.p,...uv),c);
 if(screen[0]<0||screen[0]>=rw||screen[1]<0||screen[1]>=rh)return {uv,screen,visible:false};
 gl.disable(gl.SCISSOR_TEST);gl.disable(gl.DEPTH_TEST);gl.depthMask(true);
 gl.clearColor(...bg,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.depthMask(false);
 cameraUniforms(c,false);drawMesh(m,false);
 const px=new Uint8Array(4);
 gl.readPixels(Math.floor(screen[0]),rh-1-Math.floor(screen[1]),1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);
 gl.depthMask(true);gl.enable(gl.DEPTH_TEST);render();
 return {uv,screen,visible:true,rgba:[...px]};
}
async function init(){
 try{
  initializeGL();
  // Sequential awaited uploads avoid a transient visible frame with missing layers.
  for(const p of S.patches)mountains.push(await mesh(p));
  for(const p of S.fog)fogs.push(await mesh(p,true));
  mountains.sort((a,b)=>b.p.z-a.p.z);fogs.sort((a,b)=>b.p.z-a.p.z);
  ready=true;setupEvents();syncControls();render();$('loading').hidden=true;
  window.__inkDemo={ready:true,setState:change,getState:()=>({...state}),render,
   getInfo:()=>({version:S.version,edgeMode:state.edge,alphaPipeline:'premultiplied; sorted nonintersecting slabs; transparent depth writes disabled',width:rw,height:rh,cssWidth:rw/ratio,cssHeight:rh/ratio,routeLength,textureCount:texCache.size,triangles:mountains.reduce((a,m)=>a+m.n/3,0),draws,renderCount,renderer:gl.getParameter(gl.RENDERER),webgl:gl.getParameter(gl.VERSION),limits:S.limits}),
   getMetrics:residuals,weight,capture:()=>{render();return canvas.toDataURL('image/png');},pixelHash,
   getError:()=>gl.getError(),coverageProbe,diagnosticProbe,
   getScene:()=>S,
   layerOrder:()=>[...mountains,...(state.fog?fogs:[])].sort((a,b)=>b.p.z-a.p.z).map(m=>({id:m.p.id,z:m.p.z})),
   screenAt:(id,u,v)=>{const p=S.patches.find(p=>p.id===id);return project(world(p,u,v));},
   projectPeak:(id,s,yaw=0,pitch=0)=>{const p=S.patches.find(p=>p.id===id);return project(world(p,...p.peak_uv),camera(s,yaw,pitch));},
   referenceResidual:()=>{let max=0;const c=camera(0,0,0);for(const m of mountains)for(const {P,Q}of m.samples){const a=project(P,c),b=project(Q,c);max=Math.max(max,Math.hypot(a[0]-b[0],a[1]-b[1]));}return max;}
  };
  requestAnimationFrame(tick);
 }catch(e){console.error(e);$('loading').hidden=true;$('error').hidden=false;$('error').textContent=e.message;window.__inkDemo={ready:false,error:e.message};}
}
init();
})();
