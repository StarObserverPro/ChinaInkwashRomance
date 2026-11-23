"""Self-contained, deliberately sparse gallery for the v0.5 drawing study.

All visual marks shown as artwork are embedded renderer outputs. Optional node
and force overlays are analysis views, never composited into those outputs.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path


def build_drawing_gallery(examples: str | Path, output: str | Path) -> Path:
    examples, output = Path(examples), Path(output)
    ledger = json.loads((examples / "drawing_strokes.json").read_text(encoding="utf-8"))
    strokes = ledger.get("strokes", ledger.get("records", [])) if isinstance(ledger, dict) else ledger
    if not strokes:
        raise ValueError("drawing_strokes.json contains no strokes")
    assets = {}
    for name in ("drawing_final.png", "drawing_steps.png", "drawing_turns.png", "drawing_r0.png", "drawing_r1.png", "drawing_r2.png"):
        source = examples / name
        if source.exists():
            assets[name] = "data:image/png;base64," + base64.b64encode(source.read_bytes()).decode("ascii")
    if "drawing_final.png" not in assets:
        raise FileNotFoundError(examples / "drawing_final.png")
    # The canonical ledger stays untouched. Only viewer traces are reduced,
    # retaining every action-boundary row as well as a uniform time sample.
    display_strokes = []
    for source in strokes:
        s = {k: copy.deepcopy(source[k]) for k in ("program", "context", "tag") if k in source}
        s["driver"] = {"events": copy.deepcopy(source.get("driver", {}).get("events", []))}
        trace = source.get("trace", [])
        keep = {0, len(trace) - 1} if trace else set()
        if len(trace) > 640:
            keep.update(round(i * (len(trace) - 1) / 639) for i in range(640))
        else:
            keep.update(range(len(trace)))
        for i, row in enumerate(trace):
            if row.get("event_start") or row.get("event_end"):
                keep.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(trace))
        s["trace"] = [{k: trace[i][k] for k in ("t", "local_t", "pressure", "pressure_target", "event_index", "event_start", "event_end") if k in trace[i]} for i in sorted(keep)]
        s["display_trace_decimated"] = len(keep) < len(trace)
        display_strokes.append(s)
    summary = {k: ledger[k] for k in ("width", "height", "revision", "metadata") if k in ledger} if isinstance(ledger, dict) else {}
    payload = {"ledger": summary, "strokes": display_strokes, "assets": assets}
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = _HTML.replace("__DRAWING_DATA__", embedded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三笔立石 · 离散动作与形体</title>
<style>
:root{color-scheme:light;--paper:#f4f1e8;--ink:#262b2b;--muted:#73776f;--line:#d9d8cc;--accent:#875e46;--blue:#527080}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;line-height:1.75}main{max-width:1080px;margin:auto;padding:44px 28px 80px}header{max-width:790px}.eyebrow{font-size:12px;letter-spacing:.16em;color:var(--muted)}h1{font-family:serif;font-weight:500;font-size:clamp(29px,5vw,44px);letter-spacing:.08em;margin:10px 0 16px}h2{font-weight:500;font-size:22px;margin:0 0 12px}p{margin:9px 0}.intro{font-size:17px}.subtle,figcaption{font-size:13px;color:var(--muted)}.pill{display:inline-block;border:1px solid var(--line);padding:3px 11px;border-radius:20px;font-size:12px;margin-right:7px;margin-top:12px}section{margin-top:52px;padding-top:25px;border-top:1px solid var(--line)}figure{margin:18px 0}.art{display:block;width:100%;height:auto}.hero{max-width:900px;margin:20px auto 0}.artframe{position:relative}.artframe canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:18px 0}button,select{font:inherit;background:#faf8f1;color:var(--ink);border:1px solid #bfc3b7;border-radius:5px;padding:7px 13px}button{cursor:pointer}button[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:var(--paper)}label{font-size:14px}input[type=checkbox]{accent-color:var(--accent)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.info{padding:18px 22px;border-left:2px solid var(--line);background:#ffffff30}.lead{font-size:16px;font-weight:550}.measure{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}.measure span{display:block;font-size:12px;color:var(--muted)}.measure strong{font-size:17px;font-weight:500}.plot{display:block;width:100%;height:220px;background:#ffffff35;border:1px solid var(--line);border-radius:5px}.legend{font-size:12px;color:var(--muted);display:flex;gap:20px;flex-wrap:wrap}.swatch{display:inline-block;width:24px;height:3px;vertical-align:middle;margin-right:7px}.rounds{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.rounds figure{margin:0}.rounds b{font-size:14px;font-weight:550}.review{max-width:800px}.hidden{display:none!important}footer{margin-top:48px;font-size:12px;color:var(--muted)}@media(max-width:720px){main{padding:28px 17px 55px}.grid,.rounds{grid-template-columns:1fr}.rounds figure{max-width:700px}section{margin-top:35px}.measure strong{font-size:15px}}
</style></head><body><main>
<header><div class="eyebrow">水墨代码研究 / v0.5 / DRAWING</div><h1>三笔立石</h1><p class="intro">先确定一笔要建立哪段形体，再决定在哪里落、在哪里折、在哪里收。离散动作中的每一个改向，都要能说出它对应的形体关系。</p><p>上一版以 187 次落笔累积体面，重复排布与填充过密。这次把范围收回到一块石头的勾勒：只保留三笔，先检验线能否立住形。</p><div><span class="pill" id="touchBadge">三笔勾勒</span><span class="pill">画面为代码直接生成</span><span class="pill">不添加排线与面内填充</span></div></header>
<figure class="hero"><div class="artframe"><img id="mainArt" class="art" alt="以三笔离散动作勾勒的石形"><canvas id="nodeOverlay" aria-label="可选的动作节点分析图"></canvas></div><figcaption>最终画面。下方的节点和载荷图用于解释生成过程，默认不覆盖作品。</figcaption></figure>
<section><h2>每一笔怎样承担形体</h2><p class="subtle">选择笔次，再逐个查看动作。坐标、载荷与时长读取实际绘制账本；曲线读取该笔实际执行状态。</p><div class="toolbar" id="strokeButtons"></div><div class="grid"><div class="info"><div class="lead" id="strokeTitle"></div><p id="strokeRole"></p><label for="actionSelect">离散动作</label><div class="toolbar"><select id="actionSelect"></select><label><input id="showNodes" type="checkbox"> 显示动作节点（分析）</label></div><p class="lead" id="actionReason"></p><div class="measure"><div><span>动作目标位置</span><strong id="actionXY"></strong></div><div><span>指令载荷</span><strong id="actionLoad"></strong></div><div><span>动作时长</span><strong id="actionDuration"></strong></div></div><p class="subtle" id="actionNote"></p></div><div><canvas id="forcePlot" class="plot" aria-label="该笔的离散指令载荷与实际接触载荷"></canvas><div class="legend"><span><i class="swatch" style="background:#875e46"></i>动作指令载荷</span><span><i class="swatch" style="background:#527080"></i>实际接触载荷</span></div><p class="subtle">阶梯表示动作给出的目标；实际接触载荷保留笔锋响应与接触状态。手的改向由离散位置节点定义，并未用全局波动函数代替。</p></div></div></section>
<section id="stepsSection"><h2>三次触纸，逐笔成形</h2><p>以下是实际绘制的累计帧。每一帧增加一笔，既有墨迹保留。</p><figure><img id="stepsArt" class="art" alt="第一笔、第二笔、第三笔的累计绘制过程"></figure></section>
<section id="turnsSection"><h2>转折必须有形体理由</h2><p>节点图只标注转折与形体的对应。转折的目标是交代面与棱的变化；短暂的提按与响应滞后服务于这个改向。</p><figure><img id="turnsArt" class="art" alt="三笔动作节点及其形体关系标注"><figcaption>分析示意，标注不属于作品。</figcaption></figure></section>
<section id="roundsSection"><h2>这一次的三轮视觉比对</h2><p class="subtle">三张均为实际代码输出，保留笔数约束以比较起、折、收与结构配置。</p><div class="rounds" id="rounds"></div></section>
<section class="review"><h2>当前检验的边界</h2><p>此例检验三笔能否形成明确的石形，并让每个转折在画面中承担结构。它仍是单块石头的阶段例证，不能据此声称已掌握山水画的笔墨语言。</p><p>离散的是动作决定与目标；接触、供墨和笔锋的跟随状态仍然连续演化。一笔中的顿、折不会自动拆成另一次落笔。</p></section>
<footer>全部图像与动作记录均内嵌，可离线打开。曲线仅作显示抽样，保留动作边界；完整账本见代码包。节点显示是分析覆盖层；原始图像不随控件改变。</footer>
</main><script id="drawingData" type="application/json">__DRAWING_DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('drawingData').textContent),S=D.strokes,A=D.assets;
const el=id=>document.getElementById(id);let chosenStroke=0,chosenAction=0;
const finite=(v,fallback=0)=>Number.isFinite(Number(v))?Number(v):fallback;
const label={touch:'落',drag:'行',move:'行',turn:'折',hold:'顿',dwell:'顿',lift:'收',release:'收',airmove:'空中移笔',set:'定势'};
const getActions=s=>(s.driver?.events?.length?s.driver.events:null)||s.program?.actions||s.program?.events||[];
const getTrace=s=>s.trace||[];
const scalarText=v=>typeof v==='string'?v:(Array.isArray(v)?v.join('；'):(v&&typeof v==='object'?Object.values(v).join('；'):''));
const point=v=>Array.isArray(v)?[finite(v[0]),finite(v[1])]:v&&typeof v==='object'?[finite(v.x),finite(v.y)]:null;
const coord=(a,s,i)=>point(a.to)||point(a.position)||point(a.target)||point(i?getActions(s)[i-1].to:s.program?.start);
const load=a=>Number.isFinite(Number(a.load))?Number(a.load):Number.isFinite(Number(a.pressure))?Number(a.pressure):null;
function canvasSize(c,height){const ratio=window.devicePixelRatio||1,w=Math.max(280,c.getBoundingClientRect().width||600);c.width=Math.round(w*ratio);c.height=Math.round(height*ratio);const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return{ctx,w,h:height};}
function drawForce(){const s=S[chosenStroke],actions=getActions(s),tr=getTrace(s),c=el('forcePlot'),{ctx,w,h}=canvasSize(c,220);ctx.clearRect(0,0,w,h);const p={l:40,r:14,t:17,b:29},iw=w-p.l-p.r,ih=h-p.t-p.b;const t0=finite(tr[0]?.local_t,tr[0]?.t||0);const tt=r=>Number.isFinite(Number(r.local_t))?Number(r.local_t)-finite(tr[0]?.local_t):finite(r.t)-t0;const duration=Math.max(.001,actions.reduce((sum,a)=>sum+finite(a.duration),0),...tr.map(tt));let ymin=0,ymax=Math.max(1,...actions.map(a=>load(a)||0),...tr.map(r=>finite(r.pressure,r.force||r.internal_force||0)));const x=t=>p.l+t/duration*iw,y=v=>p.t+(1-v/ymax)*ih;ctx.font='11px system-ui';ctx.fillStyle='#73776f';ctx.strokeStyle='#deded4';ctx.lineWidth=1;for(let i=0;i<=4;i++){const v=ymax*i/4;ctx.beginPath();ctx.moveTo(p.l,y(v));ctx.lineTo(w-p.r,y(v));ctx.stroke();ctx.fillText(v.toFixed(2),3,y(v)+3)}for(let i=0;i<=4;i++)ctx.fillText((duration*i/4).toFixed(2),x(duration*i/4)-9,h-9);ctx.fillText('秒',w-20,h-9);let t=0,previous=finite(s.program?.initial?.load);ctx.strokeStyle='#875e46';ctx.lineWidth=1.7;ctx.beginPath();ctx.moveTo(x(0),y(previous));for(const a of actions){const next=load(a),v=next===null?previous:next;ctx.lineTo(x(t),y(previous));ctx.lineTo(x(t),y(v));t+=finite(a.duration);ctx.lineTo(x(t),y(v));previous=v}ctx.stroke();if(tr.length){ctx.strokeStyle='#527080';ctx.lineWidth=1.5;ctx.beginPath();tr.forEach((r,i)=>{const v=finite(r.pressure,r.force||r.internal_force||0);if(i===0)ctx.moveTo(x(tt(r)),y(v));else ctx.lineTo(x(tt(r)),y(v))});ctx.stroke()}let at=0;for(let i=0;i<chosenAction;i++)at+=finite(actions[i].duration);ctx.fillStyle='rgba(135,94,70,.09)';ctx.fillRect(x(at),p.t,Math.max(1,finite(actions[chosenAction]?.duration)/duration*iw),ih);}
function drawNodes(){const c=el('nodeOverlay'),img=el('mainArt'),ctx=c.getContext('2d');c.width=img.naturalWidth||700;c.height=img.naturalHeight||500;ctx.clearRect(0,0,c.width,c.height);if(!el('showNodes').checked)return;const s=S[chosenStroke],actions=getActions(s);ctx.font='12px system-ui';ctx.lineWidth=1;let prev=point(s.program?.start);actions.forEach((a,i)=>{const q=coord(a,s,i);if(!q)return;if(prev){ctx.strokeStyle='rgba(135,94,70,.5)';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(prev[0],prev[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([])}ctx.beginPath();ctx.arc(q[0],q[1],i===chosenAction?6:3.5,0,Math.PI*2);ctx.fillStyle=i===chosenAction?'#875e46':'#e4d5b7';ctx.fill();ctx.strokeStyle='#875e46';ctx.stroke();ctx.fillStyle='#66442f';ctx.fillText(String(i+1),q[0]+7,q[1]-6);prev=q;});}
function renderAction(){const s=S[chosenStroke],actions=getActions(s),a=actions[chosenAction]||{},q=coord(a,s,chosenAction);el('actionSelect').value=String(chosenAction);el('actionReason').textContent=scalarText(a.form_event)||scalarText(a.reason)||'此动作的形体说明未记录。';el('actionXY').textContent=q?`${q[0].toFixed(1)}, ${q[1].toFixed(1)}`:'沿用前一位置';const v=load(a);el('actionLoad').textContent=v===null?'沿用前一目标':v.toFixed(3);el('actionDuration').textContent=finite(a.duration).toFixed(3)+' s';let notes=[];if(a.angle!==undefined)notes.push('锋向目标 '+finite(a.angle).toFixed(3)+' rad');if(a.kind)notes.push('动作 '+a.kind);el('actionNote').textContent=notes.join(' · ');drawForce();drawNodes();}
function renderStroke(){const s=S[chosenStroke],ctx=s.context||{},actions=getActions(s);el('strokeTitle').textContent=`第 ${chosenStroke+1} 笔 · ${scalarText(ctx.name)||scalarText(ctx.role)||s.tag||'形体勾勒'}`;el('strokeRole').textContent=scalarText(ctx.form_reason)||scalarText(ctx.form_role)||scalarText(ctx.description)||scalarText(ctx.reason)||scalarText(ctx.start_reason)||'';const select=el('actionSelect');select.replaceChildren();actions.forEach((a,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${i+1} · ${label[a.kind]||a.kind||'动作'}${a.form_event?' · '+scalarText(a.form_event):''}`;select.appendChild(option)});el('strokeButtons').querySelectorAll('button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===chosenStroke)));chosenAction=0;renderAction();}
S.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.textContent=`第 ${i+1} 笔`;b.setAttribute('aria-pressed',String(i===0));b.addEventListener('click',()=>{chosenStroke=i;renderStroke()});el('strokeButtons').appendChild(b)});
el('mainArt').src=A['drawing_final.png'];el('mainArt').addEventListener('load',drawNodes);
for(const [name,img,section]of[['drawing_steps.png','stepsArt','stepsSection'],['drawing_turns.png','turnsArt','turnsSection']]){if(A[name])el(img).src=A[name];else el(section).classList.add('hidden')}
let rounds=0;for(let i=0;i<3;i++){const src=A[`drawing_r${i}.png`];if(!src)continue;rounds++;const fig=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption'),b=document.createElement('b');im.src=src;im.className='art';im.alt=`第 ${i+1} 轮三笔石形`;b.textContent=`第 ${i+1} 轮`;cap.appendChild(b);fig.appendChild(im);fig.appendChild(cap);el('rounds').appendChild(fig)}if(!rounds)el('roundsSection').classList.add('hidden');
const meta=D.ledger.metadata||D.ledger;const contact=meta.actual_contacts;el('touchBadge').textContent=contact===undefined?`${S.length} 笔勾勒`:`${contact} 次触纸 · ${S.length} 笔`;
el('actionSelect').addEventListener('change',e=>{chosenAction=Number(e.target.value);renderAction()});el('showNodes').addEventListener('change',drawNodes);window.addEventListener('resize',()=>{drawForce();drawNodes()});renderStroke();
</script></body></html>'''
