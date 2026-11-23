"""Rebuild the v0.17 review from retained simulations; never modifies ink fields."""
from pathlib import Path
from collections import Counter
import base64,csv,html,json,math
import numpy as np
from PIL import Image,ImageDraw
from .monte_carlo_studies import ROOT,font
from .sequential_studies import assess,CASES
from .technique_studies import save_json

OUT=ROOT/'review17'
STUDY=ROOT/'study17'
NAMES={'leaf':'Press / travel / lift','dry':'Dry drag','bone':'Press at both ends','fold':'Turning stroke','side':'Side stroke','rock':'Three-stroke rock study'}

def read(path):return json.loads(Path(path).read_text())
def png(path):
    mime='image/jpeg' if Path(path).suffix.lower() in ('.jpg','.jpeg') else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode()
def density(path):
    with np.load(Path(path)/'fields.npz') as z:return z['density'].copy()
def difference(a,b):
    x=density(a);y=density(b);oa=-np.expm1(-x);ob=-np.expm1(-y)
    return dict(density_relative_l1=float(abs(y-x).sum()/max(abs(x).sum(),1e-30)),
        opacity_relative_l1=float(abs(ob-oa).sum()/max(oa.sum(),1e-30)),
        integrated_density_relative_change=float(y.sum()/max(x.sum(),1e-30)-1))

def summarize(sid):
    runs=[]
    for p in sorted(STUDY.glob('*/*/*/*/run.json')):
        d=read(p);d['relative_path']=str(p.parent.relative_to(ROOT));d['stage']=p.relative_to(STUDY).parts[0];runs.append(d)
    result=dict(selected=sid,accepted_default='C06',selection_status='Experimental candidate only; not promoted',renders=len(runs),counts=dict(Counter(d['stage'] for d in runs)),
        max_water_relative_error=max(abs(d['diagnostics']['water_relative_error']) for d in runs),
        max_pigment_relative_error=max(abs(d['diagnostics']['pigment_relative_error']) for d in runs),
        numerical={},paired_robustness=[])
    for case in CASES:
        a=STUDY/'B'/sid/'s0'/case;b=STUDY/'validation/dt2/s0'/case;c=STUDY/'validation/dt4/s0'/case
        if (c/'fields.npz').exists():result['numerical'][case]={'base_to_half':difference(a,b),'half_to_quarter':difference(b,c)}
    for seed in [0,11,37]:
        for case in CASES:
            a=STUDY/'A/A00/s0'/case if seed==0 else STUDY/f'C/C06/s{seed}'/case
            b=STUDY/f'B/{sid}/s0'/case if seed==0 else STUDY/f'C/{sid}/s{seed}'/case
            if (b/'fields.npz').exists():
                result['paired_robustness'].append(dict(seed=seed,case=case,baseline=assess(density(a)),selected=assess(density(b))))
    # Matched factorial contrasts. This is one paper/brush realization, not an
    # inferential estimate of global sensitivity or a confidence interval.
    effects={}
    for case in CASES:
        values=[]
        for i in range(8):values.append(read(STUDY/f'A/A{i:02d}/s0/{case}/run.json')['descriptors'])
        effects[case]={}
        for name,bit in [('axis_blend',2),('contact_rise',1),('fiber_coupling',0)]:
            effects[case][name]={k:float(np.mean([values[i][k] for i in range(8) if (i>>bit)&1])-np.mean([values[i][k] for i in range(8) if not (i>>bit)&1])) for k in ['head_mid_width_ratio','white_corridor_fraction','interior_opacity','tail_satellite_opacity_fraction']}
    result['single_seed_factorial_contrasts']=effects
    save_json(OUT/'summary.json',result)
    cols=['stage','id','case','seed','sampling_scope','relative_path','head_mid_width_ratio','white_corridor_fraction','interior_white_fraction','interior_opacity','satellite_opacity_fraction','tail_satellite_opacity_fraction','total_density','water_relative_error','pigment_relative_error']
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for d in runs:
            row={k:d.get(k,'') for k in cols};row.update({k:v for k,v in d['descriptors'].items() if k in cols});row.update({k:d['diagnostics'][k] for k in ['water_relative_error','pigment_relative_error']});w.writerow(row)
    return result,runs

def panel(title,rows,labels,path,cell=(520,225),footer='Same camera, ink mapping and gesture. Enlarged simulation pixels; no image retouching.'):
    cw,ch=cell;im=Image.new('RGB',(cw*len(labels),95+ch*len(rows)+55),'#f4f1e8');d=ImageDraw.Draw(im)
    d.text((20,16),title,font=font(24),fill='#233a30')
    for i,label in enumerate(labels):d.text((i*cw+16,61),label,font=font(18),fill='#47594c')
    for j,(caption,paths) in enumerate(rows):
        y=95+j*ch
        for i,p in enumerate(paths):
            x=i*cw;d.text((x+14,y+5),caption,font=font(15),fill='#526052')
            a=Image.open(p).convert('RGB');scale=min((cw-20)/a.width,(ch-38)/a.height)
            a=a.resize((round(a.width*scale),round(a.height*scale)),Image.Resampling.LANCZOS)
            im.paste(a,(x+(cw-a.width)//2,y+32))
        d.line((0,y+ch-1,im.width,y+ch-1),fill='#ccd0c4')
    d.text((17,im.height-36),footer,font=font(14),fill='#5b6156');im.save(path)

def figures(sid):
    rows=[(NAMES[c],[STUDY/f'A/A00/s0/{c}/result.png',STUDY/f'B/{sid}/s0/{c}/result.png']) for c in CASES]
    panel(f'v0.17 / local candidate {sid}, not a universal improvement',rows,['v0.16 / C06 retained',f'v0.17 / {sid} candidate'],OUT/'comparison.png')
    rows=[(f'{c} / seed {seed}',[STUDY/f'C/{s}/s{seed}/{c}/result.png' for s in ['C06',sid]]) for seed in [11,37] for c in CASES]
    panel('Matched new brush / paper realizations',rows,['v0.16 / C06',f'v0.17 / {sid}'],OUT/'robustness.png',cell=(520,210))
    rows=[(NAMES[c],[STUDY/f'validation/{s}/s0/{c}/result.png' for s in ['C06','selected']]) for c in ['fold','side','rock']]
    panel('Held-out gestures / no retuning',rows,['v0.16 / C06',f'v0.17 / {sid}'],OUT/'heldout.png',cell=(520,315))
    rows=[(NAMES[c],[STUDY/f'B/{sid}/s0/{c}/result.png',STUDY/f'validation/dt2/s0/{c}/result.png',STUDY/f'validation/dt4/s0/{c}/result.png']) for c in CASES]
    panel('Time and path-step refinement / paper grid fixed',rows,['1.5 ms / 35 um','0.75 ms / 17.5 um','0.375 ms / 8.75 um'],OUT/'numerical.png',cell=(450,205))

def report(sid,summary,runs):
    feedback={p.stem:read(p) for p in sorted(STUDY.glob('feedback_*.json'))}
    cards=[]
    for d in runs:
        m=d['descriptors'];p=ROOT/d['relative_path'];s=d['config']['structure']
        cards.append(f'''<article class="sample" data-stage="{d['stage']}" data-seed="{d['seed']}" data-case="{d['case']}"><b>{d['stage']} / {d['id']} / {d['case']} / seed {d['seed']}</b><img loading="lazy" src="{png(p/'result.png')}" alt="{d['id']} {d['case']}"><small>{html.escape(d.get('sampling_scope',''))} · opacity {m.get('interior_opacity',0):.3f} · white run {m.get('white_corridor_fraction',0):.3f}</small><details><summary>参数与守恒诊断</summary><pre>{html.escape(json.dumps({'config':d['config'],'gesture':d['gestures'],'loading':d['loading'],'descriptors':m,'conservation':{k:d['diagnostics'][k] for k in ['water_relative_error','pigment_relative_error']}},ensure_ascii=False,indent=2))}</pre></details></article>''')
    compare=[]
    for case in CASES:
        old=png(STUDY/f'A/A00/s0/{case}/result.png');new=png(STUDY/f'B/{sid}/s0/{case}/result.png')
        compare.append(f'''<div class="compare"><b>{NAMES[case]}</b><div class="overlay"><img src="{old}" alt="v0.16 {case}"><img class="new" src="{new}" alt="v0.17 {case}"></div><input aria-label="新旧对照分界" type="range" min="0" max="100" value="50" oninput="this.previousElementSibling.querySelector('.new').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'"><small>左侧 v0.17 {sid}；右侧 v0.16 C06。拖动分界；图像采用相同坐标与显色。</small></div>''')
    tables=[]
    for case,d in summary['numerical'].items():
        a=d['base_to_half'];b=d['half_to_quarter']
        tables.append(f"<tr><td>{case}</td><td>{a['density_relative_l1']:.1%} → {b['density_relative_l1']:.1%}</td><td>{a['opacity_relative_l1']:.1%} → {b['opacity_relative_l1']:.1%}</td><td>{b['integrated_density_relative_change']:+.2%}</td></tr>")
    narrative=(ROOT/'docs/sequential_v017.md').read_text() if (ROOT/'docs/sequential_v017.md').exists() else 'Report not yet finalized.'
    refs='<a href="https://www.itl.nist.gov/div898/handbook/pri/section3/pri33.htm">NIST：实验设计选择</a> · <a href="https://www.itl.nist.gov/div898/handbook/pri/section5/pri53.htm">NIST：顺序优化</a> · <a href="https://arxiv.org/abs/1910.01739">TuRBO 原论文（仅为局部搜索思路参考）</a> · <a href="https://github.com/uber-research/TuRBO">作者仓库（未运行）</a>'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水墨 v0.17 · 顺序实验</title><style>body{margin:0;background:#f4f1e8;color:#243a30;font:16px/1.7 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:32px 25px 80px}h1{font-size:32px}h2{margin-top:42px;font-size:23px}p{max-width:950px}a{color:#32684e}small{display:block;color:#637062}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.sample,.compare{border:1px solid #c6cec0;padding:13px;border-radius:8px;background:#faf8f1}.sample img{width:100%;height:auto}.overlay{position:relative}.overlay img{display:block;width:100%}.overlay .new{position:absolute;top:0;left:0;clip-path:inset(0 50% 0 0)}input[type=range]{width:100%}select{padding:8px;margin:8px;border:1px solid #a6b3a5;background:#faf8f1}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:650px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #c6cec0;padding:9px;text-align:left}.figure{width:100%;height:auto}.note{padding:15px 20px;border-left:4px solid #52735a;background:#e9eddf}details{margin:14px 0}summary{cursor:pointer}.hidden{display:none}</style><main>'''
    doc+=f'<h1>水墨 v0.17：先拆分结构，再做局部搜索</h1><p class="note">{summary["renders"]} 次新渲染；C06 继续作为默认基线，{sid} 只作为试验候选。本轮改善没有在全部随机实现中保持，不应把候选称为全面升级。自动指标不是真实度评分，参数也未做材料实测标定。</p>'
    doc+='<p>局部 12 组＋较宽范围 4 组，接在 8 组结构对照之后。三个入围配置再换两组纸张／毛束随机实现。最后锁定参数，检查新笔法、另一随机种子与更细时间／轨迹采样。没有把所有公式和材料同时重新打乱。</p>'
    doc+='<h2>这轮实际得到什么</h2><p>B08 在初始干笔样本中更连贯，但换纸纹／毛束后，白槽改善未能重复。宽范围候选 B14 的白槽较短，却伴随更薄的墨层和较多脱离主体的断墨；单一指标会选错方向。候选在折笔、侧锋和三笔石形中维持了原有墨感，起笔圆钝及石形缺乏皴法组织的问题仍在。</p><p>因此，本轮停止继续扩大参数搜索，保留 C06 默认和 B08 试验分支。下一项更有依据的改进是触纸初期的接触展开与提笔事件处理；应先提高该过程对步长的稳定性，再判断小幅参数变化的收益。4 组宽范围样本不能证明不存在更优区域。</p>'
    doc+='<h2>同条件新旧对照</h2><div class="grid">'+''.join(compare)+'</div>'
    doc+='<h2>逐轮反馈</h2>'
    for key,data in feedback.items():doc+=f'<details open><summary>{html.escape(key)}</summary><pre>{html.escape(json.dumps(data,ensure_ascii=False,indent=2))}</pre></details>'
    doc+='<h2>新随机实现与未参与选择的笔法</h2><img class="figure" src="'+png(OUT/'robustness.png')+'" alt="随机种子对照"><img class="figure" src="'+png(OUT/'heldout.png')+'" alt="新笔法对照">'
    doc+='<h2>数值稳定性</h2><p>箭头为默认步长→半步、半步→四分之一步的差异。网格一直为 50 µm，这不是网格收敛试验。密度／不透明度采用逐像素相对 L1；末列为第二次细化的总光学密度变化。总量稳定不代表飞白边缘已收敛。</p><table><tr><th>笔法</th><th>密度 L1</th><th>不透明度 L1</th><th>总密度变化</th></tr>'+''.join(tables)+'</table>'
    doc+=f'<p>最大水量相对守恒误差 {summary["max_water_relative_error"]:.2e}；最大颜料相对守恒误差 {summary["max_pigment_relative_error"]:.2e}。全部运行检查非负性与有限值。</p><img class="figure" src="'+png(OUT/'numerical.png')+'" alt="步长细化">'
    doc+='<h2>完整实验画廊</h2><label>阶段<select id="stage"><option value="">全部</option>'+''.join(f'<option>{s}</option>' for s in ['A','B','C','validation'])+'</select></label><label>笔法<select id="case"><option value="">全部</option>'+''.join(f'<option>{s}</option>' for s in ['leaf','dry','bone','fold','side','rock'])+'</select></label><label>种子<select id="seed"><option value="">全部</option>'+''.join(f'<option>{s}</option>' for s in ['0','11','37','83'])+'</select></label><span id="count"></span><div class="grid">'+''.join(cards)+'</div>'
    doc+='<h2>方法、来源与边界</h2><p>'+refs+'</p><p>真实笔触参照：<a href="https://www.mfa.org/programs/community-programs/art-making-at-home/chinese-brush-painting">Sue Yang / Museum of Fine Arts, Boston</a>。纸、光照、动作及比例未标定，因此只作定性参照。无真实图片像素进入求解器。</p>'
    ref=ROOT/'technique-references/mfa-sue-yang-step06.jpg'
    if ref.exists():doc+='<figure><img style="width:100%;max-width:860px" src="'+png(ref)+'" alt="Sue Yang 的真实笔法照片"><figcaption>Artwork and instructions courtesy of Sue Yang; Museum of Fine Arts, Boston。用于有署名的视觉研究，未声明开放许可。真实笔触有丰富的受力端部与墨层变化；此图并非逐像素目标。</figcaption></figure>'
    doc+='<details><summary>完整实验说明（原始 Markdown）</summary><pre>'+html.escape(narrative)+'</pre></details></main><script>function filter(){let n=0;document.querySelectorAll(".sample").forEach(c=>{const ok=["stage","case","seed"].every(k=>!document.getElementById(k).value||c.dataset[k]===document.getElementById(k).value);c.classList.toggle("hidden",!ok);n+=ok});document.getElementById("count").textContent=n+" 张"}document.querySelectorAll("select").forEach(s=>s.onchange=filter);filter();</script></html>'
    (OUT/'review.html').write_text(doc)

def main():
    OUT.mkdir(exist_ok=True)
    sid=read(STUDY/'feedback_C.json')['selected'];summary,runs=summarize(sid);figures(sid);report(sid,summary,runs)
    print(json.dumps({k:summary[k] for k in ['selected','renders','counts','max_water_relative_error','max_pigment_relative_error','numerical']},indent=2))

if __name__=='__main__':main()
