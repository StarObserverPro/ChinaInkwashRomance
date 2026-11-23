"""Build an auditable gallery and comparisons from completed simulations."""
from pathlib import Path
from collections import defaultdict
import json,base64,html,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter
from .monte_carlo_studies import ROOT,sheet,font
from .technique_studies import save_json

REVIEW=ROOT/'review'

def load(path):return json.loads(Path(path).read_text())
def density(path):return np.load(Path(path)/'fields.npz')['density']

def analyze():
    runs=[]
    for stage in ['global','round2','round3','validation']:
        for p in sorted((ROOT/'mc'/stage).glob('*/*/run.json')):
            d=load(p);d['stage']=stage;d['relative_output']=str(p.parent.relative_to(ROOT));runs.append(d)
    error={k:max(abs(d['diagnostics'][k]) for d in runs) for k in ['water_relative_error','pigment_relative_error']}
    convergence=[]
    for case in ['leaf','dry']:
        fields=[density(ROOT/'mc/round3/C06'/case),density(ROOT/'mc/validation/dt2'/case),density(ROOT/'mc/validation/dt4'/case)]
        for (ia,ib,label) in [(0,1,'1.5 to 0.75 ms'),(1,2,'0.75 to 0.375 ms')]:
            a,b=fields[ia],fields[ib];av=-np.expm1(-a);bv=-np.expm1(-b)
            convergence.append(dict(case=case,steps=label,
                relative_density_l1=float(abs(a-b).sum()/max(a.sum(),1e-30)),
                relative_opacity_l1=float(abs(av-bv).sum()/max(av.sum(),1e-30)),
                relative_density_l1_at_100um=float(abs(gaussian_filter(a,2)-gaussian_filter(b,2)).sum()/max(a.sum(),1e-30)),
                pigment_total_change=float((b.sum()-a.sum())/max(a.sum(),1e-30))))
    replay=float(abs(density(ROOT/'mc/global/M09/leaf')-density(ROOT/'mc/round2/B01/leaf')).max())
    groups={}
    for key in ['geometry','flux','supply','release','relief']:
        vals=defaultdict(list)
        for d in runs:
            if d['stage']=='global':vals[d['config']['structure'][key]].append(d['metrics']['score'])
        groups[key]={k:dict(n=len(v),median_proxy_score=float(np.median(v))) for k,v in vals.items()}
    summary=dict(total_runs=len(runs),counts={s:sum(d['stage']==s for d in runs) for s in ['global','round2','round3','validation']},
        max_conservation_error=error,deterministic_replay_max_density_error=replay,convergence=convergence,
        descriptive_structure_groups=groups,selected='C06',
        caveat='Scores are artifact proxies, not realism probabilities. Group effects are confounded by continuous parameters and interactions.')
    save_json(REVIEW/'validation_summary.json',summary)
    rows=[]
    for d in runs:
        row=dict(stage=d['stage'],id=d['id'],case=d['case'],seed=d.get('seed',0),**d['metrics'])
        row.update({k:d['diagnostics'][k] for k in ['water_relative_error','pigment_relative_error','paper_nl','paper_bridge_ruptures']})
        row['source']=d['relative_output'];rows.append(row)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (REVIEW/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return runs,summary

def paste_image(im,path,box,crop=None):
    a=Image.open(path).convert('RGB')
    if crop:a=a.crop(crop)
    a.thumbnail((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
    # Upscaling is for inspection only; no color or contrast adjustment.
    scale=min((box[2]-box[0])/a.width,(box[3]-box[1])/a.height)
    a=a.resize((round(a.width*scale),round(a.height*scale)),Image.Resampling.LANCZOS)
    im.paste(a,(box[0]+(box[2]-box[0]-a.width)//2,box[1]+(box[3]-box[1]-a.height)//2))

def comparisons():
    im=Image.new('RGB',(1400,1060),'#f4f1e8');d=ImageDraw.Draw(im)
    d.text((26,20),'v0.16 / structural Monte Carlo + three visual-feedback rounds',font=font(25),fill='#1e2d28')
    d.text((26,58),'Same gestures and loading across each row. Fixed optical response. No image cleanup.',font=font(16),fill='#58635c')
    cols=[('R0 / control',ROOT/'baseline/r0'),('R5 / previous',ROOT/'baseline/r5'),('C06 / retained candidate',ROOT/'mc/round3/C06')]
    for j,(label,path) in enumerate(cols):d.text((145+j*412,95),label,font=font(19),fill='#1e2d28')
    for i,(case,name) in enumerate([('leaf','Leaf'),('dry','Dry'),('dense','Dry / 2.5x\npigment')]):
        y=131+i*175;d.multiline_text((22,y+58),name,font=font(17),fill='#1e2d28')
        for j,(_,root) in enumerate(cols):
            if case=='dense':path=ROOT/'mc/validation'/(['R0','R5','dense'][j])/'dry/result.png'
            else:path=root/case/'result.png'
            paste_image(im,path,(145+j*412,y,535+j*412,y+155))
    y=666;d.text((22,y+48),'Lift detail',font=font(17),fill='#1e2d28')
    for j,(_,root) in enumerate(cols):paste_image(im,root/'leaf/result.png',(145+j*412,y,535+j*412,y+126),(157,17,266,72))
    d.text((26,816),'Real examples used for qualitative feedback / not pixel-matched targets',font=font(18),fill='#1e2d28')
    paste_image(im,ROOT/'technique-references/mfa-sue-yang-step06.jpg',(28,850,394,1012),(436,324,860,645))
    paste_image(im,ROOT/'reference-v014/dry_rock.png',(455,850,825,1012))
    d.text((860,856),'What improved',font=font(18),fill='#1e2d28')
    d.multiline_text((860,888),'Less prominent round lift marks\nBroader broken ink patches\nRemaining: long grooves, blunt onset',font=font(16),spacing=9,fill='#58635c')
    d.text((28,1025),'References: Sue Yang / MFA Boston; Shitao / The Met (public domain). Cropped and resized only.',font=font(13),fill='#58635c')
    im.save(REVIEW/'comparison.png')

    im=Image.new('RGB',(1780,530),'#f4f1e8');d=ImageDraw.Draw(im)
    d.text((20,15),'Three rounds of visual feedback / retain failures and matched controls',font=font(23),fill='#1e2d28')
    stages=[('R0',ROOT/'baseline/r0'),('R5',ROOT/'baseline/r5'),('Round 1 / M09',ROOT/'mc/global/M09'),('Round 2 / B04',ROOT/'mc/round2/B04'),('Round 3 / C06',ROOT/'mc/round3/C06')]
    for j,(name,root) in enumerate(stages):
        x=15+j*354;d.text((x,60),name,font=font(18),fill='#1e2d28')
        for i,case in enumerate(['leaf','dry']):paste_image(im,root/case/'result.png',(x,100+i*178,x+340,255+i*178))
    d.text((20,478),'All rows preserve the same gesture, loading, concentration and optical coefficient; changes are in model structure and material parameters.',font=font(14),fill='#58635c')
    im.save(REVIEW/'iterations.png')

    im=Image.new('RGB',(1280,1010),'#f4f1e8');d=ImageDraw.Draw(im)
    d.text((24,16),'Held-out gestures / same retained model / no per-gesture fitting',font=font(24),fill='#1e2d28')
    for j,(name,root) in enumerate(cols):d.text((110+j*380,67),name,font=font(17),fill='#1e2d28')
    for i,case in enumerate(['pulse','side','fold','bone']):
        y=105+i*137;d.text((20,y+44),case,font=font(17),fill='#1e2d28')
        for j,(_,root) in enumerate(cols):
            path=root/case/'result.png' if j<2 else ROOT/'mc/validation/heldout'/case/'result.png'
            paste_image(im,path,(110+j*380,y,465+j*380,y+122))
    d.text((22,673),'Rock / 3 strokes, continuous brush state, no refill',font=font(18),fill='#1e2d28')
    for j,sid in enumerate(['R0','R5','selected']):paste_image(im,ROOT/'mc/validation'/sid/'rock/result.png',(110+j*380,707,465+j*380,980))
    im.save(REVIEW/'heldout.png')

    im=Image.new('RGB',(1310,605),'#f4f1e8');d=ImageDraw.Draw(im)
    d.text((20,15),'Numerical sensitivity / same physical input / no tuning after selection',font=font(22),fill='#1e2d28')
    for j,(title,base) in enumerate([('1.5 ms',ROOT/'mc/round3/C06'),('0.75 ms',ROOT/'mc/validation/dt2'),('0.375 ms',ROOT/'mc/validation/dt4')]):
        x=30+j*430;d.text((x,65),title,font=font(20),fill='#1e2d28')
        for i,case in enumerate(['leaf','dry']):paste_image(im,base/case/'result.png',(x,105+i*185,x+400,270+i*185))
    d.text((20,507),'Changing integration step still changes fine details. Stable total pigment does not establish image convergence.',font=font(15),fill='#58635c')
    im.save(REVIEW/'time_steps.png')

def data_image(path):
    path=Path(path);mime='image/gif' if path.suffix=='.gif' else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()

def html_report(runs,summary):
    h=lambda s:html.escape(str(s))
    def img(p,alt):return f'<img loading="lazy" src="{data_image(p)}" alt="{h(alt)}">'
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>水墨 v0.16 · 结构蒙特卡洛</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f1e8;color:#25362e;font:16px/1.7 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:32px}h1{font-size:36px;line-height:1.3}h2{margin-top:44px}p{max-width:1000px}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption,.muted{color:#5a655c;font-size:14px}.stats{display:flex;gap:32px;flex-wrap:wrap}.stats b{font-size:28px;display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}.card{background:#fffdf5;border:1px solid #d9ddd1;padding:15px;border-radius:8px}.card img{display:block;width:100%}code,pre{font-family:ui-monospace,monospace}pre{font-size:12px;max-height:320px;overflow:auto;background:#eeeede;padding:14px}summary{cursor:pointer}table{border-collapse:collapse;max-width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #c6cfc3;padding:10px}a{color:#25674b}nav{display:flex;gap:18px;flex-wrap:wrap}button{font:inherit;padding:7px 17px;border:1px solid #9cae9e;border-radius:22px;background:transparent;cursor:pointer}button.active{background:#2b4e3c;color:white}.hidden{display:none}.note{border-left:4px solid #6b8972;padding:8px 20px;background:#e9ede2}
    </style><main><h1>水墨 v0.16：按最终笔迹筛选方程结构</h1><p>完成三轮反馈，暂时保留 C06。收笔圆斑减轻，枯笔内部出现较宽的断续斑块；梳状白槽、起笔圆钝和数值细节敏感仍存在。</p>''']
    parts.append(f'<div class="stats"><div><b>48</b>首轮结构组合</div><div><b>12</b>连续采样维度</div><div><b>3</b>实际视觉反馈轮次</div><div><b>{summary["total_runs"]}</b>完成渲染</div></div>')
    parts.append('<nav><a href="#iterations">逐轮反馈</a><a href="#gallery">全部样本</a><a href="#validation">验证与局限</a><a href="#reproduce">复现</a></nav>')
    parts.append('<figure>'+img(REVIEW/'comparison.png','R0、R5 与保留候选 C06 的同条件对比')+'<figcaption>各行保持相同手势和装墨。“浓墨枯笔”一行三者都提高浓度 2.5 倍；黏度保持不变。只缩放显示，未调整色阶。</figcaption></figure>')
    parts.append('<h2 id="iterations">三轮实际反馈</h2><table><tr><th>轮次</th><th>实际运行</th><th>目视决策</th></tr><tr><td>1</td><td>48 组 × 叶笔/枯笔 = 96</td><td>保留 M09。淘汰虽得分较低、但仍过淡或白槽过直的方案。</td></tr><tr><td>2</td><td>12 组同材料对照 × 2 = 24</td><td>保留 B04。有限补液 + h³ 候选导流 + 连接期排液；取消断裂瞬间额外圆帽沉积。</td></tr><tr><td>3</td><td>8 组局部随机样本 + 4 组对照，共 24</td><td>保留 C06。毛间凝聚与三体孔隙消融没有足够改善，继续保留。C07 自动分数较低但梳刷感明显。</td></tr></table>')
    parts.append(img(REVIEW/'iterations.png','逐轮叶笔与枯笔'))
    parts.append('<p class="note">自动分数只筛查过淡、碎片和长白槽；不是“真实概率”。分离小块也可能是正常枯笔，不能把连通域数直接当成墨滴数。最终选择来自逐张目视反馈。</p>')
    parts.append('<h2>保留的方程结构</h2><p>铺伏接触段和可达面积通量；有限毛尖补液；低含液导流用 h³ 候选缩放；毛尖离纸但液桥仍连通时，排液继续进入已有湿接触足迹；断裂后未转移的残墨留在毛上。纸内输运、颜料截留与光学密度仍由守恒状态计算。</p><p>h³ 与 h⁴ 在参考膜厚处匹配。本次验证的是一种候选导流律，不是完整环形自由表面求解。断裂残墨留在毛上也是极限闭合，尚未被实拍定量标定。</p>')
    parts.append('<h2 id="gallery">全部筛选样本</h2><p>上图叶笔，下图枯笔；展开可查看精确参数。每轮都有失败样本。</p><div id="tabs">'+''.join(f'<button data-stage="{s}" class="{"active" if s=="global" else ""}">{t}</button>' for s,t in [('global','第一轮 · 48'),('round2','第二轮 · 12'),('round3','第三轮 · 12')])+'</div>')
    for stage in ['global','round2','round3']:
        groups=defaultdict(dict)
        for d in runs:
            if d['stage']==stage:groups[d['id']][d['case']]=d
        parts.append(f'<div class="grid {"" if stage=="global" else "hidden"}" data-gallery="{stage}">')
        for sid,cases in sorted(groups.items()):
            cfg=cases['leaf']['config'];s=cfg['structure']
            parts.append(f'<article class="card"><b>{sid}</b><div class="muted">{h(s["geometry"])} / {h(s["flux"])} / {h(s["release"])}</div>')
            for case in ['leaf','dry']:parts.append(img(ROOT/cases[case]['relative_output']/'result.png',sid+' '+case))
            parts.append('<details><summary>模型参数</summary><pre>'+h(json.dumps(cfg,ensure_ascii=False,indent=2))+'</pre></details></article>')
        parts.append('</div>')
    parts.append('<h2 id="validation">验证与局限</h2>')
    parts.append(f'<p>全部 {summary["total_runs"]} 次运行检查有限值、非负性和材料守恒。最大水量相对误差 {summary["max_conservation_error"]["water_relative_error"]:.2e}，颜料相对误差 {summary["max_conservation_error"]["pigment_relative_error"]:.2e}。M09 → B01 同参数回放的最大密度误差为 {summary["deterministic_replay_max_density_error"]:.1e}。</p>')
    parts.append(img(REVIEW/'heldout.png','未参与搜索的笔法和山石练习'))
    parts.append('<p>山石是原有三笔手写程序的复用，不是新临摹，也未反演画家动作。材料始终延续，不在每一笔自动补墨。</p>')
    parts.append('<p>这轮未参与搜索的笔法仍暴露问题：竹节起笔偏肥、侧锋有长白槽；山石尚缺少皴法的形体组织，不能把三笔轮廓的改变算作山水画真实性通过。</p>')
    parts.append(img(REVIEW/'seeds.png','固定参数，更换纸面和毛束随机种子'))
    parts.append('<p>两个新材料种子仍保留较宽的断续着墨；斑块的具体位置发生变化。这里同时改变了纸面和毛束的随机实例，没有单独识别两者的贡献。</p>')
    parts.append(img(REVIEW/'time_steps.png','三个积分步长的实际笔迹'))
    parts.append('<table><tr><th>笔迹</th><th>时间步</th><th>密度相对 L1</th><th>不透明度相对 L1</th><th>总颜料变化</th></tr>')
    for r in summary['convergence']:parts.append(f'<tr><td>{r["case"]}</td><td>{r["steps"]}</td><td>{r["relative_density_l1"]:.1%}</td><td>{r["relative_opacity_l1"]:.1%}</td><td>{r["pigment_total_change"]:+.1%}</td></tr>')
    parts.append('</table><p class="note">这些差异不是“相似度 = 100% − 差异”。量守恒不能代替细节收敛。本版仍然是未经实测标定的降阶模型；没有完整弹性杆和毛—墨—纸三维自由界面。默认没有蒸发，图像不能称作实物充分干燥后的扫描。</p>')
    parts.append('<h2 id="reproduce">复现与继续编辑</h2><p>完整代码包含全部配置、原始密度场、对照图、反馈记录和可编辑的 selected_v016.json。需要 Python、依赖清单中的库和 C++17 编译器。</p><pre>OPENBLAS_NUM_THREADS=1 python -m inkscape_lab.monte_carlo_studies --config selected_v016.json --case leaf --output mc/my-leaf --workers 1 --animation\n\n# 复跑全局搜索；--force 会覆盖该实验已有输出\nOPENBLAS_NUM_THREADS=1 python -m inkscape_lab.monte_carlo_studies --generate --workers 6 --force</pre>')
    parts.append('<h2>真实参照与文献</h2><ul><li><a href="https://www.mfa.org/programs/community-programs/art-making-at-home/chinese-brush-painting">Sue Yang / MFA Boston 毛笔教学照片</a>：仅作有署名的研究对照，未声称开放许可。</li><li><a href="https://www.metmuseum.org/art/collection/search/49171">石涛山水 / The Met</a>：公共领域馆藏图像，裁切用于观察断口组织。</li><li><a href="https://www.cs.columbia.edu/cg/liquidhair/">Fei 等（2017）与 libWetHair</a>：相关物理建模参考，未把本实验说成完整论文实现。</li></ul>')
    parts.append('<script>document.querySelectorAll("[data-stage]").forEach(b=>b.onclick=()=>{document.querySelectorAll("[data-stage]").forEach(x=>x.classList.toggle("active",x===b));document.querySelectorAll("[data-gallery]").forEach(x=>x.classList.toggle("hidden",x.dataset.gallery!==b.dataset.stage));});</script></main></html>')
    (REVIEW/'review.html').write_text(''.join(parts))

def main():
    REVIEW.mkdir(exist_ok=True)
    runs,summary=analyze();comparisons();html_report(runs,summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
