from pathlib import Path
import json,base64,html,gzip,pickle,zipfile,hashlib
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];D=R.parent/'deliverables-v021';rows=json.loads((R/'stroke20/readings.json').read_text())['cases']
f=ImageFont.truetype(str(R/'assets/NotoSansCJKsc-Regular.otf'),22);sm=ImageFont.truetype(str(R/'assets/NotoSansCJKsc-Regular.otf'),16)
checks=[]
for p in (R/'stroke21').glob('*/*/physical/run.json'):
 j=json.loads(p.read_text());diag=j['diagnostics'];z=np.load(p.parent/'fields.npz');assert all(np.isfinite(z[k]).all() for k in z.files)
 err=max(abs(diag[k]) for k in ['water_relative_error','pigment_relative_error']);assert err<1e-8
 checks.append(dict(run=str(p.relative_to(R)),ledger_error=err,finite=True,seconds=j['seconds']))
(R/'stroke21/validation.json').write_text(json.dumps(checks,indent=2))
hold=json.loads((R/'stroke21/holdout-summary.json').read_text());stats=[]
for row in rows:
 for v in ['base','r2']:
  entry=next(r for r in hold if r['case']==row['id'] and r['variant']==v)
  stats.append(dict(case=row['id'],variant=v,seed=21,mean_coverage=float(np.mean([s['coverage'] for s in entry['measurements']['strokes']]))))
(R/'stroke21/holdout-comparison.json').write_text(json.dumps(stats,indent=2))
b=Image.new('RGB',(1080,700),'#f5f1e6');draw=ImageDraw.Draw(b);draw.text((15,10),'第二个纸面／毛束种子｜旧版 vs R2',font=f,fill='#243e32')
for i,row in enumerate(rows):
 draw.text((i*360+15,50),row['title'],font=sm,fill='#243e32')
 for j,(variant,path) in enumerate([('旧版',f'base-{row["id"]}-s21'),('R2',f'{row["id"]}-s21')]):
  a=Image.open(R/f'stroke21/holdout/{path}/physical/result.png');a=a.crop((40,40,a.width-40,a.height-40));a=ImageOps.contain(a,(315,245));x=360*i+20;y=95+j*295;draw.text((x,y-24),variant,font=sm,fill='#536557');b.paste(a,(x+(315-a.width)//2,y))
b.save(D/'shuimo_v021_holdout.jpg',quality=94)
def img(n):return '<img loading="lazy" src="data:image/jpeg;base64,'+base64.b64encode((D/n).read_bytes()).decode()+'">'
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水墨 v0.21 笔触优化</title><style>body{font:18px/1.7 system-ui;max-width:1440px;margin:30px auto;padding:20px;background:#f5f1e6;color:#243e32}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #acb6ac;text-align:left;padding:12px}h1,h2{font-weight:500}small{color:#536557}summary{cursor:pointer}a{color:#1a6955}</style><h1>水墨 v0.21｜把收益优先放在动作与读笔</h1><p>本轮完成两轮反馈，并用另一组纸面、毛束随机种子复验。改善主要是转折路径、短点长宽与提收；纸墨质感保持原模型，斧劈仍保留旧方案。</p>'''
body+=img('shuimo_v021_comparison.jpg')
body+='''<h2>分层取舍</h2><table><tr><th>层级</th><th>本轮处理</th><th>判断</th></tr><tr><td>读笔与路径</td><td>保留原始折点，分别处理勾线、斧劈与短横点。</td><td>高收益，已做</td></tr><tr><td>动作与尺度</td><td>改变按、行、提的时机；小短点单独选笔宽；补偿笔锋滞后。</td><td>高收益，已做两轮</td></tr><tr><td>供墨</td><td>连续耗墨，局部调整浓度；剩余墨量充分时优先查接触。</td><td>局部调整</td></tr><tr><td>评价与随机性</td><td>分报宽度和覆盖；第二种子复验。</td><td>诊断改进，仍需人工复核</td></tr><tr><td>笔号、底材与三维流体</td><td>真实毛长/刚度/储墨标定、纸绢分开、多尺度耦合。</td><td>暂缓，当前成本较高</td></tr><tr><td>大范围蒙特卡洛</td><td>先把真实图例分割与评价可靠性补齐。</td><td>暂缓，避免沿错误目标搜索</td></tr></table><h2>实际反馈</h2><p>R1 的短点更横向，但过细；斧劈反而过宽。R2 回调后短点和转折更完整。没有把失败版删掉，也没有把单一数值分数当作艺术质量。原图有未圈选邻笔，模拟只绘制所选三笔。</p><h2>第二个种子</h2>'''+img('shuimo_v021_holdout.jpg')
body+='<p>只改变固定纸面与毛束的随机种子，重新执行整段物理动作。两个种子只是有限的稳健性检查。</p><details><summary>诊断数值、复演与限制</summary><pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre>'
body+='''<p>覆盖率表示手工圈选路径上可检测到墨的采样比例，不代表艺术相似度。绢的织纹会干扰截面宽度，马远宽度分数不用于自动选版。纸面扫描的黑度不等同于实物墨浓度。动作仍是读图假设。</p><p>前馈 0.010 秒；位移上限勾线0.28毫米、其他0.22毫米。纸网格50微米。笔宽变化不是整支毛笔的几何相似缩放。</p><p>每个版本保留代码、动作、物理场及继续绘制状态，验证水/颜料守恒与有限数值。三类样本之外的笔法未在本轮重新验证。</p></details></html>'''
(D/'shuimo_v021_review.html').write_text(body)
selected=dict(turn='v021-r2',dab='v021-r2',axe='v020-r3',status='provisional visual selection; not a universal learned policy',remaining=['短点局部仍偏窄','尖尾/毛束条纹仍有规则感','绢本不可用现有纸模型作材料真值'])
(R/'stroke21/selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
with zipfile.ZipFile(D/'shuimo_v021_code.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for directory in ['inkscape_lab','research21','stroke21','stroke20','assets','references19','reference-v018']:
  for p in (R/directory).rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts or 'physical' in p.parts or p.suffix in ['.so','.pyc','.log','.gz','.npz'] :continue
   z.write(p,str(p.relative_to(R)))
 for n in ['selected_v016.json','README-v021.md']:z.write(R/n,n)
index=json.loads((R/'stroke21/versions.json').read_text());(D/'shuimo_v021_versions.json').write_text(json.dumps(dict(versions=index,selection=selected,validation=checks),ensure_ascii=False,indent=2))
print(json.dumps(dict(physical_runs=len(checks),max_error=max(r['ledger_error'] for r in checks),holdout=stats),ensure_ascii=False))
