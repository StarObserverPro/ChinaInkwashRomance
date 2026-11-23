from pathlib import Path
import json,base64
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from inkscape_lab.stroke_score21 import evaluate21
R=Path(__file__).resolve().parents[1];D=R.parent/'deliverables-v021';D.mkdir(exist_ok=True)
rows=json.loads((R/'stroke20/readings.json').read_text())['cases'];f=ImageFont.truetype(str(R/'assets/NotoSansCJKsc-Regular.otf'),23);sm=ImageFont.truetype(str(R/'assets/NotoSansCJKsc-Regular.otf'),16)
rounds=[x for x in [1,2] if all((R/f'stroke21/r{x}/{r["id"]}/physical/run.json').exists() for r in rows)]
cols=2+len(rounds);cw=360;ch=385;im=Image.new('RGB',(cw*cols,85+ch*3),'#f5f1e6');d=ImageDraw.Draw(im)
d.text((16,12),'v0.21｜动作、转折与逐笔尺度的分层优化',font=f,fill='#243e32');d.text((16,48),'每列保持相同局部范围；参照图有未圈选邻笔。原物理模型不变。',font=sm,fill='#536557');summary=[]
for i,row in enumerate(rows):
 y=85+i*ch;d.text((16,y),row['title'],font=f,fill='#243e32')
 cells=[('原作局部',Image.open(R/f'stroke20/{row["id"]}-reference.png'))]
 for root,label in [('stroke20/r3','v0.20 R3')]+[(f'stroke21/r{n}',f'v0.21 R{n}') for n in rounds]:
  p=R/root/row['id'];task=json.loads((p/'task.json').read_text());m=evaluate21(task,row);summary.append(dict(case=row['id'],version=label,measurements=m));a=Image.open(p/'physical/result.png');pad=round(2/.05);a=a.crop((pad,pad,a.width-pad,a.height-pad));cells.append((label,a))
 for j,(label,a) in enumerate(cells):
  a=ImageOps.contain(a.convert('RGB'),(cw-30,ch-85),Image.Resampling.LANCZOS);x=cw*j;d.text((x+16,y+40),label,font=sm,fill='#536557');im.paste(a,(x+15+(cw-30-a.width)//2,y+70+(ch-90-a.height)//2))
im.save(D/'shuimo_v021_comparison.jpg',quality=94);(R/'stroke21/comparison-metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('Rounds',rounds)
print(json.dumps([{k:r[k] for k in ['case','version']}|dict(width_ratio=[round(s['width_ratio'],2) for s in r['measurements']['strokes']],coverage=[round(s['coverage'],2) for s in r['measurements']['strokes']]) for r in summary],ensure_ascii=False))
