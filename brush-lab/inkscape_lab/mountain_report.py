"""Build the self-contained comparison viewer and publication contact sheets."""
import json,base64,io,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .landscape_engine import ROOT
from .mountain_driver import prepare,metrics
from .mountain_policy import NAMES
from .mountain_session import WIDTHS
def uri(im,limit=750,fmt='PNG'):
 im=im.copy();im.thumbnail((limit,limit),Image.Resampling.LANCZOS);b=io.BytesIO();im.save(b,format=fmt,**({'quality':89} if fmt=='JPEG' else {}));return 'data:image/'+('jpeg' if fmt=='JPEG' else 'png')+';base64,'+base64.b64encode(b.getvalue()).decode()
def crop_paper(im,A):
 x=im.width*10/A['D'].shape[1];y=im.height*10/A['D'].shape[0]
 return im.crop((round(x),round(y),round(im.width-x),round(im.height-y)))
def make(partial=False):
 rows=json.loads((ROOT/'references19/manifest.json').read_text());reviewpath=ROOT/'driver19/accepted.json';reviews=json.loads(reviewpath.read_text()) if reviewpath.exists() else {};cases=[];stats=[]
 selected=ROOT/'driver19/selected';selected.mkdir(exist_ok=True)
 for row in rows:
  A=prepare(row);rounds=[];notes=reviews.get(row['id'],{});ids=[]
  for n in [0,1,2,3]:
   out=ROOT/f'driver19/r{n}/{row["id"]}';p=out/'physical'
   if not (p/'run.json').exists():continue
   r=json.loads((p/'run.json').read_text());m,_=metrics(A,p);(out/'metrics.json').write_text(json.dumps(m,indent=2))
   recipe=json.loads((p/'full-recipe.json').read_text());strokes=[dict(op['decision'],tool_width=WIDTHS[op['tool']]) for op in recipe['operations'] if op['kind']=='stroke']
   image=crop_paper(Image.open(p/'result.png').convert('RGB'),A);image.save(out/'copy-crop.png')
   stage_rows=r['stages'];take=np.unique(np.linspace(0,len(stage_rows)-1,min(6,len(stage_rows))).round().astype(int)) if stage_rows else []
   stages=[dict(label=stage_rows[i]['label'],image=uri(crop_paper(Image.open(p/stage_rows[i]['file']),A),350)) for i in take]
   conservation=max(abs(r['diagnostics'][k]) for k in ['water_relative_error','pigment_relative_error'])
   labels={0:'R0 初始方案',1:'R1 形体校正',2:'R2 实际墨迹反馈',3:'R3 干层与低供液试验'}
   rounds.append(dict(round=n,label=labels[n],image=uri(image),review=notes.get('rounds',{}).get(str(n),'待视觉复核'),total_strokes=r['total_strokes'],metrics=m,conservation=conservation,stages=stages,counts=dict(Counter(s['family'] for s in strokes)),strokes=strokes));ids.append(n)
  if not rounds:continue
  if not partial:assert row['id'] in reviews and notes.get('reviewed'),row['id']+' has not been visually reviewed'
  chosen=notes.get('accepted',ids[int(np.argmin([x['metrics']['loss'] for x in rounds]))]);accepted=ids.index(chosen)
  source=Image.open(ROOT/row['reference']).convert('RGB');t=Image.fromarray(np.uint8((1-A['alpha'])*255));target=t.crop((10,10,t.width-10,t.height-10))
  Image.open(ROOT/f'driver19/r{chosen}/{row["id"]}/copy-crop.png').save(selected/(row['id']+'.png'))
  plans=[json.loads(f.read_text()) for f in sorted((ROOT/'driver19').glob(f'r*/{row["id"]}/plan.json'))]
  attempts=sum(p['attempts'] for p in plans)
  cases.append(dict(**{k:row[k] for k in ['id','artist','title','date','description','source','calibration']},reference=uri(source,1000,'JPEG'),thumbnail=uri(source,96,'JPEG'),target=uri(target),path_target=uri(t),size=A['size'],accepted=accepted,rounds=rounds,feedback=notes.get('feedback',[]),search_text=f'保存的主方案和反馈方案合计尝试 {attempts:,} 个候选。主方案前瞻 5 笔，保留 2 个分支；其余 10 幅各搜索两个种子，按代理损失选主方案。',tags=row['tags']))
  stats.append(dict(id=row['id'],accepted=chosen,rounds=ids,**rounds[accepted]['metrics'],conservation=rounds[accepted]['conservation'],strokes=rounds[accepted]['total_strokes']))
 if not partial:assert len(cases)==11,'11 completed reference copies required'
 ablation=[];ap=ROOT/'driver19/ablation-summary.json'
 if ap.exists():
  raw=json.loads(ap.read_text())
  for key,label in [('tutorial','文字先验 · 前瞻 5 笔'),('neutral_text','去文字先验 · 前瞻 5 笔'),('shallow','文字先验 · 前瞻 1 笔')]:
   rr=[r for r in raw if r['method']==key]
   if rr:ablation.append(dict(label=label,n=len(rr),mean=float(np.mean([r['loss'] for r in rr])),low=min(r['loss'] for r in rr),high=max(r['loss'] for r in rr),attempts=rr[0]['attempts']))
 data=dict(headline=f'{len(cases)} 处真实山体局部 · 教程先验与蒙特卡洛选笔 · 查看原图、物理墨迹及每一笔的调用理由',names=NAMES,sources=json.loads((ROOT/'research19/evidence.json').read_text()),cases=cases,ablation=ablation,footer='v0.19｜所有最终墨迹由保存的动作程序复演。原作图像：The Metropolitan Museum of Art，Public Domain。参考裁切框、随机种子、物理场、完整动作程序和误差记录均随实验包提供。')
 deliver=ROOT.parent/'deliverables';deliver.mkdir(exist_ok=True)
 field_figure=deliver/'shuimo_v019_physics_fields.png'
 data['physics_figure']=uri(Image.open(field_figure),1600) if field_figure.exists() else None
 html=(ROOT/'assets/mountain-viewer-template.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 (deliver/'shuimo_v019_viewer.html').write_text(html);(ROOT/'driver19/delivery-metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
 font=ImageFont.truetype(str(ROOT/'assets/NotoSansCJKsc-Regular.otf'),21);small=ImageFont.truetype(str(ROOT/'assets/NotoSansCJKsc-Regular.otf'),16)
 w=1440;cellw=480;cellh=310;board=Image.new('RGB',(w,114+cellh*((len(cases)+2)//3)),(245,241,230));d=ImageDraw.Draw(board)
 d.text((24,14),f'{len(cases)} 处山体局部：原作 / 毛笔纸墨模拟'+('（进度预览）' if partial else ''),font=ImageFont.truetype(str(ROOT/'assets/NotoSansCJKsc-Regular.otf'),29),fill=(40,48,43));d.text((24,58),'每格左：馆藏局部　右：保留的实际模拟版本；完整对照与逐笔决策见交互页。',font=small,fill=(85,96,86))
 for i,c in enumerate(cases):
  x=i%3*cellw;y=90+i//3*cellh;d.text((x+14,y+8),c['id']+' '+c['artist']+' · '+c['title'],font=font,fill=(40,48,43))
  for j,img in enumerate([Image.open(ROOT/next(r['reference'] for r in rows if r['id']==c['id'])),Image.open(selected/(c['id']+'.png'))]):
   img=ImageOps.contain(img.convert('RGB'),(218,245),Image.Resampling.LANCZOS);board.paste(img,(x+12+j*234+(218-img.width)//2,y+45+(245-img.height)//2))
  d.text((x+14,y+289),'原作',font=small,fill=(85,96,86));d.text((x+249,y+289),'模拟 · R'+str(c['rounds'][c['accepted']]['round']),font=small,fill=(85,96,86))
 board.save(deliver/'shuimo_v019_11_copies.jpg',quality=94)
 c=next((c for c in cases if c['id']=='C00'),None)
 if c:
  panels=[('沈周 · 原作局部',Image.open(ROOT/rows[0]['reference']))]+[(r['label'],Image.open(ROOT/f'driver19/r{r["round"]}/C00/copy-crop.png')) for r in c['rounds']]
  cw=1040;ch=round(1000/panels[0][1].width*panels[0][1].height)+65;out=Image.new('RGB',(cw,ch*len(panels)),(245,241,230));dd=ImageDraw.Draw(out)
  for i,(name,im) in enumerate(panels):
   im=ImageOps.contain(im.convert('RGB'),(1000,ch-55),Image.Resampling.LANCZOS);dd.text((20,i*ch+10),name,font=font,fill=(40,48,43));out.paste(im,(20,i*ch+48))
  out.save(deliver/'shuimo_v019_first_comparison.jpg',quality=94)
 print(json.dumps(dict(cases=len(cases),physical_rounds=sum(len(c['rounds']) for c in cases),viewer_bytes=(deliver/'shuimo_v019_viewer.html').stat().st_size),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();make(a.partial)
