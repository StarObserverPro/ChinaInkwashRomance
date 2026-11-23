"""v0.19 recovery: persistent multiple brushes on one conservative paper.

No reference pixels, image masks or texture stamps enter this renderer.
Checkpoints are trusted internal pickle files, never untrusted uploads.
"""
from pathlib import Path
import os,json,time,hashlib,pickle,gzip,argparse
from dataclasses import asdict
import numpy as np
from .landscape_engine import ROOT,LandscapeBrush,connect_air
from .brush import InkCanvas
from .technique_controls import BrushLoading,InkMaterial,InkPaperInteraction
from .technique_studies import save_json
WIDTHS={'fine':.55,'side':1.0,'wash':1.30}

def signature(recipe,seed,resume=None):
 h=hashlib.sha256(json.dumps([recipe,seed,WIDTHS],sort_keys=True).encode())
 for n in ['mountain_session.py','mountain_paper.py','mountain_paper.cpp','landscape_engine.py','structural_brush.py','structural_transfer.py','technique_controls.py','meniscus.py','technique_paper.cpp','paper_transport.py']:
  h.update((ROOT/'inkscape_lab'/n).read_bytes())
 h.update((ROOT/'selected_v016.json').read_bytes())
 if resume:
  with open(resume,'rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def initialize(recipe,seed):
 cfg=json.loads((ROOT/'selected_v016.json').read_text());W,H=recipe['size_mm']
 canvas=InkCanvas(round(W*20),round(H*20),seed=505+seed,paper_color=(245,240,225));canvas.density=canvas.density.astype(float)
 brushes={};paper=None
 from . import absorbent_brush
 factory=absorbent_brush.PaperTransport
 try:
  for name,width in WIDTHS.items():
   if paper is not None:absorbent_brush.PaperTransport=lambda *a,**k:paper
   b=LandscapeBrush(canvas,width_mm=width,structure=cfg['structure'],technique=dict(cfg.get('technique',{}),max_step_s=.0015,spatial_step_mm=.035),
    loading=BrushLoading(body_saturation=0,tip_saturation=0,body_concentration_kg_m3=0,tip_concentration_kg_m3=0),
    ink=InkMaterial(),interaction=InkPaperInteraction(**cfg.get('interaction',{})),
    paper_overrides=dict(cfg.get('paper',{}),seed=70713+seed),meniscus_overrides=dict(cfg.get('meniscus',{}),anatomy_seed=1212+seed),surface_contact_overrides=dict(seed=14009+seed),record_source=False)
   paper=b.substrate;brushes[name]=b
 finally:absorbent_brush.PaperTransport=factory
 from .mountain_paper import entry_native
 c=cfg['interaction'];paper.native=entry_native(c.get('entry_capture_fine',0),c.get('entry_capture_coarse',0))
 return dict(brushes=brushes,canvas=canvas,paper=paper,time=0.,last='fine',total_strokes=0,history=[],config=cfg)

def save_state(path,state):
 objects=[state['paper'],*state['brushes'].values()];native=[o.native for o in objects]
 try:
  for o in objects:o.native=None
  tmp=Path(str(path)+'.tmp')
  with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump(state,f,protocol=5)
  os.replace(tmp,path)
 finally:
  for o,n in zip(objects,native):o.native=n

def load_state(path):
 with gzip.open(path,'rb') as f:s=pickle.load(f)
 from .meniscus import native_film
 from .mountain_paper import entry_native
 interaction=s['config'].get('interaction',{})
 s['paper'].native=entry_native(interaction.get('entry_capture_fine',0),interaction.get('entry_capture_coarse',0))
 for b in s['brushes'].values():b.native=native_film()
 return s

def ledger(s):
 p=s['paper'];tot=p.totals();w=tot['water_m3'];m=tot['pigment_kg'];wi=mi=0.
 for b in s['brushes'].values():
  w+=float(b.volume.sum()+b.neck_v.sum()+sum(x['v'] for x in b.pores.values()))+b.rinsed_water+b.evaporated_water
  m+=float(b.mass.sum()+b.neck_m.sum()+sum(x['m'] for x in b.pores.values()))+b.rinsed_pigment
  wi+=b.water_initial+b.water_added_si+b.paper_added_water;mi+=b.mass_initial+b.mass_added_si
 return dict(water_relative_error=(w-wi)/max(wi,1e-30),pigment_relative_error=(m-mi)/max(mi,1e-30),water_input_nl=wi*1e12,pigment_input_ng=mi*1e12,substrate=tot)

def run(task):
 start=time.time();recipe=task['recipe'];seed=task.get('seed',19);out=ROOT/task['output'];out.mkdir(parents=True,exist_ok=True)
 resume=ROOT/task['resume'] if task.get('resume') else None;sig=signature(recipe,seed,resume)
 if (out/'run.json').exists() and not task.get('force'):
  old=json.loads((out/'run.json').read_text())
  if old['fingerprint']!=sig:raise ValueError('existing result has a different input/source fingerprint')
  return old
 if (out/'progress-state.pkl.gz').exists() and not task.get('force'):
  s=load_state(out/'progress-state.pkl.gz')
  if s['active_signature']!=sig:raise ValueError('checkpoint source/input mismatch')
  first=s['next_operation']
 else:
  s=load_state(resume) if resume else initialize(recipe,seed);first=0
 s['active_signature']=sig;strokes=0;stages=[]
 if first:stages=[r for r in json.loads((out/'progress.json').read_text()).get('stages',[]) if r['operation']<first]
 for i,op in enumerate(recipe['operations']):
  if i<first:continue
  name=op.get('tool',s['last']);b=s['brushes'][name]
  b.time=s['time'];b._sync_paper();b.previous_film[:]=b.film_h;b.previous_mass[:]=b.film_mass
  if op['kind']=='stroke':
   if op.get('loading'):b.rinse_load(BrushLoading(**op['loading']))
   connect_air(b,op['gesture']);b.draw_gesture(op['gesture']);s['total_strokes']+=1;strokes+=1
  elif op['kind']=='wait':b.relax(op['seconds'])
  elif op['kind']=='dry_limit':b.dry_limit()
  else:raise ValueError(op['kind'])
  s['time']=b.time;s['last']=name;s['history'].append(op);s['next_operation']=i+1
  if op.get('checkpoint') or strokes%12==0 or i==len(recipe['operations'])-1:
   b.flush_paper();d=ledger(s)
   if max(abs(d['water_relative_error']),abs(d['pigment_relative_error']))>1e-8:raise RuntimeError('conservation failure')
   label=op.get('checkpoint',f'{s["total_strokes"]} 笔');fn=f'stage-{i:03}.png';s['canvas'].to_image().save(out/fn)
   stages.append(dict(label=label,file=fn,operation=i,time=s['time']))
   print(json.dumps(dict(id=recipe['id'],round=task.get('round',0),stroke=s['total_strokes'],op=i+1,total=len(recipe['operations']),seconds=round(time.time()-start,1))),flush=True)
  save_state(out/'progress-state.pkl.gz',s)
  save_json(out/'progress.json',dict(next_operation=i+1,total_operations=len(recipe['operations']),total_strokes=s['total_strokes'],stages=stages,seconds=time.time()-start))
 b=s['brushes'][s['last']];b.relax(recipe.get('settle_s',.08));s['time']=b.time
 s['history'].append(dict(kind='wait',seconds=recipe.get('settle_s',.08),reason='recorded settling time'))
 p=s['paper'];p.check();d=ledger(s)
 if max(abs(d['water_relative_error']),abs(d['pigment_relative_error']))>1e-8:raise RuntimeError('conservation failure')
 s['canvas'].to_image().save(out/'result.png')
 np.savez_compressed(out/'fields.npz',density=s['canvas'].density,water=p.water,fixed=p.fixed,mobile=p.fine+p.coarse,film=p.film)
 save_state(out/'state.pkl.gz',s)
 result=dict(id=recipe['id'],round=task.get('round',0),fingerprint=sig,recipe=recipe,seed=seed,total_strokes=s['total_strokes'],seconds=time.time()-start,diagnostics=d,stages=stages,widths_mm=WIDTHS,grid_um=50,time_s=s['time'],resume=str(resume) if resume else None)
 save_json(out/'run.json',result);save_json(out/'full-recipe.json',dict(recipe,operations=s['history'],settle_s=0))
 return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('task');a=ap.parse_args();run(json.loads(Path(a.task).read_text()))
