"""Constrained action refinement. The accepted v0.16 contact/paper stays fixed.
No target image, alpha mask, or compositing operator enters the solver.
"""
from pathlib import Path
import copy,json,hashlib,argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from .stroke_reading import prepare,sample_path,render,build_task
from .stroke_trials import evaluate
from .technique_studies import frame
from .landscape_engine import ROOT

BASE={'turn':dict(width=1.5,pressure=1.55,water=1.45,concentration=.85,sustain=True,per_stroke={str(i):dict(pressure=.5,water=.85,concentration=.85,sustain=True) for i in [1,2]}), 'axe':dict(width=1.3,pressure=1.15,water=1.2,concentration=.85,sustain=True),'dab':dict(width=1.5,pressure=1.2,water=1.15,concentration=.8,sustain=True)}

def gesture(path,variant=1,case=''):
 p=np.array(path['points_mm']);d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];L=d[-1];vertex=d/L;f=path['family']
 if f=='turn':
  u0=np.array([0,.05,.30,.58,.70,.82,.94,1.]);depth=np.array([.105,.140,.14,.135,.118,.125,.13,.11]);tilt=np.full(8,18.);delta=35.;speed=28.;release=.012
 elif f=='axe':
  u0=np.array([0,.06,.18,.45,.70,.88,1.]);depth=np.array([.18,.29,.31,.26,.20,.10,.03]);tilt=np.array([69,71,71,66,52,32,18]);delta=75.;speed=31.;release=.012
 else:
  u0=np.array([0,.10,.28,.60,.85,1.]);depth=np.array([.15,.22,.225,.225,.205,.13]);tilt=np.array([65,67,67,67,65,58]);delta=25.;speed=24.;release=.008
  if variant>=2:delta=50.;speed=23.
  if case=='turn':depth*=.65;speed=19.
 u=np.unique(np.r_[np.linspace(0,1,25),u0,vertex]);q=np.c_[np.interp(u,vertex,p[:,0]),np.interp(u,vertex,p[:,1])]
 duration=max(.055,L/speed);t=u*duration
 compression=np.interp(u,u0,depth);tilts=np.interp(u,u0,tilt)
 # All hand-annotated corners are retained. Uniform nine-key sampling used
 # to shortcut the final turn. Pose still interpolates continuously.
 tangent=np.gradient(q,t,axis=0);angles=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))*180/np.pi
 if variant>=2:
  # Bounded feed-forward for the already present first-order tip response.
  # tau is a controller coefficient, not a new paper/brush physical constant.
  lead=tangent*.010;lim=.28 if f=='turn' else .22
  lead*=np.minimum(1,lim/np.maximum(np.linalg.norm(lead,axis=1),1e-9))[:,None]
  q=q+lead
  if f=='dab':compression*=1.12
  if f=='axe':compression*=.82
 k=[frame(0,*q[0],-.35,tilts[0],angles[0]+delta,0)]
 for a,xy,dd,tt,ang in zip(t,q,compression,tilts,angles):k.append(frame(.012+float(a),*xy,float(dd),float(tt),float(ang+delta),0))
 k.append(frame(.012+duration+release,*q[-1],-.40,tilts[-1],angles[-1]+delta,0))
 return dict(name=f'v021-{f}',keyframes=k)

def make_task(row,variant=1,seed=20):
 task=build_task(row,3,copy.deepcopy(BASE[row['id']]))
 task['seed']=seed;task['round']=variant;task['output']=f'stroke21/r{variant}/{row["id"]}/physical' if seed==20 else f'stroke21/holdout/{row["id"]}-s{seed}/physical'
 task['optimization_version']='0.21';task['controller_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 task['recipe']['controller_version']='0.21-'+str(variant)
 for i,(path,op) in enumerate(zip(row['paths'],task['recipe']['operations'])):
  op['gesture']=gesture(path,variant,row['id']);op['reading']['phases']=['边触边行','持续接触','按笔类提收'];op['reading']['limits']='Action hypothesis, not observed hand motion.'
  if row['id']=='turn' and i>0:op['tool']='detail'
 if row['id']=='turn':
  task['widths_mm']['detail']=.72
  if variant>=2:
   task['widths_mm']['fine']=.72
   op0=task['recipe']['operations'][0]['loading']
   op0['body_concentration_kg_m3']*=.80;op0['tip_concentration_kg_m3']*=.80
 task['parameters']['strategy']=dict(variant=variant,preserve_vertices=True,short_mark_axis_deg=50 if variant>=2 else 25,feed_forward_s=.010 if variant>=2 else 0,per_mark_tool=True)
 return task

def selected_task(row,seed=20):
    """Conservative provisional per-family selection after two-seed review."""
    if row['id']=='axe':
        task=build_task(row,3,copy.deepcopy(BASE['axe']))
        task['seed']=seed;task['output']=f'stroke21/selected/axe-s{seed}/physical'
        task['selection_reason']='v021 did not consistently improve this family; retain v020 R3.'
        return task
    task=make_task(row,2,seed)
    task['output']=f'stroke21/selected/{row["id"]}-s{seed}/physical'
    task['selection_reason']='Provisional v021 R2; manual reading and two-seed check only.'
    return task

def run_job(task,row):
 out=ROOT/task['output'];out.parent.mkdir(parents=True,exist_ok=True)
 (out.parent/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2))
 r=render(task);old=evaluate(task,row)
 from .stroke_score21 import evaluate21
 score=evaluate21(task,row)
 return dict(case=row['id'],output=task['output'],seconds=r['seconds'],old_score=old['loss'],measurements=score,seed=task['seed'])

def prewarm():
 from .meniscus import native_film
 from .paper_transport import native_paper
 from .mountain_paper_fast import entry_native
 native_film();native_paper();entry_native(.04,.11)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,default=1);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--seed',type=int,default=20);a=ap.parse_args()
 rows=prepare();prewarm();summary=[]
 with ProcessPoolExecutor(a.workers) as pool:
  fs=[pool.submit(run_job,make_task(row,a.round,a.seed),row) for row in rows]
  for f in as_completed(fs):v=f.result();summary.append(v);print(json.dumps(v,ensure_ascii=False),flush=True)
 path=ROOT/f'stroke21/r{a.round}-s{a.seed}-summary.json';path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
