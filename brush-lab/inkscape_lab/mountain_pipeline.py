"""Batch copies; every row uses one fixed policy and its own reference crop."""
import json,argparse,traceback,hashlib,fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .landscape_engine import ROOT
from .mountain_driver import prepare,search,save_plan,compile_recipe,metrics,VERSION
from .mountain_session import run
def render_plan(row,round_id):
 out=ROOT/f'driver19/r{round_id}/{row["id"]}';recipe=json.loads((out/'recipe.json').read_text());task=dict(recipe=recipe,round=round_id,output=str((out/'physical').relative_to(ROOT)))
 (out/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2));r=run(task);m,D=metrics(prepare(row),out/'physical');(out/'metrics.json').write_text(json.dumps(m,indent=2));return r,m,D
def feedback(row,previous=1,round_id=2,budget=28):
 base=ROOT/f'driver19/r{previous}/{row["id"]}';out=ROOT/f'driver19/r{round_id}/{row["id"]}';A=prepare(row);before,D=metrics(A,base/'physical')
 if (out/'plan.json').exists():plan=json.loads((out/'plan.json').read_text())
 else:
  plan,proxy=search(A,seed=10301+round_id*131+int(row['id'][1:]),budget=budget,branches=128,depth=5,initial=D)
  plan.update(feedback_from=previous,physical_before=before);save_plan(out,A,plan,proxy)
 recipe=json.loads((out/'recipe.json').read_text());task=dict(recipe=recipe,round=round_id,output=str((out/'physical').relative_to(ROOT)),resume=str((base/'physical/state.pkl.gz').relative_to(ROOT)))
 (out/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2));r=run(task);after,_=metrics(A,out/'physical')
 (out/'metrics.json').write_text(json.dumps(after,indent=2));review=dict(previous=previous,round=round_id,before=before,after=after,loss_improved=after['loss']<before['loss'],additive_only=True,visual_review_required=True)
 (out/'feedback.json').write_text(json.dumps(review,ensure_ascii=False,indent=2));return review
def _case(row,round_id=1,with_feedback=True):
 out=ROOT/f'driver19/r{round_id}/{row["id"]}';A=prepare(row);budget=int(np.clip(round(82+150*A['alpha'].mean()),96,132));bank=[]
 if not (out/'plan.json').exists():
  for seed in [7019,8021]:
   bp=ROOT/f'driver19/search/{row["id"]}/s{seed}'
   if (bp/'plan.json').exists():p=json.loads((bp/'plan.json').read_text());d=np.load(bp/'planning.npz')['proxy_density']
   else:p,d=search(A,seed=seed,budget=budget,branches=144,depth=5);save_plan(bp,A,p,d)
   bank.append((p,d))
  p,d=min(bank,key=lambda x:x[0]['loss']);save_plan(out,A,p,d)
 r,m,D=render_plan(row,round_id);result=dict(id=row['id'],round=round_id,metrics=m,seconds=r['seconds'])
 if with_feedback:result['feedback']=feedback(row,round_id,round_id+1,28)
 return result
def locked_case(row,fn,*args):
 locks=ROOT/'driver19/locks';locks.mkdir(exist_ok=True)
 with open(locks/(row['id']+'.lock'),'a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  return fn(row,*args)
def case(row,round_id=1,with_feedback=True):
 return locked_case(row,_case,round_id,with_feedback)
def feedback_task(row,previous,round_id,budget):
 return locked_case(row,feedback,previous,round_id,budget)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ids',default='all');ap.add_argument('--workers',type=int,default=7);ap.add_argument('--feedback-only',action='store_true');ap.add_argument('--previous',type=int,default=1);ap.add_argument('--round',type=int,default=2);ap.add_argument('--budget',type=int,default=28);ap.add_argument('--no-feedback',action='store_true');a=ap.parse_args()
 rows=json.loads((ROOT/'references19/manifest.json').read_text());rows=[r for r in rows if (r['id']!='C00' if a.ids=='all' else r['id'] in a.ids.split(','))]
 from .meniscus import native_film
 from .mountain_paper import entry_native
 native_film();entry_native(.04,.11)
 meta=dict(version=VERSION,selection='2 surrogate seeds per painting; common rules, no artist lookup',not_blind_holdout=True,physics_grid_um=50,budget='clip(82+150*mean normalized opacity,96,132)',files={n:hashlib.sha256((ROOT/'inkscape_lab'/n).read_bytes()).hexdigest() for n in ['mountain_policy.py','mountain_driver.py','mountain_session.py','mountain_paper.cpp']})
 (ROOT/'driver19/frozen-policy.json').write_text(json.dumps(meta,indent=2));result={'results':[],'failures':[]}
 summary=ROOT/('driver19/feedback-batch-summary.json' if a.feedback_only else 'driver19/batch-summary.json')
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  jobs={pool.submit(feedback_task,r,a.previous,a.round,a.budget) if a.feedback_only else pool.submit(case,r,1,not a.no_feedback):r['id'] for r in rows}
  for f in as_completed(jobs):
   try:result['results'].append(dict(id=jobs[f],result=f.result()));print('CASE COMPLETE '+jobs[f],flush=True)
   except Exception:result['failures'].append(dict(id=jobs[f],error=traceback.format_exc()));print(result['failures'][-1],flush=True)
   summary.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 if result['failures']:raise SystemExit('Some cases failed; use existing per-stroke checkpoints to resume')
if __name__=='__main__':main()
