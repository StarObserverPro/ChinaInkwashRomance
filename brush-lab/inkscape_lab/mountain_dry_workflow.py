"""C00 controlled workflow experiment: same R1 geometry, less water, dry layers.

Dry-layer limits are explicitly labelled; no prediction of evaporation time or
redistribution during drying is made. This is a fresh rendering, not an edit to
the R2 bitmap and not an additional Monte Carlo search.
"""
import json,copy
from .landscape_engine import ROOT
from .mountain_driver import compile_recipe,prepare,metrics
from .mountain_session import run

def main():
 row=json.loads((ROOT/'references19/manifest.json').read_text())[0]
 src=ROOT/'driver19/r1/C00';out=ROOT/'driver19/r3/C00';out.mkdir(parents=True,exist_ok=True)
 plan=copy.deepcopy(json.loads((src/'plan.json').read_text()))
 plan.update(attempts=0,loss=None,source_plan='r1/C00',proxy_valid=False,
  experiment='Same R1 paths. Non-wash water x0.65 and concentration /0.65. Dry-layer limit after at most 12 strokes and at phase boundaries. Fresh paper; no drying-time prediction.')
 for s in plan['strokes']:
  if s['family']!='wash':s['water']*=.65;s['concentration']/=.65
  s['reason']+='；干层对照：减少供水，干后复画；干燥时间未模拟'
 recipe=compile_recipe(plan,row);ops=[];count=0
 for op in recipe['operations']:
  ops.append(op)
  if op['kind']=='stroke':
   count+=1
   if count==12 or op.get('checkpoint'):
    ops.append(dict(kind='dry_limit',reason='Explicit dry-layer approximation; water to vapor ledger, pigment fixed in place'))
    count=0
 recipe['operations']=ops;recipe['workflow']=plan['experiment']
 task=dict(recipe=recipe,round=3,output='driver19/r3/C00/physical')
 for n,obj in [('plan.json',plan),('recipe.json',recipe),('task.json',task)]:
  (out/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 run(task);m,_=metrics(prepare(row),out/'physical');(out/'metrics.json').write_text(json.dumps(m,indent=2));print(m,flush=True)
if __name__=='__main__':main()
