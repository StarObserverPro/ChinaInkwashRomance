"""Paired proxy-only ablation; never a claim of physical artistic quality."""
import json
from .landscape_engine import ROOT
from .mountain_driver import prepare,search,save_plan
def main():
 row=json.loads((ROOT/'references19/manifest.json').read_text())[0];A=prepare(row);results=[]
 for seed in [1901,1902,1903]:
  for name,neutral,depth,branches in [('tutorial',False,5,128),('neutral_text',True,5,128),('shallow',False,1,93)]:
   out=ROOT/f'driver19/ablation/{name}-s{seed}'
   if (out/'plan.json').exists():plan=json.loads((out/'plan.json').read_text())
   else:plan,D=search(A,seed,100,branches,depth,neutral=neutral);save_plan(out,A,plan,D)
   results.append(dict(method=name,seed=seed,loss=plan['loss'],attempts=plan['attempts'],strokes=len(plan['strokes']),seconds=plan['seconds']))
   (ROOT/'driver19/ablation-summary.json').write_text(json.dumps(results,indent=2));print(results[-1],flush=True)
if __name__=='__main__':main()
