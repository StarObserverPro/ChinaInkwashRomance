from pathlib import Path
import json,copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from inkscape_lab.stroke_opt21 import prepare,prewarm,make_task,run_job,ROOT
if __name__=='__main__':
 rows=prepare();prewarm();tasks=[]
 for row in rows:
  t=json.loads((ROOT/f'stroke20/r3/{row["id"]}/task.json').read_text());t['seed']=21;t['output']=f'stroke21/holdout/base-{row["id"]}-s21/physical';tasks.append((t,row,'base'))
  t=make_task(row,2,21);tasks.append((t,row,'r2'))
 results=[]
 with ProcessPoolExecutor(3) as pool:
  fs={pool.submit(run_job,t,r):v for t,r,v in tasks}
  for f in as_completed(fs):
   r=f.result();r['variant']=fs[f];results.append(r);print(json.dumps(dict(case=r['case'],variant=r['variant'],seconds=r['seconds'])),flush=True)
 (ROOT/'stroke21/holdout-summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
