"""Replay the canonical equations with an arithmetic-identical native cache.

The canonical solver signature and input checks remain intact. Implementation
provenance and exact adoption points are saved separately, as with a compiler
backend change. Reference and cached backends can resume the same states.
"""
import json,gzip,pickle,gc,time
from .landscape_engine import ROOT

def main():
 from . import mountain_paper,mountain_paper_fast
 hashes=mountain_paper_fast.verify_source_pair()
 adoption=[]
 for f in sorted((ROOT/'driver19').glob('r*/*/physical/progress-state.pkl.gz')):
  if (f.parent/'run.json').exists():continue
  with gzip.open(f,'rb') as q:s=pickle.load(q)
  adoption.append(dict(output=str(f.parent.relative_to(ROOT)),next_operation=s['next_operation'],total_strokes=s['total_strokes']))
  del s;gc.collect()
 record=dict(timestamp=time.time(),source_hashes=hashes,adoption=adoption,new_runs='All newly started sessions from this launcher use the cache.',canonical_signature='The existing signature identifies canonical equations, physical inputs and parent bytes. This file additionally identifies the equivalent implementation.')
 path=ROOT/'driver19/runtime-implementation.json'
 old=json.loads(path.read_text()) if path.exists() else [];old.append(record);path.write_text(json.dumps(old,ensure_ascii=False,indent=2))
 mountain_paper.entry_native=mountain_paper_fast.entry_native
 from .mountain_pipeline import main as pipeline
 pipeline()

if __name__=='__main__':main()
