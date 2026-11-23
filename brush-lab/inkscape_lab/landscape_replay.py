"""Render a user-editable exported landscape preset with the shared solver."""
from pathlib import Path
import argparse,json
from .landscape_engine import run
from .technique_controls import compile_gesture

def main():
    p=argparse.ArgumentParser();p.add_argument('preset');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--detail',type=int,default=1);p.add_argument('--output');p.add_argument('--force',action='store_true');a=p.parse_args()
    r=json.loads(Path(a.preset).read_text())
    if a.detail not in (1,2,4):raise ValueError('detail must be 1, 2 or 4')
    for op in r['operations']:
        if op['kind']=='stroke':compile_gesture(op['gesture'])
    target=a.output or f'atlas18/custom/{r["id"]}/s{a.seed}-d{a.detail}'
    x=run(dict(recipe=r,round=-1,seed=a.seed,detail=a.detail,output=target,force=a.force))
    print(json.dumps(dict(output=x['output'],seconds=x['seconds'],water_error=x['diagnostics']['water_relative_error'],pigment_error=x['diagnostics']['pigment_relative_error']),ensure_ascii=False))

if __name__=='__main__':main()
