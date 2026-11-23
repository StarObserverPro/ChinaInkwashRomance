"""v0.17 sequential DOE, local/quasi-Monte-Carlo exploration, robustness blocks.

This is NOT TuRBO/GP optimization: uncalibrated proxy metrics do not replace
visual evaluation. Every next-round parent is chosen after viewing results.
"""
from pathlib import Path
from copy import deepcopy
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time,traceback
import numpy as np
from scipy.ndimage import map_coordinates,label
from scipy.stats import qmc
from PIL import Image,ImageDraw
from .monte_carlo_studies import ROOT,run_one,font
from .technique_studies import save_json

CASES=('leaf','dry','bone')
OUT=ROOT/'study17'

def assess(density):
    """Named artifact descriptors, never a scalar realism rating."""
    a=-np.expm1(-density);mask=a>.12
    labels,n=label(mask,np.ones((3,3),int))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    if n==0:return dict(empty=True)
    main=labels==sizes.argmax();yy,xx=np.nonzero(main)
    lo,hi=int(xx.min()),int(xx.max());width=[];positions=[];top=[];bottom=[]
    for x in range(lo,hi+1):
        ys=np.flatnonzero(main[:,x])
        if len(ys):positions.append(x);width.append(ys[-1]-ys[0]+1);top.append(ys[0]);bottom.append(ys[-1])
    width=np.asarray(width);positions=np.asarray(positions);t=(positions-lo)/max(hi-lo,1)
    mid=width[(t>.35)&(t<.65)];head=width[(t>.03)&(t<.20)]
    head_ratio=float(np.median(head)/max(np.median(mid),1)) if len(head) and len(mid) else 0.
    # Follow the actual ink envelope, then inspect white corridors inside it.
    x=np.linspace(lo+.12*(hi-lo),lo+.84*(hi-lo),120)
    y0=np.interp(x,positions,top);y1=np.interp(x,positions,bottom)
    v=np.linspace(.12,.88,21)[:,None]
    strip=map_coordinates(a,[y0[None,:]*(1-v)+y1[None,:]*v,np.broadcast_to(x,(21,120))],order=1,mode='nearest')
    white=strip<.20;longest=0
    for row in white:
        edges=np.flatnonzero(np.diff(np.r_[False,row,False].astype(int)))
        if len(edges):longest=max(longest,int(np.max(edges[1::2]-edges[::2])))
    outside=mask&~main;tail=outside.copy();tail[:,:round(lo+.72*(hi-lo))]=False
    return dict(empty=False,head_mid_width_ratio=head_ratio,white_corridor_fraction=longest/120,
                interior_white_fraction=float(white.mean()),interior_opacity=float(strip.mean()),
                satellite_opacity_fraction=float(a[outside].sum()/max(a[mask].sum(),1e-30)),
                tail_satellite_opacity_fraction=float(a[tail].sum()/max(a[mask].sum(),1e-30)),
                total_density=float(density.sum()),ink_length_px=int(hi-lo),ink_area_px=int(mask.sum()))

def simulate(task):
    d=run_one(task)
    out=ROOT/task['output']
    with np.load(out/'fields.npz') as a:m=assess(a['density'])
    d['descriptors']=m;d['experiment_note']=task.get('note','');d['sampling_scope']=task.get('scope','')
    save_json(out/'run.json',d)
    return d

def execute(tasks,workers=6):
    from .meniscus import native_film
    from .paper_transport import native_paper
    from .technique_paper import entry_native
    native_film();native_paper();entry_native(.1,.3)
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(simulate,t):t for t in tasks}
        for f in as_completed(jobs):
            t=jobs[f]
            try:
                d=f.result();results.append(d)
                print(json.dumps(dict(done=len(results),total=len(tasks),id=t['id'],case=t['case'],seed=t.get('seed',0),seconds=round(d['seconds'],1))),flush=True)
            except Exception:
                failures.append(dict(task=t,error=traceback.format_exc()));print(json.dumps(failures[-1]),flush=True)
    return results,failures

def tasks_for(stage,variants,cases=CASES,seeds=(0,)):
    tasks=[]
    for sid,cfg,note,scope in variants:
        for seed in seeds:
            for case in cases:
                tasks.append(dict(id=sid,config=cfg,case=case,seed=seed,output=f'study17/{stage}/{sid}/s{seed}/{case}',note=note,scope=scope))
    save_json(OUT/f'{stage}_design.json',dict(stage=stage,variants=[dict(id=sid,config=cfg,note=n,scope=s) for sid,cfg,n,s in variants],tasks=tasks))
    return tasks

def config(stage,sid):return json.loads((OUT/f'{stage}_design.json').read_text())['variants'][next(i for i,v in enumerate(json.loads((OUT/f'{stage}_design.json').read_text())['variants']) if v['id']==sid)]['config']

def design_a():
    base=json.loads((ROOT/'selected_v016.json').read_text());variants=[]
    for i,(axis,rise,fiber) in enumerate(product([0.,1.],[0.,.12],[0.,.35])):
        cfg=deepcopy(base);cfg['structure'].update(axis_blend=axis,contact_rise=rise,fiber_coupling=fiber)
        variants.append((f'A{i:02d}',cfg,'2^3 matched factorial; tuft elevation and all material parameters retained','controlled'))
    return tasks_for('A',variants)

def design_b(parent):
    base=config('A',parent);s=base['structure'];r0=base['meniscus']['neck_radius_m']
    u=qmc.Sobol(7,scramble=True,seed=170932).random_base2(4);variants=[]
    for i,r in enumerate(u):
        cfg=deepcopy(base);p=cfg['structure'];wide=i>=12
        if not wide:
            p['axis_blend']=float(np.clip(s['axis_blend']+(r[0]-.5)*.55,0,1))
            p['contact_rise']=float(np.clip(s['contact_rise']+(r[1]-.5)*.16,0,.35))
            p['relief_um']=float(s['relief_um']*(.8+.45*r[2]))
            p['correlation_mm']=float(s['correlation_mm']*(.75+.6*r[3]))
            p['contact_transition_um']=float(2.5+4*r[4])
            cfg['meniscus']['neck_radius_m']=float(r0*(.88+.27*r[5]))
            p['fiber_coupling']=float(np.clip(s['fiber_coupling']+(r[6]-.5)*.3,0,.65))
        else:
            p['axis_blend']=float(r[0]);p['contact_rise']=float(.35*r[1])
            p['relief_um']=float(8+22*r[2]);p['correlation_mm']=float(.05+.2*r[3])
            p['contact_transition_um']=float(1+11*r[4]);cfg['meniscus']['neck_radius_m']=float((2+1.8*r[5])*1e-6)
            p['fiber_coupling']=float(.65*r[6])
        variants.append((f'B{i:02d}',cfg,f'Parent {parent}; scrambled Sobol seed 170932', 'wide' if wide else 'local'))
    return tasks_for('B',variants)

def design_c(ids):
    base=json.loads((ROOT/'selected_v016.json').read_text())
    variants=[('C06',base,'Accepted v0.16 reference','reference')]
    for sid in ids.split(','):variants.append((sid,config('B',sid),'Multi-seed robustness block; fixed parameters','shortlist'))
    return tasks_for('C',variants,seeds=(11,37))

def design_validation(sid):
    cfg=config('B',sid);save_json(ROOT/'candidate_v017.json',cfg)
    base=json.loads((ROOT/'selected_v016.json').read_text())
    variants=[('C06',base,'v0.16 matched reference','reference'),('selected',cfg,f'Chosen {sid}; no further tuning','validation')]
    tasks=tasks_for('validation',variants,cases=('fold','side','rock'))
    for factor in [2,4]:
        for case in CASES:
            c=deepcopy(cfg);c['technique']=dict(max_step_s=.0015/factor,spatial_step_mm=.035/factor)
            tasks.append(dict(id=f'dt{factor}',case=case,config=c,seed=0,output=f'study17/validation/dt{factor}/s0/{case}',note='Time step sensitivity',scope='numerical'))
    for case in ['dry','leaf']:
        tasks.append(dict(id='seed83',case=case,config=deepcopy(cfg),seed=83,output=f'study17/validation/seed83/s83/{case}',note='Fresh seed after selection',scope='holdout'))
    for name,c in [('C06',base),('selected',cfg)]:
        tasks.append(dict(id=name,case='dry',config=deepcopy(c),controls={'concentration':2.5},output=f'study17/validation/{name}/dense/dry',note='2.5x pigment, same viscosity',scope='dense'))
    save_json(OUT/'validation_design.json',dict(selected=sid,tasks=tasks));return tasks

def make_sheet(stage,seed=0):
    groups=defaultdict(dict)
    for p in (OUT/stage).glob(f'*/s{seed}/*/run.json'):
        d=json.loads(p.read_text());groups[d['id']][d['case']]=p
    cols=4;cw=350;ch=356;keys=sorted(groups);rows=int(np.ceil(len(keys)/cols))
    im=Image.new('RGB',(cw*cols,rows*ch+65),'#f4f1e8');draw=ImageDraw.Draw(im)
    draw.text((18,18),f'v0.17 / round {stage} / seed {seed} / leaf, dry, bone',font=font(23),fill='#223c2e')
    for i,sid in enumerate(keys):
        x=(i%cols)*cw;y=65+(i//cols)*ch;d=json.loads(next(iter(groups[sid].values())).read_text());s=d['config']['structure']
        draw.text((x+10,y),f"{sid}  axis={s.get('axis_blend',0):.2f} rise={s.get('contact_rise',0):.2f}",font=font(15),fill='#223c2e')
        draw.text((x+10,y+21),f"fiber={s.get('fiber_coupling',0):.2f}  {d.get('sampling_scope','')}",font=font(13),fill='#58685b')
        for j,case in enumerate(CASES):
            if case in groups[sid]:
                a=Image.open(groups[sid][case].parent/'result.png').convert('RGB');a.thumbnail((330,95));im.paste(a,(x+10,y+46+j*98))
        draw.line((x,y+ch-4,x+cw,y+ch-4),fill='#c9d0c5')
    path=ROOT/'review17'/f'{stage}-seed{seed}.png';im.save(path);return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',choices=['A','B','C','validation'],required=True);p.add_argument('--parent');p.add_argument('--ids');p.add_argument('--workers',type=int,default=6);p.add_argument('--sheet',action='store_true');p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    OUT.mkdir(exist_ok=True);(ROOT/'review17').mkdir(exist_ok=True)
    if a.sheet:print(make_sheet(a.round,a.seed));raise SystemExit
    tasks=design_a() if a.round=='A' else design_b(a.parent) if a.round=='B' else design_c(a.ids) if a.round=='C' else design_validation(a.parent)
    results,failures=execute(tasks,a.workers);save_json(OUT/f'{a.round}_summary.json',dict(results=results,failures=failures))
    if failures:raise SystemExit(f'{len(failures)} failed runs')
    if a.round in ['A','B']:make_sheet(a.round)
