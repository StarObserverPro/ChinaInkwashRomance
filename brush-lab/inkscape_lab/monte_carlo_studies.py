"""Seeded, balanced structural Monte Carlo with retained failure cases.

Screening scores are hand-authored artifact proxies, NOT realism probabilities.
Human visual feedback selects the next round. Renderer never reads references.
"""
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import product
import json,math,time,argparse,traceback,os
import numpy as np
from scipy.ndimage import label as components,binary_fill_holes,gaussian_filter
from PIL import Image,ImageDraw,ImageFont
from .brush import InkCanvas
from .technique_studies import recipe,save_json,save_npz
from .technique_controls import BrushLoading,InkMaterial,InkPaperInteraction,TechniqueBrush,TechniqueSettings
from .structural_brush import StructuralBrush

ROOT=Path(__file__).resolve().parents[1]

def rock_recipe():
    from .reference_rock_study import programs
    from .technique_studies import frame
    gestures=[]
    for old in programs():
        a=old['initial'];clock=0.;x,y=old['start']
        keys=[frame(0,x*.05,y*.05,-.32,math.degrees(a['tilt']),math.degrees(a['angle']))]
        for p in old['actions']:
            clock+=p['duration'];x,y=p['to'];depth=p['load']*.65 if p['load']>0 else -.4
            keys.append(frame(clock,x*.05,y*.05,depth,math.degrees(p['tilt']),math.degrees(p['angle'])))
        gestures.append(dict(name=old['tag'],keyframes=keys))
    return gestures,BrushLoading(body_saturation=.35,tip_saturation=.35,body_concentration_kg_m3=25,tip_concentration_kg_m3=65)

def metrics(density,case,pixel_um=50.):
    a=-np.expm1(-density);mask=a>.18
    labels,n=components(mask,np.ones((3,3),int))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    if not n:return dict(score=1000.,satellite_fraction=1.,satellites=0,white_fraction=1.,rail=1.,dark_fraction=0.,opacity=0.)
    main=labels==np.argmax(sizes);sat=mask&~main
    sat_fraction=float(a[sat].sum()/max(a[mask].sum(),1e-20))
    yy,xx=np.nonzero(mask);xmin,xmax=xx.min(),xx.max();ymin,ymax=yy.min(),yy.max()
    # A geometric envelope describes missing coverage, not a target ink mask.
    from scipy.spatial import ConvexHull
    pts=np.c_[xx,yy];hull=ConvexHull(pts)
    im=Image.new('1',(a.shape[1],a.shape[0]));ImageDraw.Draw(im).polygon([tuple(map(int,p)) for p in pts[hull.vertices]],fill=1)
    envelope=np.asarray(im,bool)
    white=envelope&(a<.25);white_fraction=float(white.sum()/max(envelope.sum(),1))
    dark_fraction=float((a[envelope]>.88).mean())
    # Long nearly horizontal white runs flag rail-like corridors in dry probe.
    longest=0
    inner=white.copy();inner[:,:int(xmin+.15*(xmax-xmin))]=False;inner[:,int(xmin+.85*(xmax-xmin)):]=False
    for row in inner:
        ends=np.flatnonzero(np.diff(np.r_[False,row,False].astype(int)))
        if len(ends):longest=max(longest,int(np.max(ends[1::2]-ends[::2])))
    rail=longest/max(xmax-xmin,1)
    opacity=float(a[envelope].mean())
    dots=int(sum(1 for x in sizes[1:] if 1<=x<=18))
    if case=='leaf':
        score=12*sat_fraction+.018*min(dots,40)+3*max(0,.43-opacity)+2*max(0,white_fraction-.35)
    else:
        score=10*sat_fraction+.012*min(dots,40)+1.5*rail+3*max(0,.23-opacity)+2*max(0,dark_fraction-.6)+2*max(0,.10-white_fraction)
    return dict(score=float(score),satellite_fraction=sat_fraction,satellites=dots,white_fraction=white_fraction,
                rail=float(rail),dark_fraction=dark_fraction,opacity=opacity,ink_pixels=int(mask.sum()))

def run_one(task):
    out=Path(task['output'])
    if not out.is_absolute():out=ROOT/out
    elif 'mc' in out.parts:
        # Relocate retained designs when this project is unpacked elsewhere.
        out=ROOT/Path(*out.parts[out.parts.index('mc'):])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists() and not task.get('force',False):
        cached=json.loads((out/'run.json').read_text())
        if cached['config']!=task['config'] or cached['case']!=task['case'] or cached.get('seed',0)!=task.get('seed',0) or cached.get('controls',{})!=task.get('controls',{}) or cached.get('hold_s',.6)!=task.get('hold',.6):
            raise ValueError('Output contains a different experiment; choose a new output or --force')
        return cached
    begin=time.time();cfg=task['config'];case=task['case'];seed=task.get('seed',0)
    if case=='rock':specs,loading=rock_recipe()
    else:
        spec,loading=recipe(case);specs=[spec]
    controls=task.get('controls',{})
    for spec in specs:
        for k in spec['keyframes']:
            k['t_s']/=controls.get('speed',1.)
            if k['compression_mm']>0:k['compression_mm']*=controls.get('compression',1.)
    if controls.get('water',1)!=1:
        loading=replace(loading,body_saturation=loading.body_saturation*controls['water'],tip_saturation=loading.tip_saturation*controls['water'])
    if controls.get('concentration',1)!=1:
        loading=replace(loading,body_concentration_kg_m3=loading.body_concentration_kg_m3*controls['concentration'],
                        tip_concentration_kg_m3=loading.tip_concentration_kg_m3*controls['concentration'])
    technique=dict(cfg.get('technique',{}));pixel=technique.get('pixel_size_um',50.)
    dimensions=(14500,10500) if case=='rock' else (14250,5600)
    canvas=InkCanvas(round(dimensions[0]/pixel),round(dimensions[1]/pixel),seed=505+seed,paper_color=(245,240,225));canvas.density=canvas.density.astype(float)
    paper=dict(cfg.get('paper',{}),seed=70713+seed)
    meniscus=dict(cfg.get('meniscus',{}),anatomy_seed=1212+seed)
    surface=dict(cfg.get('surface',{}),seed=14009+seed)
    constructor=TechniqueBrush if task.get('backend')=='legacy' else StructuralBrush
    extra=dict(technique=TechniqueSettings(**technique)) if task.get('backend')=='legacy' else dict(structure=cfg['structure'],technique=technique)
    brush=constructor(canvas,loading=loading,**extra,width_mm=1.25 if case=='rock' else 1.45,
        ink=InkMaterial(**cfg.get('ink',{})),interaction=InkPaperInteraction(**cfg.get('interaction',{})),
        paper_overrides=paper,meniscus_overrides=meniscus,surface_contact_overrides=surface,record_source=False)
    frames=[];next_frame=0.
    def capture(t,b):
        nonlocal next_frame
        if task.get('animation') and t>=next_frame:
            frames.append(canvas.to_image());next_frame=t+.05
    for spec in specs:brush.draw_gesture(spec,capture)
    lift=canvas.density.copy();brush.relax(task.get('hold',.6),capture)
    brush.substrate.check();diag=brush.diagnostics()
    if max(abs(diag['water_relative_error']),abs(diag['pigment_relative_error']))>1e-8:raise ValueError('conservation failed')
    for a in [brush.volume,brush.mass,brush.neck_v,brush.neck_m,canvas.density]:
        if not np.isfinite(a).all() or a.min() < -1e-22:raise ValueError('nonfinite or negative inventory')
    canvas.to_image().save(out/'result.png')
    source={k:getattr(brush,k) for k in ['release_source','contact_source','detached_source'] if hasattr(brush,k)}
    save_npz(out/'fields.npz',density=canvas.density,lift_density=lift,**source,
        water=brush.substrate.water.sum(axis=0),film=brush.film_h)
    if frames:
        frames[0].save(out/'progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*8,duration=80,loop=0)
    m=metrics(canvas.density,case,pixel)
    data=dict(id=task['id'],case=case,seed=seed,config=cfg,gestures=specs,loading=asdict(loading),controls=controls,
              diagnostics=diag,metrics=m,seconds=time.time()-begin,output=str(out),version=brush.version,
              hold_s=task.get('hold',.6),evaluation='Artifact proxies only; no calibrated similarity or likelihood.')
    save_json(out/'run.json',data)
    return data

def latin_hypercube(n,d,seed):
    rng=np.random.default_rng(seed)
    return np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in range(d)])

def generate_global():
    (ROOT/'mc').mkdir(exist_ok=True)
    rng=np.random.default_rng(160930);u=latin_hypercube(48,12,160930)
    designs=list(product(['laid','arc'],['area','series'],['instant','finite'],['cap','memory','drain'],['independent','shared']))
    rng.shuffle(designs);tasks=[];configs=[]
    log=lambda a,b,x:float(a*(b/a)**x)
    for i,(geo,flux,supply,release,relief) in enumerate(designs):
        r=u[i]
        cfg=dict(structure=dict(geometry=geo,flux=flux,supply=supply,release=release,relief=relief,
                    relief_um=float(10+18*r[0]),correlation_mm=float(.08+.14*r[1]),supply_fraction=float(.12+.24*r[2])),
                 meniscus=dict(young_pa=log(1.5e9,3.8e9,r[3]),friction_coefficient=float(.2+.6*r[4]),
                    mechanical_tau_s=log(.006,.02,r[5]),neck_radius_m=log(1.2e-6,2.4e-6,r[6]),
                    contact_permeability_m2=log(.6e-16,3e-16,r[7]),body_feed_scale=log(.55,1.8,r[8])),
                 paper=dict(inplane_permeability_m2=log(3e-14,3e-13,r[9]),depth_permeability_m2=log(3e-16,1.5e-15,r[10])),
                 interaction=dict(entry_capture_coarse=float(.85*r[11]),entry_capture_fine=float(.32*r[11])))
        sid=f'M{i:02d}';configs.append(dict(id=sid,config=cfg))
        for case in ['leaf','dry']:
            tasks.append(dict(id=sid,config=cfg,case=case,output=str(ROOT/'mc/global'/sid/case)))
    save_json(ROOT/'mc/global_design.json',dict(seed=160930,method='Balanced 2x2x2x3x2 structural design + randomized Latin hypercube in 12 continuous dimensions',configs=configs,tasks=tasks))
    return tasks

def execute(tasks,workers):
    # Precompile native libraries before multiprocessing.
    from .meniscus import native_film
    from .paper_transport import native_paper
    from .technique_paper import entry_native
    native_film();native_paper();entry_native(.1,.3)
    results=[];failed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for f in as_completed(futures):
            t=futures[f]
            try:
                d=f.result();results.append(d)
                print(json.dumps(dict(done=len(results),total=len(tasks),id=d['id'],case=d['case'],seconds=round(d['seconds'],1),score=round(d['metrics']['score'],3))),flush=True)
            except Exception:
                failure=dict(task=t,error=traceback.format_exc());failed.append(failure);print(json.dumps(failure),flush=True)
    return results,failed

def font(size=15):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def sheet(folder,output,title,cols=4,sort=False):
    paths=list(Path(folder).glob('*/*/run.json'));groups={}
    for p in paths:
        d=json.loads(p.read_text());groups.setdefault(d['id'],{})[d['case']]=d
    keys=sorted(groups,key=lambda k:sum(v['metrics']['score'] for v in groups[k].values())/len(groups[k]) if sort else k)
    cw,ch=360,245;rows=math.ceil(len(keys)/cols)
    im=Image.new('RGB',(cols*cw,rows*ch+58),'#f5f2e9');draw=ImageDraw.Draw(im)
    draw.text((16,15),title,fill='#192423',font=font(20))
    for i,k in enumerate(keys):
        x=(i%cols)*cw;y=58+(i//cols)*ch;g=groups[k];d=next(iter(g.values()));s=d['config']['structure']
        draw.text((x+10,y),f"{k}  {s['geometry']}/{s['flux']}  {s['supply']}",fill='#182421',font=font(14))
        draw.text((x+10,y+19),f"{s['release']}  {s['relief']}  C={d['config']['interaction']['entry_capture_coarse']:.2f}",fill='#48534e',font=font(13))
        for j,case in enumerate(['leaf','dry']):
            if case not in g:continue
            p=Path(folder)/k/case/'result.png';a=Image.open(p).convert('RGB');a.thumbnail((340,96))
            im.paste(a,(x+10,y+41+j*98))
        draw.line((x,y+ch-3,x+cw,y+ch-3),fill='#c7ccc5')
    im.save(output)
    return keys

if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--design');group.add_argument('--generate',action='store_true');group.add_argument('--config')
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int);ap.add_argument('--force',action='store_true')
    ap.add_argument('--case',default='leaf',choices=['leaf','dry','pulse','side','fold','bone','rock'])
    ap.add_argument('--output',default='mc/custom');ap.add_argument('--animation',action='store_true');a=ap.parse_args()
    if a.generate:tasks=generate_global()
    elif a.config:tasks=[dict(id='custom',case=a.case,config=json.loads(Path(a.config).read_text()),output=a.output,animation=a.animation)]
    else:tasks=json.loads(Path(a.design).read_text())['tasks']
    if a.limit:tasks=tasks[:a.limit]
    if a.force:tasks=[dict(t,force=True) for t in tasks]
    results,failed=execute(tasks,a.workers)
    (ROOT/'mc').mkdir(exist_ok=True)
    summary=ROOT/'mc'/('global_summary.json' if a.generate else (Path(a.design).stem if a.design else 'custom')+'_summary.json')
    save_json(summary,dict(runs=results,failed=failed))
    if failed:raise SystemExit(f'{len(failed)} failed runs')
