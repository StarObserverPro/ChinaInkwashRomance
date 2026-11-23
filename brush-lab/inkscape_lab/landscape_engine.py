"""v0.18 landscape session: persistent brush, paper and explicit preparation.

Technique names never enter the contact/transport solver. Presets compile to
SI-derived keyframes and material operations. Rendering has no texture stamps.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,math,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from PIL import Image
from .brush import InkCanvas
from .structural_brush import StructuralBrush
from .technique_controls import BrushLoading,InkMaterial,InkPaperInteraction
from .technique_studies import save_json,save_npz,frame

ROOT=Path(__file__).resolve().parents[1]

class LandscapeBrush(StructuralBrush):
    version='landscape-vocabulary-0.18.0'
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.rinsed_water=0.;self.rinsed_pigment=0.;self.preparation_log=[]

    def rinse_load(self,loading):
        """Off-paper rinse; record the removed fluid, then add fresh solution."""
        if self.hand is not None and self.compression_m>=0:raise ValueError('rinse requires lifted brush')
        for j in np.flatnonzero(self.neck_alive):self._release(int(j))
        removed_water=float(self.volume.sum()+sum(p['v'] for p in self.pores.values()))
        removed_pigment=float(self.mass.sum()+sum(p['m'] for p in self.pores.values()))
        self.rinsed_water+=removed_water;self.rinsed_pigment+=removed_pigment
        self.volume[:]=0;self.mass[:]=0
        for p in self.pores.values():p['v']=0.;p['m']=0.;p['wet']=False
        self.load_profile(loading)
        self.preparation_log.append(dict(kind='rinse_load',at_s=self.time,removed_water_nl=removed_water*1e12,removed_pigment_ng=removed_pigment*1e12,loading=asdict(loading)))

    def dry_limit(self):
        """Dry-layer limit, NOT a time-resolved evaporation simulation.

        All paper water exits to an explicit vapor ledger; remaining pigment
        becomes fixed at its current position. This omits drying redistribution
        and is used only for a visibly labelled accumulated-ink approximation.
        """
        self.flush_paper();p=self.substrate
        vapor=float(p.water.sum()+p.film.sum()*self.fluid.pixel_m**2)
        p.fixed+=p.fine+p.coarse;p.fixed[0]+=p.film_mass
        p.fine[:]=0;p.coarse[:]=0;p.film_mass[:]=0;p.water[:]=0;p.film[:]=0
        self.evaporated_water+=vapor;self._sync_paper()
        self.previous_film[:]=self.film_h;self.previous_mass[:]=self.film_mass
        self.preparation_log.append(dict(kind='dry_layer_limit',at_s=self.time,evaporated_nl=vapor*1e12,limit='No prediction of drying time or pigment redistribution during evaporation'))

    def diagnostics(self):
        d=super().diagnostics()
        if hasattr(self,'rinsed_water'):
            w=self.water_initial+self.water_added_si+self.paper_added_water;m=self.mass_initial+self.mass_added_si
            d['water_relative_error']+=self.rinsed_water/max(w,1e-30)
            d['pigment_relative_error']+=self.rinsed_pigment/max(m,1e-30)
            d['rinsed_water_nl']=self.rinsed_water*1e12;d['rinsed_pigment_ng']=self.rinsed_pigment*1e12
        return d

def connect_air(brush,spec):
    first=spec['keyframes'][0]
    if brush.hand is None:return
    x,y=brush.hand*brush.fluid.pixel_m*1e3
    to=np.array([first['x_mm'],first['y_mm']]);distance=np.linalg.norm(to-[x,y])
    if distance<1e-7:return
    a=frame(0,float(x),float(y),-.40,math.degrees(brush.tilt),math.degrees(brush.angle),math.degrees(brush.roll))
    b=dict(first,t_s=max(.02,float(distance)/160),compression_mm=-.40)
    brush.draw_gesture(dict(name='airmove',keyframes=[a,b]))

def fingerprint(recipe,config,seed,detail):
    h=hashlib.sha256(json.dumps([recipe,config,seed,detail],sort_keys=True).encode())
    for n in ['landscape_engine.py','structural_brush.py','technique_controls.py','structural_transfer.py']:
        h.update((Path(__file__).parent/n).read_bytes())
    return h.hexdigest()

def run(task):
    start=time.time();recipe=task['recipe'];seed=task.get('seed',0);detail=task.get('detail',1)
    cfg=json.loads((ROOT/'selected_v016.json').read_text());sig=fingerprint(recipe,cfg,seed,detail)
    out=ROOT/task['output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists() and not task.get('force'):
        old=json.loads((out/'run.json').read_text())
        if old.get('fingerprint')!=sig:raise ValueError('Cached recipe/solver differs; choose another output or --force')
        return old
    width,height=recipe.get('size_mm',[14.,10.]);pixel=50.
    canvas=InkCanvas(round(width*1000/pixel),round(height*1000/pixel),seed=505+seed,paper_color=(245,240,225));canvas.density=canvas.density.astype(float)
    technique=dict(cfg.get('technique',{}),max_step_s=.0015/detail,spatial_step_mm=.035/detail)
    brush=LandscapeBrush(canvas,structure=cfg['structure'],technique=technique,width_mm=recipe.get('width_mm',.85),
        loading=BrushLoading(**recipe['loading']),ink=InkMaterial(**recipe.get('ink',{})),interaction=InkPaperInteraction(**cfg.get('interaction',{})),
        paper_overrides=dict(cfg.get('paper',{}),seed=70713+seed),meniscus_overrides=dict(cfg.get('meniscus',{}),anatomy_seed=1212+seed),
        surface_contact_overrides=dict(seed=14009+seed))
    stages=[];images=[];events=[]
    for i,op in enumerate(recipe['operations']):
        kind=op['kind']
        if kind=='stroke':
            connect_air(brush,op['gesture']);brush.draw_gesture(op['gesture'])
        elif kind=='load':brush.load_profile(BrushLoading(**op['loading']))
        elif kind=='rinse_load':brush.rinse_load(BrushLoading(**op['loading']))
        elif kind=='wait':brush.relax(op['seconds'])
        elif kind=='prewet':brush.prewet(op['saturation'])
        elif kind=='dry_limit':brush.dry_limit()
        else:raise ValueError('Unknown landscape operation '+kind)
        if op.get('checkpoint'):
            brush.flush_paper();images.append(canvas.to_image());stages.append(dict(operation=i,label=op['checkpoint'],at_s=brush.time,water_nl=brush.substrate.totals()['water_m3']*1e12))
        events.append(dict(operation=i,kind=kind,at_s=brush.time,brush_water_nl=float(brush.volume.sum()*1e12)))
    lift=canvas.density.copy();brush.relax(recipe.get('settle_s',.35));brush.substrate.check();d=brush.diagnostics()
    if max(abs(d['water_relative_error']),abs(d['pigment_relative_error']))>1e-8:raise RuntimeError('Conservation failed')
    for a in [brush.volume,brush.mass,brush.neck_v,brush.neck_m,canvas.density]:
        if not np.isfinite(a).all() or a.min() < -1e-22:raise RuntimeError('Invalid inventory')
    canvas.to_image().save(out/'result.png')
    p=brush.substrate
    save_npz(out/'fields.npz',density=canvas.density,lift_density=lift,water=p.water,fixed=p.fixed,mobile=p.fine+p.coarse,film=p.film,contact_source=brush.contact_source,detached_source=brush.detached_source)
    if images:
        images.append(canvas.to_image());images[0].save(out/'stages.gif',save_all=True,append_images=images[1:],duration=700,loop=0)
        for i,im in enumerate(images[:-1]):im.save(out/f'stage-{i:02}.png')
    alpha=-np.expm1(-canvas.density);yy,xx=np.nonzero(alpha>.04)
    border=np.r_[alpha[:2,:].ravel(),alpha[-2:,:].ravel(),alpha[:,0:2].ravel(),alpha[:,-2:].ravel()]
    result=dict(version=brush.version,fingerprint=sig,id=recipe['id'],title=recipe['title'],family=recipe['family'],round=task['round'],seed=seed,detail=detail,
        recipe=recipe,config=cfg,diagnostics=d,events=events,checkpoints=stages,preparation=brush.preparation_log,
        descriptors=dict(ink_pixels=int((alpha>.04).sum()),mean_ink_opacity=float(alpha[alpha>.04].mean()) if len(xx) else 0.,peak_opacity=float(alpha.max()),border_opacity_max=float(border.max()),bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None),
        seconds=time.time()-start,output=str(out.relative_to(ROOT)),control_trace=brush.control_trace)
    save_json(out/'run.json',result);return result

def execute(tasks,workers=6):
    from .meniscus import native_film
    from .paper_transport import native_paper
    from .technique_paper import entry_native
    native_film();native_paper();entry_native(.1,.3)
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(run,t):t for t in tasks}
        for f in as_completed(jobs):
            t=jobs[f]
            try:
                r=f.result();results.append(r);print(json.dumps(dict(done=len(results),total=len(tasks),id=r['id'],seed=r['seed'],seconds=round(r['seconds'],1)),ensure_ascii=False),flush=True)
            except Exception:
                failures.append(dict(task=t,error=traceback.format_exc()));print(json.dumps(failures[-1],ensure_ascii=False),flush=True)
    return dict(results=results,failures=failures)

def main():
    from .landscape_catalog import catalog
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,default=0);p.add_argument('--ids',default='all');p.add_argument('--seeds',default='0');p.add_argument('--workers',type=int,default=6);p.add_argument('--detail',type=int,default=1);p.add_argument('--force',action='store_true');a=p.parse_args()
    recipes=catalog(a.round);ids=a.ids.split(',')
    if a.ids!='all':
        unknown=set(ids)-{r['id'] for r in recipes}
        if unknown:raise ValueError(str(unknown))
        recipes=[r for r in recipes if r['id'] in ids]
    tasks=[dict(round=a.round,recipe=r,seed=int(s),detail=a.detail,force=a.force,output=f'atlas18/r{a.round}/{r["id"]}/s{s}'+(f'-d{a.detail}' if a.detail!=1 else '')) for r in recipes for s in a.seeds.split(',')]
    out=ROOT/'atlas18';out.mkdir(exist_ok=True);tag=f'r{a.round}-'+(a.ids if a.ids!='all' else 'all')+'-s'+a.seeds+(f'-d{a.detail}' if a.detail!=1 else '')
    save_json(out/(tag+'-design.json'),dict(tasks=tasks))
    d=execute(tasks,a.workers);save_json(out/(tag+'-summary.json'),d)
    if d['failures']:raise SystemExit('Some simulations failed')

if __name__=='__main__':main()
