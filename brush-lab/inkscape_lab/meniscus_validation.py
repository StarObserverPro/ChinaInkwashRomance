"""Material conservation, speed response and refinement probes; no image masks."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time
import numpy as np
from .meniscus import MeniscusSettings,material,MeniscusBrush,BridgeTable
from .meniscus_studies import render,save_run
from .cun import ContactSettings
from .brush import InkCanvas

def config():
    return MeniscusSettings(irregular_roots=True,length_log_sd=.22,radius_log_sd=.12,distributed_sliding_contact=True)

def program(speed=300.,fine=False):
    pose={'angle':1.57,'tilt':1.05}
    actions=[{'kind':'touch','to':[55,65],'duration':.10,'load':.65,**pose},
      {'kind':'move','to':[180,65],'duration':125/speed,'load':.65,**pose},
      {'kind':'release','to':[205,59],'duration':.10,'load':0.,**pose},
      {'kind':'airmove','to':[215,59],'duration':.10,'load':0.,**pose}]
    return {'program':{'tag':'speed and refinement probe','start':[35,65],
        'initial':{'load':0.,**pose},'max_dt':.0025 if fine else .005,'spatial_step':.2 if fine else .4,'actions':actions},'curves':[None]*4}

def substrate_test():
    b=MeniscusBrush(InkCanvas(72,52),ContactSettings(width=12,bunches=12));yy,xx=np.indices((52,72));area=b.fluid.pixel_m**2
    source=np.exp(-((xx-36)**2+(yy-26)**2)/6);source=source/source.sum()*1e-13
    largest=0.
    for i in range(70):
        b.film_h+=source/area;b.film_mass+=source*(28 if i<35 else 14)
        b.paper_bounds=(20,33,30,43);b._surface(.0015)
        active=b.film_h>1e-12
        cmax=(b.film_mass[active]/b.film_h[active]/area).max() if active.any() else 0.
        largest=max(largest,cmax)
        assert cmax<=28.00001 and b.paper_mass.min()>-1e-21
    v=b.paper_water.sum()+b.film_h.sum()*area;m=b.paper_mass.sum()+b.film_mass.sum()
    assert abs(v/7e-12-1)<1e-10 and abs(m/1.47e-10-1)<1e-10
    table=BridgeTable();d=np.array([2.15,2.5,3.,4.]);A=np.array([.2,.5,1.2,4.])
    R,H,a,F,valid=table.lookup(d,A);t=a+.45
    reconstructed=2*R**2*(t-np.pi/2+.5*np.sin(2*t))+2*R*(np.sin(2*a+.45)-np.sin(.45))-(2*a-np.sin(2*a))
    error=np.max(abs(reconstructed[valid]-A[valid])/A[valid])
    assert error<.02
    return {'water_relative_error':float(v/7e-12-1),'pigment_relative_error':float(m/1.47e-10-1),
            'max_tracer_concentration_kg_m3':float(largest),'young_laplace_section_area_relative_error':float(error)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['slow','fast','fine','no_pores','substrate'],required=True);args=ap.parse_args()
    p=Path('meniscus-validation')/args.case;p.mkdir(parents=True,exist_ok=True)
    if args.case=='substrate':
        d=substrate_test();(p/'validation.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));return
    fine=args.case=='fine';mc=config()
    if fine:mc=replace(mc,max_dt_s=mc.max_dt_s/2)
    if args.case=='no_pores':mc=replace(mc,pores_enabled=False)
    prog=program(900 if args.case=='fast' else 300,fine)
    cfg=ContactSettings(width=32,bunches=48,initial_water=.48)
    begin=time.time();c,b,steps,frames,states=render([prog],cfg,mc,replace(material(),paper_permeability_m2=7e-16),(255,125))
    d=save_run(p,c,b,steps,frames,states,[prog])
    diag=b.diagnostics();assert abs(diag['water_relative_error'])<1e-9 and abs(diag['pigment_relative_error'])<1e-9
    assert c.density.min()>-1e-10 and np.isfinite(c.density).all()
    print(json.dumps({'case':args.case,'elapsed_s':time.time()-begin,**diag}),flush=True)

if __name__=='__main__':main()
