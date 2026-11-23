"""Three authored structural strokes after the Zhao rock crop.

This is a small application check, not recovery of the artist's stroke order.
Only the hand paths below enter the renderer; no reference pixels are read.
"""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,json,time
import numpy as np
from .brush import InkCanvas
from .cun import ContactSettings
from .meniscus import MeniscusSettings,material
from .paper_transport import PaperSettings
from .reference_brush import ReferenceBrush,SurfaceContactSettings
from .form_driver import compile_form

def programs():
    # x, y, pressure, roll angle, tilt. Identical in the baseline and candidate.
    strokes=[('ridge',[(35,177,.32,1.0,1.1),(49,119,.45,1.05,1.1),(78,80,.28,1.30,1.0),
        (120,45,.44,1.48,.96),(157,26,.27,1.62,.83),(173,44,.49,1.1,1.12),
        (202,37,.33,1.34,1.04),(231,66,.25,1.60,.90),(251,124,0.,1.82,.60)]),
        ('inner fold',[(129,87,.22,.6,1.0),(151,112,.34,1.1,1.12),(129,145,.21,1.5,.95),(145,176,0.,1.85,.60)]),
        ('lower short edge',[(65,138,.19,1.1,.90),(99,162,.31,1.50,1.05),(122,164,0.,1.8,.65)])]
    result=[];previous=None
    for name,nodes in strokes:
        start=[nodes[0][0]-3,nodes[0][1]-2]
        actions=[]
        if previous:
            actions.append({'kind':'airmove','to':start,'duration':.20,'load':0.,'angle':nodes[0][3],'tilt':nodes[0][4]})
            initial=previous;current=previous['to'];start0=current
        else:
            initial={'load':0.,'angle':nodes[0][3],'tilt':nodes[0][4]};current=start;start0=start
        for i,(x,y,load,angle,tilt) in enumerate(nodes):
            before=start if i==0 else list(nodes[i-1][:2])
            duration=max(.05,float(np.linalg.norm(np.array([x,y])-before))/180.)
            actions.append({'kind':'touch' if i==0 else 'release' if i==len(nodes)-1 else 'turn',
                'to':[x,y],'duration':duration,'load':load,'angle':angle,'tilt':tilt,
                'form_event':name+' changes plane at the authored corner'})
        p={'tag':name,'start':start0,'initial':{k:initial[k] for k in ['load','angle','tilt']},
            'max_dt':1/200,'spatial_step':.6,'actions':actions}
        result.append(p);previous=actions[-1]
    return result

def run(output,config):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path(config).read_text());begin=time.time()
    programs_ = programs();drivers=[compile_form(p,[None]*len(p['actions'])) for p in programs_]
    c=InkCanvas(290,210,seed=505,paper_color=(245,240,225));c.density=c.density.astype(float)
    bs=replace(ContactSettings(width=25,bunches=48,initial_water=.35),**cfg.get('brush',{}))
    ps=replace(PaperSettings(),texture_model='fibers',heterogeneity=.25,anisotropy=.35,**cfg.get('paper',{}))
    mc=replace(MeniscusSettings(),**cfg.get('meniscus',{}));sc=SurfaceContactSettings(**cfg.get('surface_contact',{}))
    b=ReferenceBrush(c,bs,fluid=replace(material(),concentration_kg_m3=40.),meniscus=mc,paper=ps,surface_contact=sc,record_source=True)
    if cfg.get('match_initial_inventory'):
        r0=json.loads(Path('reference-rounds/r0/dry/run.json').read_text())
        target=r0['diagnostics']['initial_nl']*1e-12/.25*.35;ratio=target/b.volume.sum()
        assert np.all(b.volume*ratio<=b.capacity)
        b.volume*=ratio;b.mass*=ratio;b.water_initial=float(b.volume.sum());b.mass_initial=float(b.mass.sum());b._sync()
    frames=[];next_frame=0.;last=-1
    def capture(t,brush):
        nonlocal next_frame,last
        if t>=next_frame:frames.append(c.to_image());next_frame=t+.09
        if int(t)>last:last=int(t);c.save(out/'preview.png');print(json.dumps({'out':str(out),'t':t}),flush=True)
    for driver in drivers:b.draw_driver(driver,frame_callback=capture)
    b.save_source(out/'source.npz');b.relax(4,capture);b.substrate.check();diag=b.diagnostics()
    assert abs(diag['water_relative_error'])<1e-9 and abs(diag['pigment_relative_error'])<1e-9
    c.save(out/'result.png');b.substrate.save(out/'fields.npz')
    frames[0].save(out/'progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*10,duration=75,loop=0)
    data={'config':cfg,'programs':programs_,'brush':asdict(bs),'paper':asdict(ps),'surface_contact':asdict(sc),
        'diagnostics':diag,'elapsed_s':time.time()-begin,'note':'Authored three-stroke structural study, not a pixel reconstruction or known historical stroke order.'}
    (out/'run.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'out':str(out),'elapsed_s':data['elapsed_s'],'water_error':diag['water_relative_error'],'pigment_error':diag['pigment_relative_error']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--config',required=True)
    a=ap.parse_args();run(a.output,a.config)
