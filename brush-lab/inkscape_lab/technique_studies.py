"""Reproducible technique probes and safely published study outputs."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,os,time,tempfile
import numpy as np
from PIL import Image
from .brush import InkCanvas
from .technique_controls import TechniqueBrush,TechniqueSettings,BrushLoading,InkMaterial,InkPaperInteraction
from .reference_evidence import save_png
from .paper_studies import label

ROOT=Path(__file__).resolve().parents[1]

def save_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save_npz(path,**data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(tmp,**data)
    with np.load(tmp) as a:
        for k in a.files:assert np.isfinite(a[k]).all(),k
    os.replace(tmp,path)

def frame(t,x,y,d,tilt=50,azimuth=85,roll=0):
    return dict(t_s=t,x_mm=x,y_mm=y,compression_mm=d,tilt_deg=tilt,azimuth_deg=azimuth,roll_deg=roll)

def recipe(name):
    base=BrushLoading(body_saturation=.40,tip_saturation=.40)
    if name=='pulse':
        k=[frame(0,1.4,2.6,-.25),frame(.05,1.7,2.6,.055),frame(.16,4.2,2.6,.095),
           frame(.30,7.0,2.6,.38),frame(.43,9.7,2.6,.10),frame(.54,11.9,2.6,.025),frame(.60,12.5,2.6,-.25)]
        note='Light-heavy-light compression, at roughly steady travel speed.'
    elif name=='dry':
        base=replace(base,body_saturation=.20,tip_saturation=.20,body_concentration_kg_m3=38,tip_concentration_kg_m3=38)
        k=[frame(0,1.4,2.6,-.25,62),frame(.04,1.7,2.6,.20,62),frame(.19,5.0,2.5,.30,64,85,-10),
           frame(.35,8.3,2.9,.23,60,100,12),frame(.49,11.8,2.4,.13,56,85,0),frame(.55,12.5,2.3,-.25,45)]
        note='Lower liquid charge, higher concentration, with a small authored rotation.'
    elif name=='side':
        base=replace(base,body_concentration_kg_m3=7,tip_concentration_kg_m3=32,side_contrast=.85,side_axis_deg=90)
        k=[frame(0,1.6,3.5,-.25,62,60),frame(.05,1.9,3.4,.08,62,60),frame(.19,5.0,3.1,.36,70,70,0),
           frame(.34,8.4,2.5,.26,63,80,20),frame(.48,11.5,1.7,.015,35,95,30),frame(.54,12.2,1.5,-.25,30,95,30)]
        note='Unequal body/tip concentration and a cross-bundle concentration gradient.'
    elif name=='fold':
        base=replace(base,body_saturation=.29,tip_saturation=.29,body_concentration_kg_m3=30,tip_concentration_kg_m3=30)
        k=[frame(0,1.7,1.5,-.25,55,65),frame(.06,2.0,1.5,.20,55,65),frame(.24,5.9,1.5,.24,55,65),
           frame(.30,6.25,1.6,.30,62,105,25),frame(.47,6.5,3.6,.19,55,125,35),
           frame(.64,10.7,3.7,.19,55,75,5),frame(.71,11.9,3.7,-.25,35,75,5)]
        note='One continuous folded gesture; no brush/ink reset at either corner.'
    elif name=='leaf':
        base=replace(base,body_saturation=.60,tip_saturation=.60,body_concentration_kg_m3=12,tip_concentration_kg_m3=80,
            side_contrast=.55,side_axis_deg=45)
        k=[frame(0,1.5,3.6,-.28,62,125),frame(.04,1.85,3.5,.32,62,125),frame(.13,3.6,3.25,.42,60,120),
           frame(.30,6.9,2.65,.32,50,105,15),frame(.43,9.7,2.05,.12,35,90,25),
           frame(.50,11.3,1.7,.03,20,80,30),frame(.56,13.0,1.25,-.50,12,75,30)]
        note='Authored leaf exercise: press, travel, lift while moving faster; tip-rich loading. Not an inferred historical motion.'
    elif name=='bone':
        base=replace(base,body_saturation=.65,tip_saturation=.65,body_concentration_kg_m3=10,tip_concentration_kg_m3=55)
        k=[frame(0,1.5,2.65,-.28,62,85),frame(.055,1.9,2.65,.40,62,85),frame(.14,3.3,2.65,.22,50,85),
           frame(.42,9.0,2.65,.20,50,85),frame(.50,10.5,2.65,.33,60,85),frame(.58,11.8,2.65,-.45,25,85)]
        note='Authored bamboo bone-stroke exercise: press at each end, lighter travel in the middle.'
    else:raise ValueError('unknown technique probe')
    return {'name':name,'note':note,'keyframes':k},base

def run(output,round_id,name,*,speed=1.,compression_scale=1.,water_scale=1.,concentration_scale=1.,paper_scale=1.,hold=.6,config_path=None,preset_file=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);begin=time.time()
    cfg=json.loads(Path(config_path or ROOT/f'technique-r{round_id}-config.json').read_text())
    if min(speed,compression_scale,water_scale,concentration_scale,paper_scale)<=0 or hold<0:raise ValueError('invalid study scale')
    spec,loading=recipe(name)
    if preset_file:
        preset=json.loads(Path(preset_file).read_text());spec=preset['gesture'];loading=BrushLoading(**preset['loading']);name=spec.get('name','custom')
    for k in spec['keyframes']:
        k['t_s']/=speed
        if k['compression_mm']>0:k['compression_mm']*=compression_scale
    loading=replace(loading,body_saturation=loading.body_saturation*water_scale,tip_saturation=loading.tip_saturation*water_scale,
        body_concentration_kg_m3=loading.body_concentration_kg_m3*concentration_scale,
        tip_concentration_kg_m3=loading.tip_concentration_kg_m3*concentration_scale)
    pixel_um=cfg.get('technique',{}).get('pixel_size_um',50.)
    c=InkCanvas(round(14250/pixel_um),round(5600/pixel_um),seed=505,paper_color=(245,240,225));c.density=c.density.astype(float)
    paper=dict(cfg.get('paper',{}))
    paper['inplane_permeability_m2']=paper.get('inplane_permeability_m2',1e-13)*paper_scale
    b=TechniqueBrush(c,loading=loading,technique=TechniqueSettings(**cfg.get('technique',{})),
        paper_overrides=paper,ink=InkMaterial(**cfg.get('ink',{})),interaction=InkPaperInteraction(**cfg.get('interaction',{})),
        brush_overrides=cfg.get('brush',{}))
    frames=[];times=[];next_frame=0.;next_report=0.
    def capture(t,brush):
        nonlocal next_frame,next_report
        if t>=next_frame:
            im=c.to_image();label(im,(8,6),f'{t:.2f}s',11);frames.append(im);times.append(t);next_frame=t+.06
        if t>=next_report:
            save_png(c.to_image(),out/'preview.png');print(json.dumps({'out':str(out),'t':round(t,3)}),flush=True);next_report=t+.5
    b.draw_gesture(spec,capture)
    lift=c.density.copy();lift_time=b.time;save_png(c.to_image(),out/'at_lift.png')
    b.relax(hold,capture);b.substrate.check();diag=b.diagnostics()
    assert abs(diag['water_relative_error'])<1e-9 and abs(diag['pigment_relative_error'])<1e-9,diag
    save_png(c.to_image(),out/'result.png')
    p=b.substrate
    save_npz(out/'fields.npz',water=p.water,fine=p.fine,coarse=p.coarse,fixed=p.fixed,film=p.film,film_mass=p.film_mass,
        capacity=p.capacity,density=c.density,lift_density=lift,tip_trace=np.asarray(b.tip_trace))
    tmp=out/'source.tmp.npz';b.save_source(tmp)
    with np.load(tmp) as a:
        for key in a.files:assert np.isfinite(a[key]).all()
    os.replace(tmp,out/'source.npz')
    tmp=out/'progress.tmp.gif';frames[0].save(tmp,save_all=True,append_images=frames[1:]+[frames[-1]]*8,duration=80,loop=0)
    with Image.open(tmp) as im:
        for i in range(im.n_frames):im.seek(i);im.load()
    os.replace(tmp,out/'progress.gif')
    metadata={'version':b.version,'round':round_id,'case':name,'config':cfg,'gesture':spec,'loading':asdict(loading),
        'ink':asdict(b.ink_material),'interaction':asdict(b.ink_paper),'paper':asdict(p.settings),'fluid':asdict(b.fluid),
        'technique':asdict(b.technique),'brush':asdict(b.settings),'meniscus':asdict(b.meniscus),
        'diagnostics':diag,'lift_time_s':lift_time,'hold_s':hold,'times':times,'control_trace':b.control_trace,
        'elapsed_s':time.time()-begin,'seed_note':'All materials and initial preparation use fixed seeds; no time-noise texture.'}
    save_json(out/'run.json',metadata)
    save_json(out/'gesture.json',spec)
    print(json.dumps({'out':str(out),'seconds':metadata['elapsed_s'],'water_error':diag['water_relative_error'],
        'pigment_error':diag['pigment_relative_error'],'deposited_nl':diag['deposited_nl_including_rupture']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--round',type=int,default=0)
    ap.add_argument('--case',choices=['pulse','dry','side','fold','leaf','bone'],required=True)
    for k in ['speed','compression-scale','water-scale','concentration-scale','paper-scale']:ap.add_argument('--'+k,type=float,default=1.)
    ap.add_argument('--hold',type=float,default=.6);ap.add_argument('--config');ap.add_argument('--preset-file');a=ap.parse_args()
    run(a.output,a.round,a.case,speed=a.speed,compression_scale=a.compression_scale,water_scale=a.water_scale,
        concentration_scale=a.concentration_scale,paper_scale=a.paper_scale,hold=a.hold,config_path=a.config,preset_file=a.preset_file)
