"""Reproducible capillary transport controls and nine-stroke comparison."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict,replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .brush import InkCanvas
from .capillary import CapillaryBrush,FluidSettings
from .cun import ContactSettings
from .form_driver import compile_form
from .reference_iteration import make_programs,settings,SIZE,PAPER,make_steps,image_hash
from .studies import label

VERSION='capillary-study-0.10.0'

def probe_program(speed=200.,pause=None,max_dt=1/240,spatial_step=.4):
    pose={'angle':math.pi/2,'tilt':.9}
    actions=[{'kind':'touch','to':[45,65],'duration':20/speed,'load':.6,**pose}]
    if pause is None:
        actions += [{'kind':'move','to':[500,65],'duration':455/speed,'load':.6,**pose}]
    else:
        actions += [
            {'kind':'move','to':[250,65],'duration':205/speed,'load':.6,**pose},
            {'kind':'release','to':[262,65],'duration':.06,'load':0.,**pose},
            {'kind':'airmove','to':[262,65],'duration':pause,'load':0.,**pose},
            {'kind':'airmove','to':[285,65],'duration':.08,'load':0.,**pose},
            {'kind':'touch','to':[300,65],'duration':15/speed,'load':.6,**pose},
            {'kind':'move','to':[500,65],'duration':200/speed,'load':.6,**pose}]
    actions += [{'kind':'release','to':[530,65],'duration':30/speed,'load':0.,**pose}]
    p={'tag':'capillary control','start':[25,65],'initial':{'load':0.,**pose},
       'max_dt':max_dt,'spatial_step':spatial_step,'actions':actions}
    return {'program':p,'curves':[None]*len(actions),'form_reason':'Fixed straight stroke for material validation.'}

def render(programs=None,cfg=None,fluid=None,size=SIZE):
    c=InkCanvas(*size,seed=505,paper_color=PAPER);c.density=c.density.astype(np.float64)
    b=CapillaryBrush(c,cfg or settings(),seed=606,fluid=fluid)
    frames=[]
    for item in make_programs() if programs is None else programs:
        b.draw_driver(compile_form(item['program'],item['curves']))
        frames.append(c.to_image())
    return c,b,frames

def fields_hash(b):
    h=hashlib.sha256()
    for field in (b.canvas.density,b.paper_water,b.paper_mass,b.paper_mobile,b.paper_fixed,b.volume,b.mass):
        h.update(np.ascontiguousarray(field).tobytes())
    return h.hexdigest()

def save_run(output,c,b,frames,programs):
    p=Path(output);p.mkdir(parents=True,exist_ok=True)
    data={'version':VERSION,'canvas':{'size':[c.width,c.height],'paper':list(PAPER),'seed':505,'brush_seed':606},
          'settings':asdict(b.settings),'fluid':asdict(b.fluid),'programs':programs,
          'records':b.records,'image_sha256':image_hash(c),'fields_sha256':fields_hash(b),'diagnostics':b.diagnostics()}
    (p/'ledger.json.gz').write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    (p/'diagnostics.json').write_text(json.dumps(data['diagnostics'],indent=2),encoding='utf-8')
    (p/'form_actions.json').write_text(json.dumps(programs,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(p/'fields.npz',density=c.density,paper_liquid_m3=b.paper_water,
                       paper_pigment_kg=b.paper_mass,brush_liquid_m3=b.volume,brush_pigment_kg=b.mass,
                       paper_mobile_pigment_kg=b.paper_mobile,paper_fixed_pigment_kg=b.paper_fixed,
                       paper_gap_m=b.gap_field,capacity_m3=b.capacity,tip_area_m2=b.storage_area_m2)
    c.save(p/'result.png')
    for i,frame in enumerate(frames,1):frame.save(p/f'step_{i:02d}.png')
    return data

def replay(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
    if data['version']!=VERSION:raise ValueError('unsupported capillary study version')
    canvas=data['canvas']
    if canvas['paper']!=list(PAPER) or canvas['seed']!=505 or canvas['brush_seed']!=606:
        raise ValueError('unsupported canvas')
    c,b,frames=render(data['programs'],ContactSettings(**data['settings']),FluidSettings(**data['fluid']),tuple(canvas['size']))
    if fields_hash(b)!=data['fields_sha256'] or image_hash(c)!=data['image_sha256'] or b.records!=data['records']:
        raise ValueError('state, mass, liquid or image differs on replay')
    return c,b,frames

def controls(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(output);p.mkdir(parents=True,exist_ok=True)
    cfg=ContactSettings(width=28,bunches=24,initial_water=.35,paper_relief=.15)
    cases={};metrics={}
    for feed in (True,False):
        for pause in (.03,.80):
            name=f'feed_{int(feed)}_pause_{pause:.2f}'
            program=probe_program(pause=pause)
            c,b,frames=render([program],cfg,FluidSettings(refill_multiplier=float(feed),paper_lateral_permeability_m2=0.),(560,130))
            save_run(p/name,c,b,frames,[program]);cases[(feed,pause)]=(c,b)
            trace=b.fluid_trace
            start_index=next(i for i,r in enumerate(trace) if r['phase']=='airmove' and r['event']==3)
            stationary=[r for r in trace if r['phase']=='airmove' and r['event']==3]
            metrics[name]={'pause_s':pause,'feed_enabled':feed,
                'pause_tip_start_nl':trace[start_index-1]['tip_volume_m3']*1e12,
                'pause_tip_end_nl':stationary[-1]['tip_volume_m3']*1e12,
                'restart_mean_optical_depth':float(c.density[50:80,300:345].mean()),
                **b.diagnostics()}
    plate=Image.new('RGB',(1200,495),PAPER)
    label(plate,(22,14),'NO RELOAD / DOES AN AIR PAUSE RESTORE INK?',27)
    for col,feed in enumerate((True,False)):
        x=20+600*col
        label(plate,(x,62),'Capillary supply ON' if feed else 'Capillary supply OFF',22)
        for row,pause in enumerate((.03,.80)):
            label(plate,(x,104+row*172),f'Air pause {pause:.2f} s',18)
            plate.paste(cases[(feed,pause)][0].to_image(),(x,132+row*172))
    label(plate,(22,443),'Within each column: identical geometry, initial ink, speed and paper; only air-pause duration differs.',15)
    label(plate,(22,468),'Paper lateral flow is disabled in this control to isolate liquid transport inside the brush.',15)
    plate.save(p/'pause_comparison.png')

    # Pure fluid subsystem: same depleted initial state, all paper contact off.
    source=copy.deepcopy(cases[(True,.03)][1])
    on=copy.deepcopy(source);off=copy.deepcopy(source);off.conductance[:]=0.
    times=np.linspace(0.,1.,501);tip_on=[on.volume[cfg.bunches:].sum()*1e12];tip_off=[off.volume[cfg.bunches:].sum()*1e12]
    body_on=[on.volume[:cfg.bunches].sum()*1e12]
    for dt in np.diff(times):
        on.fluid_step(float(dt),None);off.fluid_step(float(dt),None)
        tip_on.append(on.volume[cfg.bunches:].sum()*1e12);tip_off.append(off.volume[cfg.bunches:].sum()*1e12)
        body_on.append(on.volume[:cfg.bunches].sum()*1e12)
    relaxation=np.c_[times,tip_on,tip_off,body_on]
    np.savetxt(p/'air_recovery.csv',relaxation,delimiter=',',header='time_s,tip_on_nl,tip_off_nl,body_on_nl',comments='')
    metrics['air_recovery']={'initial_tip_nl':tip_on[0],'final_tip_on_nl':tip_on[-1],
                            'final_tip_off_nl':tip_off[-1],
                            'body_loss_nl':body_on[0]-body_on[-1],
                            'paper_liquid_change_m3':float(on.paper_water.sum()-source.paper_water.sum())}
    fig,axes=plt.subplots(1,2,figsize=(11.4,3.6),layout='constrained')
    axes[0].plot(times,tip_on,color='#1d5759',lw=2.4,label='Darcy supply on')
    axes[0].plot(times,tip_off,color='#ac7045',ls='--',lw=2,label='Supply off')
    axes[0].set(xlabel='Air pause / s',ylabel='Liquid at tips / nL',title='Recovery without adding ink')
    axes[0].legend(frameon=False);axes[0].grid(alpha=.18)
    axes[1].plot(times,np.array(tip_on)-tip_on[0],color='#1d5759',lw=2.5,label='Tip gain')
    axes[1].plot(times,body_on[0]-np.array(body_on),color='#ac7045',ls='--',lw=2,label='Body loss')
    axes[1].set(xlabel='Air pause / s',ylabel='Transferred liquid / nL',title='The recovered liquid comes from the body')
    axes[1].legend(frameon=False);axes[1].grid(alpha=.18)
    fig.savefig(p/'air_recovery.png',dpi=170);plt.close(fig)

    speed_images=[]
    for speed in (100.,200.,400.):
        program=probe_program(speed)
        c,b,frames=render([program],cfg,size=(560,130))
        name=f'speed_{int(speed)}';save_run(p/name,c,b,frames,[program])
        speed_images.append(c.to_image())
        metrics[name]={'speed_mm_s':speed*b.fluid.pixel_m*1000,
            'mean_optical_depth':float(c.density[50:80,70:485].mean()),**b.diagnostics()}
    plate=Image.new('RGB',(610,605),PAPER)
    label(plate,(20,16),'SPEED / FIXED LIQUID AND PAPER',24)
    for i,(im,speed) in enumerate(zip(speed_images,(5,10,20))):
        label(plate,(20,67+160*i),f'{speed} mm/s',20);plate.paste(im,(20,97+160*i))
    label(plate,(20,577),'Illustrative parameters; not fitted to measured brushstrokes.',14)
    plate.save(p/'speed_comparison.png')
    (p/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({'controls':'complete','air_recovery':metrics['air_recovery']},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='capillary-final')
    parser.add_argument('--replay')
    parser.add_argument('--controls',action='store_true')
    args=parser.parse_args()
    if args.replay:
        c,b,_=replay(args.replay);print(json.dumps({'replay':'PASS','fields_sha256':fields_hash(b),**b.diagnostics()}));return
    if args.controls:controls(args.output);return
    programs=make_programs();c,b,frames=render(programs)
    data=save_run(args.output,c,b,frames,programs);make_steps(frames,Path(args.output)/'steps.png')
    plate=Image.new('RGB',(1360,650),PAPER)
    label(plate,(20,15),'FIXED NINE ACTIONS / TRANSPORT PROTOTYPE',27)
    for col,(image,title) in enumerate([(Image.open('dry-final/result.png').convert('RGB'),'v0.9 / Correlated procedural supply'),(c.to_image(),'v0.10 / Finite capillary supply')]):
        x=20+680*col;label(plate,(x,65),title,21);plate.paste(image,(x,102))
    label(plate,(20,631),'Different material models and ink units. This is not a calibrated likeness comparison.',16)
    plate.save(Path(args.output)/'comparison.png')
    print(json.dumps({'image_sha256':data['image_sha256'],**b.diagnostics()}))

if __name__=='__main__':main()
