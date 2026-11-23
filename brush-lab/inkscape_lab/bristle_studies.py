"""Reproducible v0.11 tuft-contact studies, reference rock and visual controls."""
from __future__ import annotations
import argparse,gzip,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label as components
from .bristle import BristleBrush,BristleSettings,material_settings
from .capillary import CapillaryBrush,FluidSettings
from .capillary_studies import fields_hash,probe_program
from .brush import InkCanvas
from .cun import ContactSettings
from .form_driver import compile_form
from .reference_iteration import make_programs,settings as reference_settings,SIZE,PAPER,make_steps,image_hash
from .studies import label

VERSION='bristle-study-0.11.0'

def settings():
    # The reference rock uses a fuller initial load; the 0.40 single-stroke
    # mechanism controls retain their own explicit, matched initial condition.
    return replace(reference_settings(),initial_water=.55)

def curve_program():
    actions=[
      {'kind':'touch','to':[80,100],'duration':.22,'load':.35,'tilt':.9,'angle':1.57},
      {'kind':'move','to':[220,75],'duration':.70,'load':.8,'tilt':1.1,'angle':1.57},
      {'kind':'turn','to':[340,135],'duration':.65,'load':.45,'tilt':.9,'angle':1.,'form_event':'press through the lower bend'},
      {'kind':'turn','to':[445,85],'duration':.55,'load':.3,'tilt':.7,'angle':1.4,'form_event':'lift into the upper return'},
      {'kind':'release','to':[535,60],'duration':.45,'load':0.,'tilt':.3,'angle':1.4}]
    return {'program':{'tag':'press turn lift','start':[45,100],'max_dt':1/240,'spatial_step':.35,'actions':actions},
            'curves':[None,[160,50],[270,90],[410,150],None],'form_reason':'One continuous contact with pressure changes and two turns.'}

def render(programs=None,cfg=None,fluid=None,bristles=None,size=SIZE,material='bristle',animate=False):
    c=InkCanvas(*size,seed=505,paper_color=PAPER);c.density=c.density.astype(np.float64)
    if material=='bristle':b=BristleBrush(c,cfg or settings(),seed=606,fluid=fluid,bristles=bristles)
    elif material=='capillary':b=CapillaryBrush(c,cfg or settings(),seed=606,fluid=fluid)
    else:raise ValueError('unknown material')
    frames=[];animation=[];next_frame=0.
    def capture(t,canvas):
        nonlocal next_frame
        if t+1e-10>=next_frame:
            animation.append(canvas.to_image());next_frame=t+.08
    for spec in make_programs() if programs is None else programs:
        driver=compile_form(spec['program'],spec['curves'])
        if animate and material=='bristle':b.draw_driver(driver,frame_callback=capture)
        else:b.draw_driver(driver)
        frames.append(c.to_image())
    return c,b,frames,animation

def save_run(output,c,b,frames,programs,material='bristle'):
    p=Path(output);p.mkdir(parents=True,exist_ok=True)
    data={'version':VERSION,'material':material,'canvas':{'size':[c.width,c.height],'paper':list(PAPER),'seed':505,'brush_seed':606},
          'settings':asdict(b.settings),'fluid':asdict(b.fluid),
          'bristles':asdict(b.bristles) if isinstance(b,BristleBrush) else None,
          'programs':programs,'records':b.records,'image_sha256':image_hash(c),
          'fields_sha256':fields_hash(b),'diagnostics':b.diagnostics()}
    (p/'ledger.json.gz').write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    (p/'diagnostics.json').write_text(json.dumps(data['diagnostics'],indent=2)+'\n')
    (p/'form_actions.json').write_text(json.dumps(programs,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(p/'fields.npz',density=c.density,paper_liquid_m3=b.paper_water,
        paper_pigment_kg=b.paper_mass,paper_mobile_pigment_kg=b.paper_mobile,paper_fixed_pigment_kg=b.paper_fixed,
        brush_liquid_m3=b.volume,brush_pigment_kg=b.mass,centers=b.centers)
    c.save(p/'result.png')
    for i,frame in enumerate(frames,1):frame.save(p/f'step_{i:02d}.png')
    return data

def replay(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:d=json.load(f)
    if d['version']!=VERSION:raise ValueError('unsupported bristle-study version')
    c,b,frames,_=render(d['programs'],ContactSettings(**d['settings']),FluidSettings(**d['fluid']),
        BristleSettings(**d['bristles']) if d['bristles'] else None,tuple(d['canvas']['size']),d['material'])
    if fields_hash(b)!=d['fields_sha256'] or image_hash(c)!=d['image_sha256'] or b.records!=d['records']:
        raise ValueError('material state, field or image differs on replay')
    return c,b,frames

def texture_summary(c,b):
    # Relative threshold avoids claiming the darker ink alone improves topology.
    positive=c.density[c.density>1e-8]
    threshold=float(np.percentile(positive,90)*.2) if len(positive) else 0.
    lab,n=components(c.density>threshold,np.ones((3,3)))
    areas=np.bincount(lab.ravel())[1:]
    return {'threshold_optical_depth':threshold,'components':int(n),
        'components_up_to_4px':int(np.count_nonzero(areas<=4)),
        'largest_component_fraction':float(areas.max()/areas.sum()) if n else 0.,
        **b.diagnostics()}

def controls(output):
    p=Path(output);p.mkdir(parents=True,exist_ok=True)
    cfg=ContactSettings(width=35,bunches=36,initial_water=.4)
    spec=curve_program();cases={};metrics={}
    configs=[('v010',cfg,None,None,'capillary'),('v011',cfg,None,None,'bristle'),
       ('wet',replace(cfg,initial_water=.68),None,None,'bristle'),
       ('dry',replace(cfg,initial_water=.20),None,None,'bristle'),
       ('no_cohesion',cfg,None,replace(BristleSettings(),cohesive_enabled=False),'bristle'),
       ('flat_paper',cfg,None,replace(BristleSettings(),paper_height_fraction=0.,paper_height_m=0.),'bristle')]
    for name,config,fluid,bristles,material in configs:
        c,b,frames,animation=render([spec],config,fluid,bristles,(630,190),material,animate=name=='v011')
        save_run(p/name,c,b,frames,[spec],material);cases[name]=c.to_image();metrics[name]=texture_summary(c,b)
        if animation:
            animation+=[frames[-1]]*12
            animation[0].save(p/'stroke_progress.gif',save_all=True,append_images=animation[1:],duration=80,loop=0)
        print(json.dumps({'control':name,'components':metrics[name]['components'],'bonds_changed':getattr(b,'bond_changes',None)}),flush=True)
    plate=Image.new('RGB',(690,565),PAPER)
    label(plate,(24,18),'SAME HAND PATH / CHANGED BRUSH CONTACT',25)
    for i,(key,title) in enumerate([('v010','v0.10 / Pixel-scale film gating'),('v011','v0.11 / Persistent tufts and connected wet contact')]):
        label(plate,(25,80+i*230),title,20);plate.paste(cases[key],(25,116+i*230))
    label(plate,(25,539),'Material and optical parameters differ; this is a visual version comparison.',14)
    plate.save(p/'stroke_comparison.png')
    plate=Image.new('RGB',(690,530),PAPER)
    label(plate,(24,18),'SAME v0.11 BRUSH / INITIAL LIQUID ONLY',25)
    for i,(key,title) in enumerate([('wet','More liquid / initial saturation 0.68'),('dry','Less liquid / initial saturation 0.20')]):
        label(plate,(25,71+i*223),title,20);plate.paste(cases[key],(25,104+i*223))
    plate.save(p/'wet_dry.png')
    plate=Image.new('RGB',(690,770),PAPER)
    label(plate,(24,18),'MECHANISM CONTROLS / SAME INK AND PATH',24)
    for i,(key,title) in enumerate([('v011','Cohesion + paper relief'),('no_cohesion','Cohesion disabled'),('flat_paper','Paper relief forces and contact modulation disabled')]):
        label(plate,(25,70+i*230),title,18);plate.paste(cases[key],(25,103+i*230))
    plate.save(p/'mechanism_comparison.png')
    (p/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='bristle-final');parser.add_argument('--controls',action='store_true');parser.add_argument('--replay')
    args=parser.parse_args()
    if args.replay:
        c,b,_=replay(args.replay);print(json.dumps({'replay':'PASS','image_sha256':image_hash(c),'fields_sha256':fields_hash(b),**b.diagnostics()}));return
    if args.controls:controls(args.output);return
    programs=make_programs();c,b,frames,_=render(programs)
    data=save_run(args.output,c,b,frames,programs);make_steps(frames,Path(args.output)/'steps.png')
    plate=Image.new('RGB',(1360,665),PAPER)
    label(plate,(24,17),'NINE STROKES / v0.10 - v0.11',27)
    for col,(im,title) in enumerate([(Image.open('capillary-final/result.png').convert('RGB'),'v0.10 / Film threshold'),(c.to_image(),'v0.11 / Bristle contact')]):
        x=20+680*col;label(plate,(x,65),title,22);plate.paste(im,(x,103))
    label(plate,(24,640),'Same hand programs; contact response, material and optical parameters changed.',15)
    plate.save(Path(args.output)/'comparison.png')
    print(json.dumps({'image_sha256':data['image_sha256'],**b.diagnostics()}))

if __name__=='__main__':main()
