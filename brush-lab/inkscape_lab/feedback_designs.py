"""Recorded visual-feedback decisions; no automatic realism claim."""
from pathlib import Path
from copy import deepcopy
import json
from .monte_carlo_studies import ROOT
from .technique_studies import save_json

def make_tasks(round_name,variants,cases=('leaf','dry')):
    tasks=[]
    for sid,cfg,note in variants:
        for case in cases:
            tasks.append(dict(id=sid,config=cfg,case=case,output=f'mc/{round_name}/{sid}/{case}',feedback_note=note))
    path=ROOT/'mc'/f'{round_name}_design.json'
    save_json(path,dict(variants=[dict(id=sid,config=cfg,note=note) for sid,cfg,note in variants],tasks=tasks))
    return path

def round2():
    parent=json.loads((ROOT/'mc/global/M09/leaf/run.json').read_text())['config']
    variants=[]
    specifications=[
      ('B00',{'release':'cap'}, {}, 'M09 with old circular cap release; matched control'),
      ('B01',{}, {}, 'M09 replay: remembered footprint split'),
      ('B02',{'release':'drain'}, {}, 'M09 drainage-only limiting closure'),
      ('B03',{'neck_law':'film'}, {}, 'Matched h^3 rather than h^4 neck conductance'),
      ('B04',{'release':'drain','neck_law':'film'}, {}, 'Drainage closure with h^3 conductance'),
      ('B05',{'neck_law':'film','memory_seconds':.015}, {}, 'Causal 15 ms wetted-footprint history'),
      ('B06',{'geometry':'arc','neck_law':'film'}, {}, 'Rising arc / tilted tuft at same material parameters'),
      ('B07',{'flux':'series','neck_law':'film'}, {}, 'Series resistance at same material parameters'),
      ('B08',{'relief':'shared','neck_law':'film'}, {}, 'Shared fiber-height field'),
      ('B09',{'neck_law':'film'}, {'entry_capture_coarse':.45,'entry_capture_fine':.17}, 'Moderate entry capture with h^3'),
      ('B10',{'release':'drain','neck_law':'film'}, {'entry_capture_coarse':.45,'entry_capture_fine':.17}, 'Moderate capture; drainage-only'),
      ('B11',{'geometry':'arc','arc_rise_scale':.35,'neck_law':'film'}, {}, 'Reduced arc rise; tilted tuft still active')]
    for sid,s,i,note in specifications:
        cfg=deepcopy(parent);cfg['structure'].update(s);cfg['interaction'].update(i);variants.append((sid,cfg,note))
    return make_tasks('round2',variants)

def round3():
    from .monte_carlo_studies import latin_hypercube
    parent=json.loads((ROOT/'mc/round2/B04/leaf/run.json').read_text())['config']
    variants=[('C00',deepcopy(parent),'B04 repeat: visual anchor')]
    cfg=deepcopy(parent);cfg['structure']['release']='memory'
    variants.append(('C01',cfg,'Same geometry / h^3 with remembered-footprint split'))
    cfg=deepcopy(parent);cfg['meniscus']['cohesion']=False
    variants.append(('C02',cfg,'Cohesion ablation; retained as diagnostic, not physical recommendation'))
    cfg=deepcopy(parent);cfg['meniscus']['pores_enabled']=False
    variants.append(('C03',cfg,'Three-hair pore ablation'))
    u=latin_hypercube(8,7,160932)
    for i,r in enumerate(u):
        cfg=deepcopy(parent);s=cfg['structure'];m=cfg['meniscus']
        s['relief_um']=float(10+16*r[0]);s['correlation_mm']=float(.07+.13*r[1])
        m['neck_radius_m']*=float(.9+.55*r[2]);m['friction_coefficient']=float(.18+.37*r[3])
        m['young_pa']*=float(.65+.55*r[4]);m['body_feed_scale']*=float(.7+.8*r[5])
        cap=float(.1+.4*r[6]);cfg['interaction']=dict(entry_capture_coarse=cap,entry_capture_fine=cap*.32/.85)
        variants.append((f'C{i+4:02d}',cfg,'Local randomized Latin hypercube around B04; seed 160932'))
    return make_tasks('round3',variants)

def validation(parent_id):
    parent=json.loads((ROOT/'mc/round3'/parent_id/'leaf/run.json').read_text())['config']
    save_json(ROOT/'selected_v016.json',parent)
    tasks=[]
    def add(sid,case,**extras):
        tasks.append(dict(id=sid,config=deepcopy(parent),case=case,output=f'mc/validation/{sid}/{case}',**extras))
    for seed in [11,37]:
        for case in ['leaf','dry']:add(f'seed{seed}',case,seed=seed)
    for factor in [2,4]:
        for case in ['leaf','dry']:
            add(f'dt{factor}',case)
            tasks[-1]['config']['technique']=dict(max_step_s=.0015/factor,spatial_step_mm=.035/factor)
    for case in ['pulse','side','fold','bone']:add('heldout',case,animation=True)
    for sid,speed in [('fast',1.6),('slow',.65)]:add(sid,'leaf',controls=dict(speed=speed))
    add('dense','dry',controls=dict(concentration=2.5),animation=True)
    add('selected','rock',hold=.8,animation=True)
    for rid in [0,5]:
        cfg=json.loads((ROOT/f'technique-r{rid}-config.json').read_text())
        for case in ['dry','rock']:
            tasks.append(dict(id=f'R{rid}',case=case,config=cfg,backend='legacy',
                output=f'mc/validation/R{rid}/{case}',controls=dict(concentration=2.5) if case=='dry' else {},hold=.8 if case=='rock' else .6))
    path=ROOT/'mc/validation_design.json';save_json(path,dict(parent=parent_id,tasks=tasks));return path

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,default=2);p.add_argument('--parent',default='C06');a=p.parse_args()
    print(round2() if a.round==2 else round3() if a.round==3 else validation(a.parent))
