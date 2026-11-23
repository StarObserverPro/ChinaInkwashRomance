"""Small controlled wet/dry strokes for reference-guided visual comparisons."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,time
import numpy as np
from PIL import Image
from .brush import InkCanvas
from .cun import ContactSettings
from .meniscus import MeniscusSettings,material
from .paper_transport import PaperSettings
from .reference_brush import ReferenceBrush,SurfaceContactSettings
from .form_driver import compile_form
from .paper_studies import label

PAPER=(245,240,225)

def program(kind):
    scale=.78 if kind=='dry' else 1.
    actions=[
        {'kind':'touch','to':[39,76],'duration':.09,'load':.55*scale,'angle':1.38,'tilt':1.05},
        {'kind':'move','to':[124,64],'duration':.34,'load':.53*scale,'angle':1.28,'tilt':1.0},
        {'kind':'turn','to':[202,87],'duration':.30,'load':.66*scale,'angle':1.69,'tilt':1.12,'form_event':'press and rotate into the lower side of the curved mark'},
        {'kind':'move','to':[256,74],'duration':.24,'load':.40*scale,'angle':1.53,'tilt':.93},
        {'kind':'release','to':[304,56],'duration':.16,'load':0.,'angle':1.47,'tilt':.58},
        {'kind':'airmove','to':[318,51],'duration':.10,'load':0.,'angle':1.47,'tilt':.58}]
    return {'program':{'tag':kind+' reference feature probe','start':[24,76],
        'initial':{'load':0.,'angle':1.38,'tilt':1.05},'max_dt':1/200,'spatial_step':.55,'actions':actions},
        'curves':[None,[76,56],[168,62],[236,95],None,None]}

def run(output,config,kind,hold=4.):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);begin=time.time()
    config=json.loads(Path(config).read_text())
    p=replace(PaperSettings(),texture_model='fibers',heterogeneity=.25,anisotropy=.35)
    p=replace(p,**config.get('paper',{}));contact=SurfaceContactSettings(**config.get('surface_contact',{}))
    spec=program(kind);driver=compile_form(spec['program'],spec['curves'])
    c=InkCanvas(345,150,seed=505,paper_color=PAPER);c.density=c.density.astype(float)
    cfg=ContactSettings(width=29,bunches=48,initial_water=.25 if kind=='dry' else .57)
    cfg=replace(cfg,**config.get('brush',{}))
    f=replace(material(),concentration_kg_m3=44. if kind=='dry' else 24.)
    mc=replace(MeniscusSettings(),**config.get('meniscus',{}))
    b=ReferenceBrush(c,cfg,fluid=f,meniscus=mc,paper=p,surface_contact=contact,record_source=True)
    if config.get('match_initial_inventory'):
        baseline=json.loads((Path('reference-rounds/r0')/kind/'run.json').read_text())
        target=baseline['diagnostics']['initial_nl']*1e-12
        ratio=target/b.volume.sum()
        if np.any(b.volume*ratio>b.capacity):raise ValueError('matched charge exceeds brush capacity')
        b.volume*=ratio;b.mass*=ratio
        b.water_initial=float(b.volume.sum());b.mass_initial=float(b.mass.sum());b._sync()
    frames=[];times=[];next_frame=0.;last_report=-1
    def capture(t,brush):
        nonlocal next_frame,last_report
        if int(t*2)>last_report:
            last_report=int(t*2);c.save(out/'preview.png')
            print(json.dumps({'output':str(out),'t':t}),flush=True)
        if t>=next_frame:
            im=c.to_image();label(im,(8,7),f'{t:.2f}s',12);frames.append(im);times.append(t);next_frame=t+.09
    b.draw_driver(driver,frame_callback=capture)
    at_lift=c.density.copy();lift=b.time;c.save(out/'at_lift.png')
    b.save_source(out/'source.npz')
    b.relax(hold,capture);b.substrate.check();diag=b.diagnostics()
    assert abs(diag['water_relative_error'])<1e-9 and abs(diag['pigment_relative_error'])<1e-9
    c.save(out/'result.png');b.substrate.save(out/'fields.npz')
    np.savez_compressed(out/'lift.npz',density=at_lift)
    frames[0].save(out/'progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*10,duration=75,loop=0)
    data={'version':b.version,'kind':kind,'config':config,'paper':asdict(p),'surface_contact':asdict(contact),
        'brush':asdict(cfg),'fluid':asdict(f),'meniscus':asdict(mc),'program':spec,'lift_time_s':lift,'hold_s':hold,
        'diagnostics':diag,'source_water_nl':b.deposited_water*1e12,'source_pigment_ng':b.deposited_mass*1e12,
        'post_lift_density_relative_l1':float(abs(c.density-at_lift).sum()/max(at_lift.sum(),1e-30)),
        'elapsed_s':time.time()-begin,'times':times,'trace':b.bridge_trace,
        'comparison_note':'Same authored hand program, seed, initial brush load within each wet/dry column. Contact variants change actual ink input. Reference pixels are not renderer inputs.'}
    (out/'run.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'case':kind,'output':str(out),'elapsed_s':data['elapsed_s'],'deposited_nl':data['source_water_nl'],
        'water_relative_error':diag['water_relative_error'],'pigment_relative_error':diag['pigment_relative_error']}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--kind',choices=['wet','dry'],required=True);ap.add_argument('--hold',type=float,default=4.)
    a=ap.parse_args();run(a.output,a.config,a.kind,a.hold)

if __name__=='__main__':main()
