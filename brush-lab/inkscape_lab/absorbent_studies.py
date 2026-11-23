"""Render or replay a fixed conservative brush source on absorbent paper."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,time
import numpy as np
from PIL import Image
from .absorbent_brush import AbsorbentBrush
from .meniscus import MeniscusSettings,material
from .meniscus_studies import single_program,save_run
from .paper_transport import PaperSettings
from .paper_studies import PAPER,label
from .brush import InkCanvas
from .cun import ContactSettings
from .form_driver import compile_form
from .reference_iteration import make_programs,SIZE,make_steps

def run(output,config=None,source=None,hold=6.,rock=False):
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    ps=PaperSettings();cfg=ContactSettings(width=35,bunches=64,initial_water=.48)
    if config:
        data=json.loads(Path(config).read_text());ps=replace(ps,**data.get('paper',{}));cfg=replace(cfg,**data.get('settings',{}))
    tape=np.load(source) if source else None
    shape=tuple(tape['shape']) if source else tuple(reversed(SIZE if rock else (610,190)))
    c=InkCanvas(shape[1],shape[0],seed=505,paper_color=PAPER);c.density=c.density.astype(float)
    b=AbsorbentBrush(c,cfg,meniscus=MeniscusSettings(),paper=ps,record_source=source is None)
    programs=make_programs() if rock else [single_program()]
    frames=[];times=[];trace=[];nextframe=0.;lastprint=-1
    def capture(t,brush):
        nonlocal nextframe,lastprint
        if t>=nextframe:
            im=c.to_image();label(im,(12,12),f'{t:.2f} s',16);frames.append(im);times.append(t)
            trace.append({'t':t,**b.substrate.totals()});nextframe=t+.09
        if int(t)>lastprint:
            lastprint=int(t);c.save(path/'preview.png')
            print(json.dumps({'t':t,'film_nl':b.substrate.totals()['film_nl'],'max_density':float(c.density.max())}),flush=True)
    begin=time.time();steps=[]
    if source:
        b.water_added_si=float(tape['water'].sum());b.mass_added_si=float(tape['pigment'].sum())
        for i,dt in enumerate(tape['dt']):
            a,bb=tape['offsets'][i:i+2];idx=tape['ids'][a:bb]
            np.add.at(b.film_h.ravel(),idx,tape['water'][a:bb]/b.fluid.pixel_m**2)
            np.add.at(b.film_mass.ravel(),idx,tape['pigment'][a:bb])
            b._surface(float(dt));b.time+=dt;capture(b.time,b)
        b.flush_paper();steps=[c.to_image()]
    else:
        for prog in programs:
            b.draw_driver(compile_form(prog['program'],prog['curves']),frame_callback=capture);steps.append(c.to_image())
        b.save_source(path/'source.npz')
    end=c.to_image();end.save(path/'at_lift.png');lift_time=b.time
    lift_density=c.density.copy();b.relax(hold,capture)
    diag=b.diagnostics();b.substrate.check()
    assert abs(diag['water_relative_error'])<1e-9 and abs(diag['pigment_relative_error'])<1e-9
    save_run(path,c,b,steps,[],[],programs)
    c.save(path/'result.png');b.substrate.save(path/'paper_fields.npz')
    np.savez_compressed(path/'lift_state.npz',density=lift_density)
    frames[0].save(path/'progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*16,duration=70,loop=0)
    plate=Image.new('RGB',(650,6*220+65),PAPER);label(plate,(20,15),'ABSORPTION CONTINUES AFTER THE BRUSH LIFTS',22)
    goals=[.78,1.38,lift_time,lift_time+.4,lift_time+2,b.time]
    for i,g in enumerate(goals):
        j=int(np.argmin(abs(np.array(times)-g)));label(plate,(20,55+i*220),f't = {times[j]:.2f} s',18)
        plate.paste(frames[j],(20,80+i*220))
    plate.save(path/'evolution.png')
    if rock:make_steps(steps,path/'steps.png')
    data={'version':b.version,'paper':asdict(ps),'source':str(source) if source else 'recorded during brush simulation',
        'lift_time_s':lift_time,'hold_s':hold,'trace':trace,'diagnostics':diag,'elapsed_s':time.time()-begin,
        'post_lift_density_relative_l1':float(abs(c.density-lift_density).sum()/np.maximum(lift_density.sum(),1e-30))}
    (path/'paper_run.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'elapsed_s':time.time()-begin,**diag}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--source');p.add_argument('--hold',type=float,default=6.);p.add_argument('--rock',action='store_true')
    a=p.parse_args();run(a.output,a.config,a.source,a.hold,a.rock)

if __name__=='__main__':main()
