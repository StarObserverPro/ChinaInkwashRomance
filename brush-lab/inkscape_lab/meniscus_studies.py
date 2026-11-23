"""Visual iteration and physical diagnostics for the v0.12 meniscus backend."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
from .brush import InkCanvas
from .cun import ContactSettings
from .form_driver import compile_form
from .bristle_studies import curve_program
from .reference_iteration import make_programs,SIZE,PAPER,make_steps
from .meniscus import MeniscusBrush,MeniscusSettings,material
from .studies import label

def single_program():
    pose={'angle':1.57,'tilt':1.05}
    actions=[
        {'kind':'touch','to':[70,96],'duration':.18,'load':.52,**pose},
        {'kind':'move','to':[200,85],'duration':.60,'load':.66,**pose},
        {'kind':'move','to':[335,112],'duration':.32,'load':.50,**pose},
        {'kind':'move','to':[335,112],'duration':.28,'load':.50,**pose},
        {'kind':'turn','to':[440,78],'duration':.55,'load':.35,'form_event':'turn upward while unloading',**pose},
        {'kind':'release','to':[530,58],'duration':.28,'load':0.,**pose},
        {'kind':'airmove','to':[550,54],'duration':.20,'load':0.,**pose}]
    return {'program':{'tag':'advance accelerate pause lift','start':[45,96],
        'initial':{'load':0.,**pose},'max_dt':1/200,'spatial_step':.4,'actions':actions},
        'curves':[None,[130,78],[265,92],None,[390,122],None,None]}

def render(programs=None,cfg=None,mc=None,fluid=None,size=(610,190),animate=False,checkpoint=None):
    c=InkCanvas(*size,seed=505,paper_color=PAPER);c.density=c.density.astype(float)
    b=MeniscusBrush(c,cfg or ContactSettings(width=35,bunches=64,initial_water=.48),meniscus=mc,fluid=fluid)
    frames=[];next_frame=0.;states=[];last_report=-1
    def capture(t,brush):
        nonlocal next_frame,last_report
        if int(t*4)>last_report:
            last_report=int(t*4)
            print(json.dumps({'t':round(t,3),'film_max_um':float(b.film_h.max()*1e6),'paper_nl':float(b.paper_water.sum()*1e12),'density_max':float(c.density.max()),'wet_pores':sum(p['wet'] for p in b.pores.values()),'film_substeps':b.film_substeps}),flush=True)
            if checkpoint:
                cp=Path(checkpoint);cp.mkdir(exist_ok=True,parents=True);c.save(cp/'preview.png')
        if animate and t>=next_frame:
            frames.append(c.to_image())
            states.append({'t':t,'centers':b.centers.copy(),'anchor':b.anchor.copy(),
                'gap':b.gap.copy(),'edges':b.edge_alive.copy(),'neck':b.neck_alive.copy(),
                'film':b.film_h.copy(),'density':c.density.copy(),'pores':[(key,p['wet']) for key,p in b.pores.items()]})
            next_frame=t+.045
    steps=[]
    for spec in programs or [single_program()]:
        b.draw_driver(compile_form(spec['program'],spec['curves']),frame_callback=capture)
        steps.append(c.to_image())
    return c,b,steps,frames,states

def save_run(path,c,b,steps,frames,states,programs):
    p=Path(path);p.mkdir(parents=True,exist_ok=True)
    c.save(p/'result.png')
    d={'version':b.version,'settings':asdict(b.settings),'meniscus':asdict(b.meniscus),'fluid':asdict(b.fluid),
       'programs':programs,'diagnostics':b.diagnostics(),'trace':b.bridge_trace,
       'canvas_size':[c.width,c.height],'canvas_seed':505,'brush_seed':606,
       'trace_notes':'to_film_m3 and air_supply_nl exclude rupture drops; use diagnostics/fields for total material balance. Noncontact bridge means no solid-tip contact, not complete fluid detachment.',
       'density_sha256':hashlib.sha256(c.density.tobytes()).hexdigest(),
       'final_pores':[{'hairs':[int(i) for i in k],**v} for k,v in b.pores.items()]}
    (p/'run.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v)+'\n')
    np.savez_compressed(p/'fields.npz',density=c.density,film=b.film_h,film_pigment=b.film_mass,
        paper_water=b.paper_water,paper_pigment=b.paper_mass,brush_water=b.volume,brush_pigment=b.mass,
        bridge_water=b.neck_v,bridge_pigment=b.neck_m,pore_water=np.array([v['v'] for v in b.pores.values()]),pore_pigment=np.array([v['m'] for v in b.pores.values()]))
    for i,s in enumerate(steps):s.save(p/f'step_{i+1:02d}.png')
    if frames:
        frames[0].save(p/'progress.gif',save_all=True,append_images=frames[1:]+[steps[-1]]*15,duration=55,loop=0)
    if states:
        selected=[states[int((len(states)-1)*v)] for v in (.13,.34,.61,.83,.93,1)]
        im=Image.new('RGB',(650,6*235+65),PAPER);label(im,(20,15),'WET CONTACT / SURFACE FILM / DEPOSITED INK',22)
        for i,s in enumerate(selected):
            y=55+i*235;label(im,(20,y),f"t = {s['t']:.3f} s",17)
            rgb=np.asarray(steps[-1]).copy();rgb[:]=np.asarray(c.paper_color)
            alpha=1-np.exp(-np.maximum(s['density'],0));rgb=(np.asarray(c.paper_color)[None,None,:]*(1-alpha[:,:,None])+(24,26,25)*alpha[:,:,None]).astype('uint8')
            v=Image.fromarray(rgb);dr=ImageDraw.Draw(v)
            for a,bidx in zip(*np.nonzero(np.triu(s['edges'],1))):
                ca,cb=s['centers'][a],s['centers'][bidx];dr.line([tuple(ca),tuple(cb)],fill=(33,132,151),width=1)
            for j,xy in enumerate(s['centers']):
                if s['neck'][j]:
                    an=s['anchor'][j];dr.line([tuple(xy),tuple(an)],fill=(207,93,56),width=1)
                    dr.ellipse([an[0]-1.5,an[1]-1.5,an[0]+1.5,an[1]+1.5],fill=(207,93,56))
            im.paste(v,(20,y+30))
        im.save(p/'stages.png')
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',default='meniscus-round1')
    ap.add_argument('--rock',action='store_true');ap.add_argument('--config');ap.add_argument('--animate',action='store_true')
    args=ap.parse_args();mc=MeniscusSettings();cfg=ContactSettings(width=35,bunches=64,initial_water=.48)
    f=material()
    if args.config:
        data=json.loads(Path(args.config).read_text());mc=replace(mc,**data.get('meniscus',{}));cfg=replace(cfg,**data.get('settings',{}));f=replace(f,**data.get('fluid',{}))
    programs=make_programs() if args.rock else [single_program()];size=SIZE if args.rock else (610,190)
    begin=time.time();c,b,steps,frames,states=render(programs,cfg,mc,f,size,args.animate,args.output)
    save_run(args.output,c,b,steps,frames,states,programs)
    if args.rock:make_steps(steps,Path(args.output)/'steps.png')
    print(json.dumps({'elapsed_s':time.time()-begin,**b.diagnostics()}),flush=True)

if __name__=='__main__':main()
