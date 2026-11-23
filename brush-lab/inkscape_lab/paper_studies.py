"""Controlled substrate probes and review images for the v0.13 paper model."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse,json,time,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .brush import InkCanvas
from .meniscus import material,MeniscusBrush
from .cun import ContactSettings
from .paper_transport import PaperTransport,PaperSettings

PAPER=(244,240,228)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def label(im,xy,text,size=18):
    ImageDraw.Draw(im).text(xy,text,fill=(38,47,46),font=ImageFont.truetype(FONT,size))

def image_density(d):
    c=InkCanvas(d.shape[1],d.shape[0],seed=505,paper_color=PAPER);c.density=d
    return c.to_image()

def radius(field,threshold,center,dx):
    yy,xx=np.indices(field.shape);rr=np.hypot(xx-center[0],yy-center[1])*dx
    active=field>threshold
    return float(rr[active].max()*1e3) if active.any() else 0.

def mass_radius(field,fraction,center,dx):
    yy,xx=np.indices(field.shape);rr=np.hypot(xx-center[0],yy-center[1]);order=np.argsort(rr.ravel())
    cumulative=np.cumsum(field.ravel()[order])
    return float(rr.ravel()[order][np.searchsorted(cumulative,cumulative[-1]*fraction)]*dx*1e3)

def droplet(output,legacy=False,config=None,duration=6.,dt=.006,scale=1,coupled=False):
    path=Path(output);path.mkdir(exist_ok=True,parents=True)
    shape=(150*scale,170*scale);center=(85*scale+(scale-1)/2,75*scale+(scale-1)/2)
    f=replace(material(),pixel_m=material().pixel_m/scale);s=PaperSettings()
    if config:s=replace(s,**json.loads(Path(config).read_text()))
    yy,xx=np.indices(shape);r=np.hypot(xx-center[0],yy-center[1])
    kernel=np.maximum(0.,1-(r/(7*scale))**2);kernel/=kernel.sum()
    water=kernel*5e-12;pigment=water*f.concentration_kg_m3
    if legacy:
        canvas=InkCanvas(shape[1],shape[0],seed=505,paper_color=PAPER)
        model=MeniscusBrush(canvas,ContactSettings(width=12,bunches=12),fluid=f)
        model.film_h[:]=water/f.pixel_m**2;model.film_mass[:]=pigment
        model.paper_bounds=(60*scale,91*scale,70*scale,101*scale)
    elif coupled:
        from .absorbent_brush import AbsorbentBrush
        canvas=InkCanvas(shape[1],shape[0],seed=505,paper_color=PAPER)
        brush=AbsorbentBrush(canvas,ContactSettings(width=12,bunches=12),fluid=f,paper=s)
        model=brush.substrate;model.film[:]=water/f.pixel_m**2;model.film_mass[:]=pigment
    else:
        model=PaperTransport(shape,f,s);model.film[:]=water/f.pixel_m**2;model.film_mass[:]=pigment
    times=[0.,.03,.1,.3,1.,3.,duration];times=sorted(set(t for t in times if t<=duration))
    frames=[];snapshots=[];trace=[];t=0.;next_frame=0.;clock=time.time()
    def state():
        if legacy:
            d=(model.paper_mass+model.film_mass)*f.extinction_m2_kg/f.pixel_m**2;v=model.paper_water+model.film_h*f.pixel_m**2
            m=model.paper_mass+model.film_mass
            return d,v,m
        return model.density(),model.water.sum(axis=0)+model.film*f.pixel_m**2,model.pigment.sum(axis=0)+model.film_mass
    for target in times:
        while t<target-1e-10:
            step=min(dt,target-t)
            if legacy:model._surface(step)
            elif coupled:brush._surface(step)
            else:model.advance(step)
            t+=step
            if t>=next_frame:
                d,v,m=state();frame=image_density(d);label(frame,(8,6),f'{t:.2f} s',14);frames.append(frame);next_frame=t+.06
        if coupled:brush.flush_paper();brush.previous_film[:]=brush.film_h;brush.previous_mass[:]=brush.film_mass
        d,v,m=state()
        # Equal-initial-concentration thresholds plus independent mass radii.
        tr={'t':t,'water_nl':float(v.sum()*1e12),'pigment_ng':float(m.sum()*1e12),
            'water_front_radius_mm':radius(v/f.pixel_m**2,.08/(f.extinction_m2_kg*f.concentration_kg_m3),center,f.pixel_m),
            'ink_front_radius_mm':radius(d,.08,center,f.pixel_m),
            'water_mass95_radius_mm':mass_radius(v,.95,center,f.pixel_m),
            'pigment_mass95_radius_mm':mass_radius(m,.95,center,f.pixel_m),
            'pigment_second_moment_px2':float((((xx-center[0])**2+(yy-center[1])**2)*m).sum()/m.sum())}
        trace.append(tr);snapshots.append(image_density(d))
        print(json.dumps(tr),flush=True)
    if not legacy:model.check();model.save(path/'fields.npz')
    else:np.savez_compressed(path/'fields.npz',water=v,pigment=m,density=d)
    snapshots[-1].save(path/'result.png')
    for i,im in enumerate(snapshots):im.save(path/f'time_{i:02d}.png')
    frames[0].save(path/'progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*15,duration=60,loop=0)
    plate=Image.new('RGB',(len(times)*250,305),PAPER)
    for i,(ts,im) in enumerate(zip(times,snapshots)):
        label(plate,(i*250+12,10),f'{ts:g} s',21)
        plate.paste(im.crop(tuple(v*scale for v in (40,30,130,120))).resize((240,240),Image.Resampling.NEAREST),(i*250+5,52))
    plate.save(path/'evolution.png')
    totals={'water_relative_error':trace[-1]['water_nl']/5-1,'pigment_relative_error':trace[-1]['pigment_ng']/140-1}
    assert abs(totals['water_relative_error'])<1e-9 and abs(totals['pigment_relative_error'])<1e-9
    data={'case':'legacy' if legacy else ('layered_coupled' if coupled else 'layered_paper_only'),'settings':asdict(s),'fluid':asdict(f),'times':times,'trace':trace,'validation':totals,'elapsed_s':time.time()-clock,'scale':scale}
    (path/'run.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'elapsed_s':time.time()-clock,**totals}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--legacy',action='store_true');ap.add_argument('--config');ap.add_argument('--duration',type=float,default=6.);ap.add_argument('--dt',type=float,default=.006);ap.add_argument('--scale',type=int,default=1);ap.add_argument('--coupled',action='store_true')
    args=ap.parse_args();droplet(args.output,args.legacy,args.config,args.duration,args.dt,args.scale,args.coupled)

if __name__=='__main__':main()
