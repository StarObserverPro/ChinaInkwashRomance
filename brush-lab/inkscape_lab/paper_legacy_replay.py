"""Replay the same conserved brush input through the unchanged v0.12 paper."""
from pathlib import Path
import argparse,json,time
import numpy as np
from .meniscus import MeniscusBrush,material
from .brush import InkCanvas
from .cun import ContactSettings
from .paper_studies import PAPER,label

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);ap.add_argument('--hold',type=float,default=6.);a=ap.parse_args()
    tape=np.load(a.source);H,W=tape['shape'];p=Path(a.output);p.mkdir(parents=True,exist_ok=True)
    c=InkCanvas(int(W),int(H),seed=505,paper_color=PAPER);c.density=c.density.astype(float)
    b=MeniscusBrush(c,ContactSettings(width=35,bunches=64,initial_water=.48),fluid=material())
    frames=[];trace=[];t=0.;nextframe=0.;begin=time.time()
    def capture():
        nonlocal nextframe
        if t>=nextframe:
            im=c.to_image();label(im,(12,12),f'{t:.2f} s',16);frames.append(im);nextframe=t+.09
            trace.append({'t':t,'film_nl':float(b.film_h.sum()*b.fluid.pixel_m**2*1e12),
                'paper_nl':float(b.paper_water.sum()*1e12),'mobile_pigment_ng':float(b.paper_mobile.sum()*1e12)})
    last=-1
    for i,dt in enumerate(tape['dt']):
        l,r=tape['offsets'][i:i+2];ids=tape['ids'][l:r]
        np.add.at(b.film_h.ravel(),ids,tape['water'][l:r]/b.fluid.pixel_m**2)
        np.add.at(b.film_mass.ravel(),ids,tape['pigment'][l:r])
        if len(ids):
            yy,xx=np.divmod(ids,W);b._extend((int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1))
        b._surface(float(dt));t+=dt;capture()
        if int(t)>last:last=int(t);print(json.dumps({'t':t}),flush=True)
    lift=t;c.save(p/'at_lift.png');at_lift=c.density.copy()
    while t<lift+a.hold-1e-10:
        dt=min(.012,lift+a.hold-t);b._surface(dt);t+=dt;capture()
    c.save(p/'result.png');np.savez_compressed(p/'fields.npz',density=c.density,paper_water=b.paper_water,paper_mobile=b.paper_mobile,paper_fixed=b.paper_fixed,film=b.film_h,film_mass=b.film_mass)
    np.savez_compressed(p/'lift_state.npz',density=at_lift)
    frames[0].save(p/'progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*16,duration=70,loop=0)
    v=float(b.paper_water.sum()+b.film_h.sum()*b.fluid.pixel_m**2);m=float(b.paper_mobile.sum()+b.paper_fixed.sum()+b.film_mass.sum())
    diag={'water_relative_error':v/tape['water'].sum()-1,'pigment_relative_error':m/tape['pigment'].sum()-1,
        'post_lift_density_relative_l1':float(abs(c.density-at_lift).sum()/at_lift.sum())}
    assert abs(diag['water_relative_error'])<1e-9 and abs(diag['pigment_relative_error'])<1e-9
    (p/'run.json').write_text(json.dumps({'source':str(a.source),'lift_time_s':lift,'hold_s':a.hold,'trace':trace,'diagnostics':diag,'elapsed_s':time.time()-begin},indent=2)+'\n')
    print(json.dumps({'elapsed_s':time.time()-begin,**diag}),flush=True)

if __name__=='__main__':main()
