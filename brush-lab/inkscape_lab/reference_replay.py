"""Reconstruct paper fields from a reference study's recorded brush source."""
from pathlib import Path
import argparse,json,os,time
import numpy as np
from PIL import Image
from .reference_brush import ReferenceBrush,SurfaceContactSettings
from .paper_transport import PaperSettings
from .capillary import FluidSettings
from .meniscus import MeniscusSettings
from .cun import ContactSettings
from .brush import InkCanvas
from .paper_studies import label

def replay(path,progress=False):
    path=Path(path);d=json.loads((path/'run.json').read_text());tape=np.load(path/'source.npz');H,W=tape['shape'];begin=time.time()
    c=InkCanvas(int(W),int(H),seed=505,paper_color=(245,240,225));c.density=c.density.astype(float)
    b=ReferenceBrush(c,ContactSettings(**d['brush']),fluid=FluidSettings(**d['fluid']),
        meniscus=MeniscusSettings(**d['meniscus']),paper=PaperSettings(**d['paper']),
        surface_contact=SurfaceContactSettings(**d['surface_contact']))
    frames=[];frame_index=0
    def capture(t,brush):
        nonlocal frame_index
        if not progress or frame_index>=len(d['times']):return
        target=d['times'][frame_index]
        if t<target-1e-9:return
        assert abs(t-target)<1e-8, ('Replay missed a recorded frame time',t,target)
        im=c.to_image();label(im,(8,7),f'{target:.2f}s',12)
        frames.append(im);frame_index+=1
    for i,dt in enumerate(tape['dt']):
        lo,hi=tape['offsets'][i:i+2];ids=tape['ids'][lo:hi]
        np.add.at(b.film_h.ravel(),ids,tape['water'][lo:hi]/b.fluid.pixel_m**2)
        np.add.at(b.film_mass.ravel(),ids,tape['pigment'][lo:hi])
        b._surface(float(dt));b.time+=dt
        capture(b.time,b)
    b.flush_paper();b.relax(d['hold_s'],capture);b.substrate.check()
    old=np.asarray(Image.open(path/'result.png').convert('RGB'),float);new=np.asarray(c.to_image(),float)
    error=float(abs(old-new).max());mean=float(abs(old-new).mean())
    assert error<=1 and mean<.01, ('Source replay differs from original rendering',error,mean)
    temporary=path/'fields.replay.tmp.npz';b.substrate.save(temporary);os.replace(temporary,path/'fields.npz')
    if progress:
        assert frame_index==len(d['times'])
        temporary=path/'progress.replay.tmp.gif'
        frames[0].save(temporary,save_all=True,append_images=frames[1:]+[frames[-1]]*10,duration=75,loop=0)
        with Image.open(temporary) as check_image:
            assert check_image.n_frames==len(d['times'])
            for i in range(check_image.n_frames):check_image.seek(i);check_image.load()
        os.replace(temporary,path/'progress.gif')
    check={'max_saved_image_channel_difference':error,'mean_saved_image_channel_difference':mean,
        'elapsed_s':time.time()-begin,'progress_rebuilt':progress,
        'note':'Fields rebuilt conservatively from the recorded source and original paper/fluid parameters; original PNG kept.'}
    d['source_replay_verification']=check;(path/'run.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(check))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--progress',action='store_true')
    a=p.parse_args();replay(a.path,a.progress)
