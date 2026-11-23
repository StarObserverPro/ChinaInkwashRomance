"""Three persistent gestures after the retained Zhao rock study."""
from pathlib import Path
from dataclasses import asdict
import json,math,os
import numpy as np
from PIL import Image
from .reference_rock_study import programs
from .technique_controls import TechniqueBrush,TechniqueSettings,BrushLoading,InkPaperInteraction
from .technique_studies import frame,save_json,save_npz
from .reference_evidence import save_png
from .brush import InkCanvas

def main():
    root=Path(__file__).resolve().parents[1];out=root/'technique-applications/rock';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'technique-r3-config.json').read_text())
    c=InkCanvas(290,210,seed=505,paper_color=(245,240,225));c.density=c.density.astype(float)
    b=TechniqueBrush(c,width_mm=1.25,technique=TechniqueSettings(**cfg['technique']),
        interaction=InkPaperInteraction(**cfg['interaction']),
        loading=BrushLoading(body_saturation=.35,tip_saturation=.35,body_concentration_kg_m3=25,tip_concentration_kg_m3=65))
    gestures=[];states=[];frames=[];next_frame=0.
    def capture(t,brush):
        nonlocal next_frame
        if t>=next_frame:
            frames.append(c.to_image());next_frame=t+.1
    for old in programs():
        a=old['initial'];time=0.;x,y=old['start']
        keys=[frame(0,x*.05,y*.05,-.32,math.degrees(a['tilt']),math.degrees(a['angle']))]
        for p in old['actions']:
            time+=p['duration'];x,y=p['to'];d=p['load']*.65 if p['load']>0 else -.4
            keys.append(frame(time,x*.05,y*.05,d,math.degrees(p['tilt']),math.degrees(p['angle'])))
        spec={'name':old['tag'],'keyframes':keys};gestures.append(spec)
        before=float(b.volume.sum());b.draw_gesture(spec,capture);states.append({'name':old['tag'],
            'before_brush_nl':before*1e12,'after_brush_nl':float(b.volume.sum()*1e12),'diagnostics':b.diagnostics()})
        save_png(c.to_image(),out/f'step-{len(states)}.png');print(old['tag'],flush=True)
    b.relax(.8,capture);b.substrate.check();d=b.diagnostics()
    assert max(abs(d['water_relative_error']),abs(d['pigment_relative_error']))<1e-9
    save_png(c.to_image(),out/'result.png');p=b.substrate
    save_npz(out/'fields.npz',water=p.water,pigment=p.pigment,film=p.film,film_mass=p.film_mass,density=c.density)
    tmp=out/'progress.tmp.gif';frames[0].save(tmp,save_all=True,append_images=frames[1:]+[frames[-1]]*8,duration=90,loop=0)
    with Image.open(tmp) as im:
        for i in range(im.n_frames):im.seek(i);im.load()
    os.replace(tmp,out/'progress.gif')
    save_json(out/'run.json',{'config':cfg,'loading':b.loading,'gestures':gestures,'states':states,'diagnostics':d,
        'control_trace':b.control_trace,'note':'Three authored gestures, same reservoir and paper throughout; no implicit reload. Not recovery of the historical process. v0.14 used different control semantics, so this is not an isolated backend comparison.'})
    print(json.dumps(d),flush=True)

if __name__=='__main__':main()
