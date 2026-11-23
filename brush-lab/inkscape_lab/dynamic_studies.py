"""Experiments that separate dynamic modulation, segmentation and contact events."""
from pathlib import Path
from copy import deepcopy
import json
import textwrap
import numpy as np
from PIL import Image,ImageDraw
from .brush import InkCanvas
from .dynamic import dynamic_stroke
from .studies import label

PAPER=(244,240,228)

def wave(mean,amplitude,cycles=3,phase=0):
    return {'knots':[[0,mean],[1,mean]],'waves':[{'amp':amplitude,'cycles':cycles,'phase':phase}]}

def base_spec(points,seed=73):
    return {'points':[[float(x),float(y)] for x,y in points], 'width':27.,'ink':.68,'water':.38,'dry':.38,
            'pressure':.65,'speed':145.,'tilt':.35,'seed':seed,'path_mode':'smooth',
            'boundary':{'enter':{'mode':'point','span':.065},'leave':{'mode':'lift','span':.13}}}

def tile(spec,width=1050,height=145,chunk_size=None):
    c=InkCanvas(width,height,seed=87)
    rec=dynamic_stroke(c,spec,chunk_size=chunk_size)
    return c.to_image(),rec,c

def continuity(out):
    im=Image.new('RGB',(1440,1140),PAPER)
    label(im,(44,27),'04 / ONE STROKE, CHANGING STATES',31)
    label(im,(44,75),'The trajectory is shared. Modulation, physical lifting and numerical chunks are distinct.',18)
    xs=np.linspace(45,995,21); ys=66+12*np.sin(np.linspace(0,3.7,21)); path=list(zip(xs,ys))
    fixed=base_spec(path);fixed['flow_rate']=28
    dynamic=deepcopy(fixed);dynamic.update(pressure=wave(.60,.24),speed=wave(145,50,phase=np.pi),tilt=wave(.48,.34,1.5))
    specs=[fixed,dynamic,deepcopy(dynamic),deepcopy(dynamic),deepcopy(dynamic)]
    specs[3]['air_intervals']=[[.28,.33],[.62,.67]]
    specs[4].update(dry=.95,water=.07)
    names=['A / Steady control','B / Continuous modulation','C / 8 restarted strokes','D / 2 true air intervals','E / Dry contact, no lift']
    details=['Constant pressure and speed','Pressure up, speed down; same brush memory','Each segment gets a new entry / exit / ink load','Brush lifts; remaining ink is carried forward','White gaps belong to the mark, not the contact topology']
    records=[]
    for i,spec in enumerate(specs):
        yy=135+i*181
        label(im,(44,yy),names[i],20)
        for k,line in enumerate(textwrap.wrap(details[i],33)):
            label(im,(44,yy+34+21*k),line,15)
        if i!=2:
            pic,record,c=tile(spec);records.append(record)
        else:
            c=InkCanvas(1050,145,seed=87)
            # Each physical restart has its own boundary and freshly initialized load.
            for k in range(8):
                a,b=k/8,(k+1)/8; u=np.linspace(a,b,6)
                pts=list(zip(np.interp(u,np.linspace(0,1,len(xs)),xs),np.interp(u,np.linspace(0,1,len(ys)),ys)))
                frag=base_spec(pts,73+k);frag['flow_rate']=28
                frag['pressure']={'knots':[[float((q-a)/(b-a)),float(.60+.24*np.sin(q*6*np.pi))] for q in u]}
                frag['speed']={'knots':[[float((q-a)/(b-a)),float(145+50*np.sin(q*6*np.pi+np.pi))] for q in u]}
                frag['tilt']={'knots':[[float((q-a)/(b-a)),float(.48+.34*np.sin(q*3*np.pi))] for q in u]}
                frag['boundary']={'enter':{'mode':'point','span':.12},'leave':{'mode':'lift','span':.16}}
                record=dynamic_stroke(c,frag);records.append(record)
            pic=c.to_image()
        im.paste(pic,(341,yy-10))
    label(im,(44,1063),'A numerical chunk does not restart a brush. The same B, evaluated in chunks, must be pixel-identical.',17)
    im.save(out/'dynamics_01_continuity.png')
    return specs[1],records

def boundaries(out):
    im=Image.new('RGB',(1440,980),PAPER)
    label(im,(44,25),'05 / ENTRY, DWELL AND RELEASE',31)
    label(im,(44,72),'An endpoint is a state transition. Concealed entry / return need an actual reversal of the path.',18)
    modes=[
      ('Point > lift','Zero pressure at both contact boundaries',[(50,82),(180,62),(400,73),(580,68)],'point','lift',None),
      ('Press > lift','Faster engagement, then gradual release',[(50,82),(180,62),(400,73),(580,68)],'press','lift',None),
      ('Loaded > loaded','A cropped in-contact segment, not a physical birth or death',[(50,82),(180,62),(400,73),(580,68)],'loaded','loaded',None),
      ('Reverse entry > lift','Reverse, roll and continue: a geometric recipe',[(80,83),(46,81),(61,72),(180,62),(400,73),(580,68)],'point','lift',None),
      ('Pause > lift','0.7 s at u=0.48; motion stops, state continues',[(50,82),(180,62),(400,73),(580,68)],'point','lift',[{'u':.48,'duration':.7}]),
      ('Turn back > lift','Return along a short arc before lifting',[(50,82),(180,62),(460,73),(580,68),(566,57),(537,65)],'point','lift',None),
    ]
    records=[]
    for i,(name,detail,pts,enter,leave,dwells) in enumerate(modes):
        col=i%2;row=i//2;x=42+col*714;y=134+row*272
        label(im,(x,y),name,23);label(im,(x,y+37),detail,14)
        spec=base_spec(pts,121);spec.update(width=32,flow_rate=26)
        spec['boundary']={'enter':{'mode':enter,'span':.035 if enter=='press' else .085},'leave':{'mode':leave,'span':.16}}
        if dwells: spec['dwells']=dwells
        if i==3: spec['tilt']={'knots':[[0,.9],[.1,.4],[1,.4]]}
        pic,rec,_=tile(spec,660,145);im.paste(pic,(x,y+67));records.append(rec)
    label(im,(44,949),'All boundary names are engineering recipes, not a claim to reconstruct a historical artist\'s motion.',16)
    im.save(out/'dynamics_02_boundaries.png')
    return records

def cun_study(out,revision=0):
    im=Image.new('RGB',(1500,840),PAPER)
    label(im,(42,24),f'06 / DYNAMIC CUN - STUDY {revision}',31)
    label(im,(42,72),'Group direction follows the rock. Within each stroke, pressure, speed and orientation evolve together.',17)
    records=[]
    for col,style in enumerate(['pima','zhedai','axe']):
        c=InkCanvas(445,580,seed=500+col);r=np.random.default_rng(701+col)
        shape=[(34,512),(54,391),(87,360),(115,204),(158,228),(199,92),(235,103),(257,218),(301,257),(339,397),(410,515)]
        c.wash(shape,ink=.09,water=.7,tag=style+'/wash',fade_bottom=100)
        for j in range(1,len(shape)-1):
            if j%4!=0: c.stroke(shape[j-1:j+1],width=1.3,ink=.38,water=.2,dry=.5,tag=style+'/contour')
        for k in range(21):
            group=k//7; offset=k%7
            x=105+group*93+offset*4+r.normal(0,2)
            y=[270,144,310][group]+r.uniform(2,61)
            length=r.uniform(64,126)
            if style=='pima':
                pts=[(x,y),(x+5,y+length*.3),(x-7,y+length*.7),(x+5,y+length)]
                spec=base_spec(pts,1000+k);spec.update(width=3.2,ink=.55,water=.18,dry=.57)
                spec.update(pressure=wave(.55,.27,cycles=1.8,phase=k*.9),speed=wave(98,28,cycles=1.8,phase=k*.9+np.pi))
            elif style=='zhedai':
                pts=[(x-22,y),(x+25,y+5),(x+29,y+length*.52),(x+58,y+length*.60)]
                spec=base_spec(pts,1100+k);spec.update(width=4.5,ink=.55,water=.12,dry=.63,path_mode='polyline')
                spec['pressure']={'knots':[[0,.32],[.32,.52],[.40,.94],[.52,.46],[1,.30]]}
                spec['speed']={'knots':[[0,115],[.32,105],[.40,38],[.52,100],[1,140]]}
            else:
                pts=[(x,y),(x-8,y+length*.17),(x-29,y+length*.49)]
                spec=base_spec(pts,1200+k);spec.update(width=12.,ink=.56,water=.15,dry=.62,tilt=.95)
                spec['pressure']={'knots':[[0,.7],[.17,.96],[.5,.43],[1,.035]]}
                spec['speed']={'knots':[[0,52],[.2,70],[1,215]]}
                spec['boundary']={'enter':{'mode':'press','span':.045},'leave':{'mode':'lift','span':.2}}
            if revision>=1:
                # R0 was too faint in thin strokes and too regular at the turns.
                spec['width']*={'pima':1.44,'zhedai':1.10,'axe':.86}[style]
                spec['dry']*=.80
                spec['flow_rate']={'pima':32.,'zhedai':25.,'axe':15.}[style]
                if style=='zhedai':
                    # Shorter, varied ledges preserve the face instead of poking outside it.
                    spec['points']=[[x-18,y],[x+float(r.uniform(10,20)),y+float(r.uniform(-3,8))],
                       [x+float(r.uniform(15,23)),y+length*.45],[x+float(r.uniform(28,42)),y+length*.58]]
                # Project only path controls back into the local face; not a global erase mask.
                for pt in spec['points']:
                    left=float(np.interp(pt[1],[92,204,228,360,391,512],[199,115,107,87,54,34]))
                    right=float(np.interp(pt[1],[103,218,257,397,515],[235,257,301,339,410]))
                    pt[0]=min(right-7,max(left+7,pt[0]))
            spec['tag']=style+'/dynamic-cun';rec=dynamic_stroke(c,spec);records.append(rec)
        im.paste(c.to_image(),(45+col*490,132))
        label(im,(49+col*490,739),['Pima / long pressure rhythm','Zhedai / turn without lifting','Axe / press, drag, accelerate'][col],20)
        label(im,(49+col*490,776),['Correlated waviness, no repeated caps','A corner changes derivatives, not brush identity','Load and contact width decline along the stroke'][col],14)
    im.save(out/f'dynamics_03_cun_r{revision}.png')
    return records

def ledger_demo(out):
    c=InkCanvas(1200,800,seed=909)
    cases=[]
    for i in range(4):
        xx=np.linspace(90,1110,19); yy=140+i*173+19*np.sin(np.linspace(0,4,19))
        spec=base_spec(list(zip(xx,yy)),seed=812+i)
        spec.update(width=28,flow_rate=26,pressure=wave(.60,.26,cycles=3.2,phase=.3),speed=wave(150,50,cycles=3.2,phase=np.pi+.3),tilt=wave(.5,.32,cycles=1.5))
        if i==1: spec['air_intervals']=[[.35,.40],[.70,.75]]
        if i==2: spec['dwells']=[{'u':.48,'duration':.8}]
        if i==3: spec.update(dry=.95,water=.08)
        spec['tag']=['continuous/pima','lifted/retained-load','dwell/continuous-time','dry/no-lift'][i]
        dynamic_stroke(c,spec)
    c.save(out/'dynamics_04_ledger.png')
    data={'width':c.width,'height':c.height,'recipe':{'seed':909},'paper_color':list(c.paper_color),
          'renderer_version':'0.2.0','image':'dynamics_04_ledger.png','strokes':c.strokes}
    (out/'dynamics_strokes.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    return data

def make_dynamic_studies(out,revision=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if revision==0:
        _,records=continuity(out);records+=boundaries(out)
        (out/'dynamics_experiments.json').write_text(json.dumps({'strokes':records},ensure_ascii=False,separators=(',',':')),encoding='utf8')
    cun_study(out,revision)
    ledger_demo(out)
    print('Dynamic studies and per-sample state ledger written to',out)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('examples'));p.add_argument('--revision',type=int,default=0)
    args=p.parse_args();make_dynamic_studies(args.output,args.revision)
