"""Controlled experiments. Plates deliberately keep the varied parameter isolated."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .brush import InkCanvas
from .physics import physics_demo
from .landscape import Painter,Recipe

def _font(size):
    # Plate text is ASCII. Use common fonts on Linux, Windows and macOS;
    # Pillow's bundled font still makes the studies runnable without them.
    for font in ('DejaVuSans.ttf', 'Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 'C:/Windows/Fonts/arial.ttf',
                 '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 9 and 10.0 have an unsized bitmap fallback.
        return ImageFont.load_default()


def label(im,xy,text,size=19,color=(61,64,59)):
    ImageDraw.Draw(im).text(xy,text,font=_font(size),fill=color)

def brush_atlas(output):
    im=Image.new('RGB',(1400,1020),(244,240,228))
    label(im,(45,28),'01 / THE BRUSH AS A STATE',32)
    label(im,(46,77),'Same path and seed; one parameter changes in each row.',18)
    rows=[('Pressure',[.2,.5,.95]),('Water',[.05,.45,.95]),('Dry contact',[.0,.45,.95]),('Side angle',[0,.6,1.2])]
    records=[]
    for row,(name,values) in enumerate(rows):
        yy=145+row*206; label(im,(46,yy),name,22)
        for col,value in enumerate(values):
            c=InkCanvas(365,135,seed=100)
            kw=dict(width=23,ink=.65,water=.25,dry=.3,pressure=[.15,.75,.65,.1],tilt=0)
            if row==0: kw['pressure']=[value]*4
            elif row==1: kw['water']=value
            elif row==2: kw['dry']=value
            else: kw['tilt']=value
            c.stroke([(30,79),(113,37),(218,82),(333,54)],tag=f'control/{name}',**kw)
            x=225+col*383; im.paste(c.to_image(),(x,yy-8)); label(im,(x+18,yy+129),f'{value:.2f}',18)
            records.append(c.strokes[0])
    im.save(output/'brush_atlas.png')
    (output/'brush_atlas_strokes.json').write_text(json.dumps(records,indent=2),encoding='utf8')

def forms(output):
    im=Image.new('RGB',(1500,1120),(244,240,228))
    label(im,(45,28),'02 / FORM BEFORE TEXTURE',32)
    label(im,(45,79),'Top: controlled idioms. Bottom: terrain, grouping, scale cues.',18)
    for i,texture in enumerate(['pima','axe','raindrop']):
        p=Painter(Recipe(round=3,seed=120+i,width=1000,height=1200))
        # Same nominal silhouette, distinct directional vocabularies.
        p.mountain([(38,343),(66,220),(111,245),(158,105),(192,133),(218,117),(258,206),(294,231),(337,358)],388,.62,texture,310,'study-'+texture)
        crop=p.c.to_image().crop((0,60,380,440)).resize((430,430))
        im.paste(crop,(45+i*485,140)); label(im,(56+i*485,585),['A / Pima: long, bending fibers','B / Axe: press, turn, release','C / Raindrop: short grouped marks'][i],18)
    p=Painter(Recipe(round=3,seed=2026,width=1000,height=1200))
    p.rock(100,260,105,90,.65); p.rock(200,285,175,130,.65);p.rock(300,305,130,78,.5)
    p.pine(585,333,1.38,-.3,.75);p.pavilion(787,270,1.5)
    strip=p.c.to_image().crop((0,90,910,370)).resize((1380,425))
    im.paste(strip,(55,648)); label(im,(67,1052),'D / Unequal rock groups                           E / Trunk > branch > needles          F / Pavilion',18)
    im.save(output/'forms.png')

def physics_plate(output):
    frames=physics_demo(shape=(96,128),seed=42,steps=240,frames=5)
    im=Image.new('RGB',(1500,800),(244,240,228)); label(im,(42,24),'03 / TRANSPORT, ABSORPTION, DEPOSITION',30)
    label(im,(42,75),'Conservative small-grid experiment; not the full-landscape renderer.',18)
    maxd=max(float(f['density'].max()) for f in frames); maxw=max(float(f['water'].max()) for f in frames)
    stats=[]
    for k,f in enumerate(frames):
        for row,field in enumerate(['water','density']):
            a=np.clip(f[field]/(maxw if row==0 else maxd),0,1)
            rgb=(np.array([244,240,228])[None,None,:]*(1-a[:,:,None]) + np.array([87,125,141] if row==0 else [29,32,28])*a[:,:,None]).astype('uint8')
            tile=Image.fromarray(rgb).resize((275,207),Image.Resampling.NEAREST)
            im.paste(tile,(40+k*292,166+row*290))
        label(im,(44+k*292,390),f't = {f["time"]:.1f}',18)
        mass=float(np.sum(f['suspended']+f['deposited']))
        label(im,(44+k*292,678),f'ink mass {mass:.5f}',16)
        stats.append({'time':f['time'],'ink_mass':mass,'water_mass':float(f['water'].sum())})
    label(im,(42,126),'WATER / same scale in every frame',17); label(im,(42,427),'TOTAL INK / suspended + fixed, same scale in every frame',17)
    label(im,(42,744),'Transport redistributes ink. Deposition transfers mass between two states. Evaporation removes water only.',18)
    im.save(output/'physics.png');(output/'physics_metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf8')

def make_studies(output):
    output.mkdir(parents=True,exist_ok=True)
    brush_atlas(output); forms(output); physics_plate(output)
    print('Created brush_atlas.png, forms.png, physics.png')
