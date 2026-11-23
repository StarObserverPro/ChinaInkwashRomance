"""Attributed reference crops, all iterations and one-factor control plates."""
from pathlib import Path
import json,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .reference_evidence import save_png
from .technique_studies import save_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'technique-review'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def text(im,xy,s,size=22,fill='#25322f'):
    ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(FONT,size),fill=fill)
def fit(im,path,box,nearest=False):
    a=Image.open(path).convert('RGB');x,y,w,h=box
    a=ImageOps.contain(a,(w,h),Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    im.paste(a,(x+(w-a.width)//2,y+(h-a.height)//2))
def start(w,h,title,subtitle):
    im=Image.new('RGB',(w,h),'#f8f6ee');text(im,(28,22),title,32);text(im,(28,70),subtitle,18);return im
def main():
    OUT.mkdir(exist_ok=True)
    rounds=[('R0','Old contact, new controls'),('R1','Tilt + curved contact'),('R2','Series resistance'),
        ('R3','Entry particle capture'),('R4','Shared fiber relief'),('R5','Finite neck supply')]
    names=['pulse','dry','side','fold'];labels=['Light / heavy / light','Lower liquid loading','Tip + side concentration','Continuous folded turn']
    im=start(1920,958,'v0.15 | five iterations after the baseline','Same authored gesture and initial loading within each row. No image noise, contrast matching or post-render holes.')
    for i,(tag,note) in enumerate(rounds):
        x=20+i*316;text(im,(x,116),tag,25);text(im,(x,149),note,16)
        for j,name in enumerate(names):
            y=184+j*166
            fit(im,ROOT/f'technique-rounds/r{i}/{name}/result.png',(x,y,304,120),True)
            if i==0:text(im,(x+4,y+128),labels[j],16)
    text(im,(28,886),'R4: smoother rails; not retained in R5. R5 branches from R3 and changes only the neck-feed closure.',19)
    text(im,(28,918),'Remaining failures: long parallel gaps, overly circular breakup marks, and sensitivity of fine detail to the integration step.',18)
    save_png(im,OUT/'all_rounds.png')

    # Retain original pixels; all coordinates refer to Sue Yang's step-6 photo.
    photo=Image.open(ROOT/'technique-references/mfa-sue-yang-step06.jpg')
    crops={'leaf':(436,324,860,645),'bone':(18,122,421,303)}
    for name,box in crops.items():save_png(photo.crop(box),OUT/f'real_{name}.png')
    im=start(1500,1000,'Real references / baseline / R5','Reference scans and photos guide qualitative critique. Physical scale, paper, lighting and historical motion are not matched.')
    for i,title in enumerate(['Real reference','R0 | original contact model','R5 | latest experimental model']):text(im,(28+492*i,120),title,24)
    rows=[('leaf',OUT/'real_leaf.png','Sue Yang / MFA: leaf marks'),('bone',OUT/'real_bone.png','Sue Yang / MFA: bamboo stroke'),('dry',ROOT/'reference-v014/dry_rock.png','Shitao / The Met: dry rock texture')]
    for j,(name,ref,caption) in enumerate(rows):
        y=162+j*258;fit(im,ref,(28,y,464,210));text(im,(28,y+216),caption,16)
        for i,r in enumerate([0,5]):fit(im,ROOT/f'technique-rounds/r{r}/{name}/result.png',(520+492*i,y,456,210),True)
    text(im,(28,946),'Observed gap: real marks combine coherent ink, irregular openings and a more decisive taper. R5 has not yet reproduced that combination.',17)
    save_png(im,OUT/'reference_comparison.png')
    save_json(OUT/'reference_crops.json',{'original':'technique-references/mfa-sue-yang-step06.jpg','crops':crops,
        'source':'https://www.mfa.org/programs/community-programs/art-making-at-home/chinese-brush-painting',
        'credit':'Artwork and instructions courtesy of Sue Yang / Museum of Fine Arts Boston',
        'transform':'Crop and display resize only. No recolor or retouch. No open license asserted.'})

    pairs=[('Compression displacement','pressure_low','pressure_high','0.55 x | lighter','1.20 x | heavier'),
        ('Travel speed','speed_slow','speed_fast','0.65 x | slower','1.70 x | faster'),
        ('Initial liquid saturation','water_low','water_high','0.33 | drier','0.75 | wetter'),
        ('Pigment concentration','ink_light','ink_dark','0.45 x | dilute','1.30 x | concentrated'),
        ('In-plane paper permeability','paper_tight','paper_open','2.5e-14 m2','3.0e-13 m2')]
    im=start(1440,1290,'One control at a time | R5','Same leaf gesture, seed, optics and remaining controls. More liquid also loads more pigment when concentration is held fixed.')
    records=[]
    for j,(title,a,b,la,lb) in enumerate(pairs):
        y=120+j*216;text(im,(28,y),title,23)
        for i,(name,lab) in enumerate([(a,la),(b,lb)]):
            p=ROOT/'technique-controls-r5'/name;data=json.loads((p/'run.json').read_text());v=data['diagnostics']['deposited_nl_including_rupture']
            fit(im,p/'result.png',(28+704*i,y+31,672,147),True)
            text(im,(40+704*i,y+178),f'{lab}   |   paper received {v:.2f} nL',18)
            records.append({'control':title,'variant':name,'label':lab,'water_to_paper_nl':v,
                'initial_loading':data['loading'],'stroke_duration_s':data['lift_time_s']})
    text(im,(28,1230),'Concentration comparisons hold viscosity fixed. These are reproducible sensitivity studies, not a calibrated chart of real brushes or papers.',17)
    save_png(im,OUT/'controls.png');save_json(OUT/'control_results.json',records)

    im=start(1440,656,'Back to the rock | three persistent gestures','Retained hand-designed ridge and two inner edges. No reservoir reset between strokes. This does not recover the historical stroke order.')
    panels=[(ROOT/'reference-v014/zhao_rock.png','Zhao Mengfu / The Met'),(ROOT/'reference-rock/r5/result.png','v0.14 | old action semantics'),
        (ROOT/'technique-applications/rock/result.png','v0.15 / R3 | mm + seconds')]
    for i,(p,l) in enumerate(panels):
        fit(im,p,(28+i*470,139,444,397),True);text(im,(28+i*470,554),l,20)
    text(im,(28,610),'The new rock is still too continuous and cable-like. Different loading/control semantics prevent an isolated backend comparison here.',17)
    save_png(im,OUT/'rock_comparison.png')
    print(OUT)

if __name__=='__main__':main()
