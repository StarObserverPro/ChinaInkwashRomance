"""Fixed nine actions, correlated dry-brush material study v0.9.

python -m inkscape_lab.dry_studies --output dry-final
python -m inkscape_lab.dry_studies --replay dry-final/ledger.json.gz
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .brush import InkCanvas
from .cun import ContactSettings
from .dry_brush import CorrelatedDryBrush
from .form_driver import compile_form
from .reference_iteration import make_programs, settings, make_steps, SIZE, PAPER, image_hash
from .studies import label

VERSION='correlated-dry-study-0.9.0'

def render(programs=None,cfg=None,texture_seed=909,paper_seed=50509,correlated=True,new_paper=True):
    c=InkCanvas(*SIZE,seed=505,paper_color=PAPER)
    c.density=c.density.astype(np.float64)
    b=CorrelatedDryBrush(c,cfg or settings(),seed=606,texture_seed=texture_seed,
                         paper_seed=paper_seed,correlated=correlated,new_paper=new_paper)
    frames=[]
    for item in make_programs() if programs is None else programs:
        b.draw_driver(compile_form(item['program'],item['curves']))
        frames.append(c.to_image())
    return c,b,frames

def save_run(output,c,b,frames,programs):
    p=Path(output);p.mkdir(parents=True,exist_ok=True)
    options={k:getattr(b,k) for k in ('texture_seed','paper_seed','correlated','new_paper')}
    data={'version':VERSION,'canvas':{'size':list(SIZE),'paper':list(PAPER),'seed':505,'brush_seed':606},
          'material':options,'settings':asdict(b.settings),'programs':programs,
          'records':b.records,'domain_log':b.domain_log,'diagnostics':b.diagnostics(),
          'image_sha256':image_hash(c)}
    raw=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    (p/'ledger.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    (p/'form_actions.json').write_text(json.dumps(programs,ensure_ascii=False,indent=2),encoding='utf-8')
    (p/'diagnostics.json').write_text(json.dumps(data['diagnostics'],indent=2),encoding='utf-8')
    c.save(p/'result.png')
    # Rendering paper tone and physical contact depth are intentionally named
    # separately. The display background never gates pigment transfer.
    np.savez_compressed(p/'fields.npz',density=c.density,display_paper=c.paper,
                        contact_depth=b.tooth_height,domain_ids=b.domain_ids,
                        base_offsets=b.base_offsets)
    for i,frame in enumerate(frames,1):frame.save(p/f'step_{i:02d}.png')
    make_steps(frames,p/'steps.png')
    return data

def replay(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
    if data['version']!=VERSION or data['canvas']!={'size':list(SIZE),'paper':list(PAPER),'seed':505,'brush_seed':606}:
        raise ValueError('unsupported study or canvas')
    c,b,frames=render(data['programs'],ContactSettings(**data['settings']),**data['material'])
    if image_hash(c)!=data['image_sha256'] or b.records!=data['records'] or b.domain_log!=data['domain_log']:
        raise ValueError('image or material history differs')
    return c,b,frames

def comparison(before,after,path):
    plate=Image.new('RGB',(1360,975),PAPER)
    label(plate,(28,18),'SAME NINE ACTIONS / DIFFERENT MATERIAL',28)
    for col,(im,title) in enumerate([(before,'v0.8 / Repeated fine gaps'),(after,'v0.9 / Correlated contact and supply')]):
        x=20+680*col
        label(plate,(x,66),title,21)
        plate.paste(im,(x,104))
        label(plate,(x,650),'Left wall / same 2x crop',18)
        crop=im.crop((25,260,167,397)).resize((284,274),Image.Resampling.NEAREST)
        plate.paste(crop,(x,686))
        label(plate,(x+310,650),'Crown / same 2x crop',18)
        crop=im.crop((238,54,393,180)).resize((310,252),Image.Resampling.NEAREST)
        plate.paste(crop,(x+310,686))
    plate.save(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='dry-final')
    parser.add_argument('--replay')
    parser.add_argument('--texture-seed',type=int,default=909)
    args=parser.parse_args()
    if args.replay:
        c,b,_=replay(args.replay)
        print(json.dumps({'replay':'PASS','image_sha256':image_hash(c)},ensure_ascii=False));return
    programs=make_programs()
    c,b,frames=render(programs,texture_seed=args.texture_seed)
    data=save_run(args.output,c,b,frames,programs)
    comparison(Image.open('iteration-final/result.png').convert('RGB'),c.to_image(),Path(args.output)/'comparison.png')
    print(json.dumps({'image_sha256':data['image_sha256'],**b.diagnostics()},ensure_ascii=False))

if __name__=='__main__':main()
