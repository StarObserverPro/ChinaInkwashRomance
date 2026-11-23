"""Traceable, unretouched museum crops and a reference observation plate."""
from pathlib import Path
import json,hashlib,io,os,tempfile
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BG=(245,242,232)

def text(im,xy,s,size=20,fill=(38,43,40)):
    ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(FONT,size),fill=fill)

def fit(im,size):
    a=im.copy();a.thumbnail(size,Image.Resampling.LANCZOS);return a

def save_png(im,path):
    """Publish a complete, decodable figure after the write has finished."""
    path=Path(path);buffer=io.BytesIO();im.save(buffer,format='PNG')
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.png',delete=False) as f:
        temporary=Path(f.name);f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
    try:
        with Image.open(temporary) as check:check.load()
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

def main():
    p=ROOT/'reference-v014';p.mkdir(exist_ok=True)
    entries=[
        {'key':'wet','artist':'Shitao','title':'Hibiscus, Lotus, and Rock','date':'ca. 1705-7','object_id':41492,
         'original':'reference-v014/shitao_wet_original.jpg','image_url':'https://images.metmuseum.org/CRDImages/as/original/DP-36812-002.jpg',
         'crops':{'wet_stroke':[530,1630,1460,1970],'wet_dot':[1220,2650,1630,3090],'wet_leaf':[360,2620,1150,3450]},
         'observation':'Dark core remains defined; edge spread varies locally. Overlap is visible.'},
        {'key':'dry','artist':'Shitao','title':'Landscape with solitary figure','date':'1660s','object_id':49171,
         'original':'reference-v014/shitao_dry_original.jpg','image_url':'https://images.metmuseum.org/CRDImages/as/original/DP153922.jpg',
         'crops':{'dry_rock':[1510,1910,2445,2580],'dry_tree':[805,1030,1700,1860]},
         'observation':'Short interrupted contacts coexist with coherent dark ridges.'},
        {'key':'feibai','artist':'Zhao Mengfu','title':'Twin Pines, Level Distance','date':'ca. 1310','object_id':40508,
         'original':'reference-copy/zhao_original.jpg','image_url':'https://images.metmuseum.org/CRDImages/as/original/DP153682.jpg',
         'crops':{'zhao_rock':[3184,471,3508,732]},
         'observation':'White gaps break up and rejoin; widths and lengths vary.'}
    ]
    for e in entries:
        raw=ROOT/e['original'];im=Image.open(raw)
        e.update(museum='The Metropolitan Museum of Art',public_domain=True,
            object_url=f"https://www.metmuseum.org/art/collection/search/{e['object_id']}",
            image_acquired='2026-09-21' if e['key']=='feibai' else '2026-09-29',
            catalog_verified='2026-09-29',size=list(im.size),sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            analysis_scope='Qualitative appearance reference. Unknown historic paper sizing, loading, drying and image response; not a parameter calibration.',
            crop_convention='Pillow half-open coordinates. Crop only; no retouching, thresholding, recoloring or generated content.')
        for name,box in e['crops'].items():save_png(im.crop(box),p/(name+'.png'))
    (p/'manifest.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    board=Image.new('RGB',(1530,1190),BG)
    text(board,(28,18),'Real ink on paper | three museum references',30)
    text(board,(28,60),'Original scans and identified crops; these pixels are never fed into the renderer.',18)
    headers=['A | Wet ink: crisp core + local spread','B | Dry texture: intermittent contact','C | Flying white: irregular connected gaps']
    comments=[['A1: an edge can stay quite sharp.','A2: neighboring marks have unequal spread.'],
              ['B1: small breaks follow the rock structure.','B2: fine gaps interrupt, without equal spacing.'],
              ['C1: solid sections alternate with openings.','C2: the white gaps are not long parallel rails.']]
    for i,e in enumerate(entries):
        x=28+i*502;text(board,(x,103),headers[i],19)
        text(board,(x,132),e['artist']+' | '+e['date'],17)
        original=Image.open(ROOT/e['original']).convert('RGB')
        context=fit(original,(465,220));board.paste(context,(x,164))
        text(board,(x,398),e['title'],17)
        if i==0:
            a=fit(Image.open(p/'wet_stroke.png'),(465,205));board.paste(a,(x,439))
            a=fit(Image.open(p/'wet_dot.png'),(290,305));board.paste(a,(x,649))
        elif i==1:
            a=fit(Image.open(p/'dry_rock.png'),(465,345));board.paste(a,(x,439))
            a=fit(Image.open(p/'dry_tree.png'),(250,180));board.paste(a,(x,786))
        else:
            a=Image.open(p/'zhao_rock.png').resize((465,375),Image.Resampling.NEAREST);board.paste(a,(x,439))
            text(board,(x,838),'Nearest-neighbor enlargement',16)
        for j,line in enumerate(comments[i]):text(board,(x,995+j*30),line,17)
        text(board,(x,1068),f"The Met | object {e['object_id']} | Public Domain",16)
    text(board,(28,1130),'Observation is from finished, aged artworks. Physical brush speed, water load and paper parameters are unknown.',18)
    save_png(board,p/'real_reference_analysis.png')
    print(json.dumps({'references':len(entries),'crops':sum(len(e['crops']) for e in entries)}))

if __name__=='__main__':main()
