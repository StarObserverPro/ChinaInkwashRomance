"""Contact sheets for inspecting solver outputs; no edits to simulated marks."""
from pathlib import Path
import argparse,json,math
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1]
def font(size):
    p=ROOT/'assets/NotoSansCJKsc-Regular.otf'
    return ImageFont.truetype(str(p) if p.exists() else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def board(paths,target,title,columns=4,cell=(420,370)):
    paths=list(paths);cw,ch=cell;head=95
    out=Image.new('RGB',(cw*columns,ch*math.ceil(len(paths)/columns)+head),(250,248,241));d=ImageDraw.Draw(out)
    d.text((22,14),title,font=font(29),fill=(37,40,36))
    d.text((22,56),'同一物理内核 · 仅改变动作、装墨与叠笔 · 图像等比放大',font=font(17),fill=(88,89,83))
    for i,p in enumerate(paths):
        p=Path(p);r=json.loads((p/'run.json').read_text());x=(i%columns)*cw;y=head+(i//columns)*ch
        d.text((x+12,y+6),r['id']+'  '+r['title'],font=font(21),fill=(35,37,32))
        im=Image.open(p/'result.png').convert('RGB');scale=min((cw-22)/im.width,(ch-91)/im.height)
        im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
        out.paste(im,(x+(cw-im.width)//2,y+41))
        note=f"R{r['round']} · seed {r['seed']} · {r['recipe']['width_mm']:g} mm"
        if r['recipe']['status']=='approximation':note+=' · 近似'
        d.text((x+12,y+ch-35),note,font=font(15),fill=(100,101,93))
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);out.save(target);return str(target)
def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,default=0);p.add_argument('--ids',default='all');p.add_argument('--output',default='atlas18/review.png');p.add_argument('--columns',type=int,default=4);a=p.parse_args()
    root=ROOT/f'atlas18/r{a.round}';ids=sorted(d.name for d in root.iterdir()) if a.ids=='all' else a.ids.split(',')
    paths=[root/i/'s0' for i in ids if (root/i/'s0/run.json').exists()]
    print(board(paths,ROOT/a.output,f'山水笔法 · R{a.round}',a.columns))
if __name__=='__main__':main()
