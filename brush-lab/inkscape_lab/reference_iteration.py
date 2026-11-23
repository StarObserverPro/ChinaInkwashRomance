"""v0.8: joined contact patches and explicit side-brush pose after Zhao.

Keeps the complete v0.7 study/replay module. Reference pixels enter only the
comparison plate. All marks still come from contact with finite ink inventory.
Run: python -m inkscape_lab.reference_iteration --output iteration-output
Replay: python -m inkscape_lab.reference_iteration --replay iteration-output/ledger.json.gz
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from .brush import InkCanvas
from .cun import ContactSettings
from .form_driver import compile_form
from .reference_studies import ToothContactBrush, specs as old_specs
from .reference_studies import PAPER, SIZE, SCALE, image_hash
from .studies import label

VERSION = "zhao-reference-iteration-0.8.0"


class JoinedContactBrush(ToothContactBrush):
    """Pressure broadens material contact domains until adjacent domains join.

    The bunch is a finite patch of bristle tips, not one thin bristle. Its
    irregular boundary is fixed in brush coordinates. Pressure/water determine
    footprint and conductivity before ink leaves its reservoir. There is no
    silhouette mask, post-render threshold, newly randomized path or hatch.
    """
    version = "joined-contact-study-0.8.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rng = np.random.default_rng(self.seed + 808)
        self.patch_phase = rng.uniform(-math.pi, math.pi, self.settings.bunches)
        self.patch_scale = rng.uniform(.82, 1.22, self.settings.bunches)
        # A laid-over tuft is a shallow fan. A disk keeps many more tips along
        # the middle of every travelled strip and produced a rounded, shaded
        # cable even after its individual patches were joined. Fan coordinates
        # remain attached to the brush; they are not additional drawing paths.
        transverse = np.linspace(-.96, .96, self.settings.bunches)
        transverse += rng.uniform(-.014, .014, self.settings.bunches)
        self.offsets[:, 0] = transverse
        self.offsets[:, 1] *= .34
        self.rest = .045 + .19 * np.abs(transverse) ** 3 + rng.uniform(-.016,.016,self.settings.bunches)
        fine = self.canvas.paper.astype(float) - .5
        self.tooth_height = np.clip(.46 + .65 * fine + .35 * gaussian_filter(fine, 1.5), 0., 1.)

    def _paper_contact(self, yy, xx, compression, wet):
        height = map_coordinates(self.tooth_height, [yy.ravel(), xx.ravel()],
                                 order=1, mode="constant", cval=.5).reshape(xx.shape)
        reach = .20 + .50 * wet + .42 * math.sqrt(compression)
        return np.clip((reach - height + .05) / .10, 0., 1.)

    def _bunch_footprint(self, index, center, compression, wet, side, angle):
        size = self.patch_scale[index]
        sx = size * (.80 + 4.9 * compression ** .72 + .32 * side)
        sy = size * (.68 + 3.9 * compression ** .70)
        pad = math.ceil(1.25 * max(sx, sy) + 2.)
        x, y = center
        x0, y0 = math.floor(x) - pad, math.floor(y) - pad
        yy, xx = np.mgrid[y0:y0 + 2 * pad + 2, x0:x0 + 2 * pad + 2]
        co, si = math.cos(angle), math.sin(angle)
        dx, dy = xx - x, yy - y
        u, v = (dx * co + dy * si) / sx, (-dx * si + dy * co) / sy
        radius, phi = np.hypot(u, v), np.arctan2(v, u)
        phase = self.patch_phase[index]
        boundary = 1. + .12 * np.cos(3. * phi + phase) + .07 * np.sin(5. * phi - phase)
        edge = .48 / min(sx, sy)
        envelope = np.clip((boundary + edge - radius) / (2. * edge), 0., 1.)
        kernel = envelope * self._paper_contact(yy, xx, compression, wet)
        area = float(kernel.sum())
        return x0, y0, kernel, area, area / max(float(envelope.sum()), 1e-15)


def specs():
    out = copy.deepcopy(old_specs())
    # Brush azimuth is authored for the plane being drawn. It no longer has
    # to chase the hand tangent at every corner. Angles below are in degrees.
    out[0]["angles"] = [5, 4, 6, 3, 25, 35, 20, 20, 2, 6, 0, 14, 7, 0, -5, -5]
    out[1]["angles"] = [3, -4, 3, 4, -4, -10, -12, -8, -8]
    out[2]["angles"] = [5, 3, 3, 3]
    out[3]["angles"] = [0, -10, -8, -5, -2, -2]
    out[4]["angles"] = [0, -10, -10]
    out[5]["angles"] = [2, -8, -10, -10]
    # Strength is moved to form-bearing turns, with light connecting motion.
    def load_change(index, loads):
        nodes = out[index]["nodes"]
        out[index]["nodes"] = [(n[0], n[1], loads[i], *n[3:]) for i, n in enumerate(nodes)]
    load_change(0, [.39,.52,.30,.22,.51,.48,.38,.48,.43,.30,.21,.25,.22,.23,.25,0.])
    load_change(1, [.83,.95,.93,.78,.89,.57,.33,.28,0.])
    load_change(2, [.52,.71,.59,0.])
    load_change(3, [.23,.52,.78,.31,.43,0.])
    load_change(4, [.19,.37,0.])
    load_change(5, [.23,.51,.43,0.])
    out[2]["reload"] = [.73,.31]
    out[2]["nodes"][2]=(142,54,.59,102,.80,[130,66],"内棱向冠脊汇拢")
    out[2]["nodes"][3]=(153,47,0.,99,.30,None,"在冠脊接合处提收")
    out[3]["reload"] = [.68,.27]
    out[1]["reload"] = [1.,.30]
    for i,tilt in zip(range(9,16),[.42,.25,.30,.20,.22,.27,.20]):
        n=out[0]["nodes"][i]
        out[0]["nodes"][i]=(*n[:4],tilt,*n[5:])
    # The source belly has an outward hook, a short return ledge and a lower
    # bearing corner. Keep these turns instead of reducing them to one S arc.
    out[3].update(start=[161,124],angles=[0,-7,-5,-2,-3,0,0,0],nodes=[
        (172,138,.26,103,.59,None,"裂面上端斜入，先轻后实"),
        (183,154,.46,94,.88,None,"上棱向右下突出"),
        (169,162,.36,86,.75,None,"凸棱向内短折，形成横向阶缘"),
        (173,178,.66,85,1.03,None,"阶缘压下，前侧面承担重墨"),
        (163,193,.23,98,.32,None,"侧面向内收进，连接段减载收锋"),
        (147,207,.24,99,.30,None,"下腹换向，细笔接住上段的斜势"),
        (146,216,.43,94,.81,None,"下部小石棱短按"),
        (157,219,0.,114,.26,None,"向右提收，与底缘之间留白")])
    out[5].update(start=[224,197],angles=[0,-7,-8,-5,-5],nodes=[
        (238,211,.25,103,.65,None,"右足内侧短棱斜入"),
        (250,207,.38,94,.82,None,"短棱向外侧汇合"),
        (244,223,.40,91,.83,None,"石足折下承重"),
        (230,231,.31,101,.72,None,"石足向内收束"),
        (215,234,0.,114,.25,None,"在底口之前提收")])
    # These short marks correspond to distinct visible planes in the source.
    # They are authored once; none is a parallel offset or procedural hatch.
    out.insert(3,{"role":"左肩内折","start":[66,124],"why":"肩下的小内折分开左壁与石腹。",
        "angles":[-3,1,1],"nodes":[
        (76,125,.25,107,.63,None,"肩下内缘短横出"),
        (71,141,.35,112,.79,None,"凹进的面棱折下"),
        (57,155,0.,124,.26,[62,148],"在侧壁前提收，留出窄白面")]})
    out.append({"role":"腹左小层缘","start":[114,173],"why":"一处短层缘把空白腹面向左下翻，不沿外轮廓排线。",
        "angles":[0,-10,-10,-10],"nodes":[
        (127,181,.24,103,.65,None,"轻触石腹局部层缘"),
        (141,178,.30,96,.74,None,"短层缘向上折回"),
        (144,173,.31,95,.65,None,"层缘短棱向上收束"),
        (148,169,0.,114,.25,None,"提收在主内棱之前")]})
    out.append({"role":"上腹短裂","start":[190,120],"why":"主内棱上方独立的小裂面，保持大片空白。",
        "angles":[-3,-3,-3],"nodes":[
        (195,129,.23,111,.61,None,"短裂面轻入"),
        (201,138,.35,103,.78,None,"裂面压向右下"),
        (205,146,0.,124,.26,None,"短促提收")]})
    return out


def make_programs(specifications=None, max_dt=1/240, spatial_step=.35):
    programs, previous = [], None
    for spec in specifications or specs():
        start = [v * SCALE for v in spec["start"]]
        first_angle = math.radians(spec["angles"][0])
        p = {"tag": spec["role"], "start": start,
             "initial": {"load":0., "angle":first_angle, "tilt":.5},
             "max_dt":max_dt, "spatial_step":spatial_step, "actions":[]}
        curves = []
        if previous is not None:
            p.update(start=previous["to"], initial={k:previous[k] for k in ("load","angle","tilt")})
            if "reload" in spec:
                p["actions"].append({"kind":"reload", "duration":.25, "load":0.,
                    "angle":previous["angle"], "tilt":previous["tilt"],
                    "ink_load":spec["reload"][0], "water_load":spec["reload"][1],
                    "form_event":"为独立结构笔明确补墨；材料形变历史保留"})
                curves.append(None)
            p["actions"].append({"kind":"airmove", "duration":max(.18,math.dist(p["start"],start)/700),
                "to":start,"load":0.,"angle":first_angle,"tilt":.5,"form_event":"离纸移到下一结构落点"})
            curves.append(None)
        current = spec["start"]
        for i, (node, azimuth) in enumerate(zip(spec["nodes"], spec["angles"], strict=True)):
            x,y,load,speed,tilt,control,why=node
            kind="touch" if i==0 else "release" if i==len(spec["nodes"])-1 else "turn"
            length=math.dist(current,[x,y]) if control is None else math.dist(current,control)+math.dist(control,[x,y])
            p["actions"].append({"kind":kind,"to":[x*SCALE,y*SCALE],"duration":max(.04,length/speed),
                "load":load,"tilt":tilt,"angle":math.radians(azimuth),"form_event":why})
            curves.append(None if control is None else [v*SCALE for v in control])
            current=[x,y]
        previous=p["actions"][-1]
        programs.append({"program":p,"curves":curves,"form_reason":spec["why"]})
    return programs


def settings():
    return ContactSettings(width=30., bunches=36, initial_pigment=12000.,
        initial_water=.35, paper_relief=.21, transfer=.52, evaporation=.008)


def render(programs=None, cfg=None, material="joined"):
    canvas = InkCanvas(*SIZE, seed=505, paper_color=PAPER)
    canvas.density=canvas.density.astype(np.float64)
    cls={"joined":JoinedContactBrush,"tooth":ToothContactBrush}[material]
    brush=cls(canvas,cfg or settings(),seed=606)
    frames=[]
    for item in programs or make_programs():
        brush.draw_driver(compile_form(item["program"],item["curves"]))
        frames.append(canvas.to_image())
    return canvas,brush,frames


def save_run(output, canvas, brush, frames, programs, material="joined"):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    canvas.save(output/'result.png')
    data={"version":VERSION,"material":material,
        "canvas":{"size":list(SIZE),"paper":list(PAPER),"paper_seed":505,"brush_seed":606},
        "programs":programs,"settings":asdict(brush.settings),"records":brush.records,
        "diagnostics":brush.diagnostics(),"image_sha256":image_hash(canvas)}
    raw=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    (output/'ledger.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    (output/'form_actions.json').write_text(json.dumps(programs,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'diagnostics.json').write_text(json.dumps(data['diagnostics'],indent=2),encoding='utf-8')
    np.savez_compressed(output/'fields.npz',density=canvas.density,paper=canvas.paper)
    for i,frame in enumerate(frames,1):frame.save(output/f'step_{i:02d}.png')
    return data


def replay(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
    if data['version']!=VERSION or data['canvas']!={"size":list(SIZE),"paper":list(PAPER),"paper_seed":505,"brush_seed":606}:
        raise ValueError('unsupported study version or canvas')
    canvas,brush,frames=render(data['programs'],ContactSettings(**data['settings']),data['material'])
    if image_hash(canvas)!=data['image_sha256'] or brush.records!=data['records']:
        raise ValueError('image or complete state trace differs')
    return canvas,brush,frames


def make_plate(source, before, after, path):
    plate=Image.new('RGB',(1710,570),PAPER)
    label(plate,(24,15),'ZHAO MENGFU / REFERENCE - v0.7 - v0.8',27)
    for i,(im,title) in enumerate([(source,'Reference / The Met'),(before,'v0.7 / Separate bunch tracks'),(after,'v0.8 / Joined contact patches')]):
        label(plate,(24+i*570,66),title,19)
        plate.paste(im.resize((540,435),Image.Resampling.LANCZOS),(15+i*570,102))
    label(plate,(24,543),'Reference is an unretouched crop. Both studies use authored strokes; paper tone is not matched to the aged original.',16)
    plate.save(path)


def make_steps(frames, path):
    names=['Crown and right side','Left bearing wall','Inner crown ridge',
           'Fold below left shoulder','Main belly fold','Short fissure branch',
           'Right foot','Small belly ledge','Upper fissure']
    rows=math.ceil(len(frames)/3)
    plate=Image.new('RGB',(1200,75+rows*334),PAPER)
    label(plate,(24,19),'NINE CONTACTS / CONTINUOUS BRUSH STATE',26)
    for i,frame in enumerate(frames):
        x,y=11+(i%3)*400,75+(i//3)*334
        plate.paste(frame.resize((378,305),Image.Resampling.LANCZOS),(x,y))
        label(plate,(x+9,y+308),f'{i+1} / {names[i]}',16)
    plate.save(path)


def make_details(source, before, after, path):
    regions=[('Crown / side-brush turn',(100,23,217,105)),
             ('Left wall / connected ink',(9,110,89,209)),
             ('Belly / ledge and bearing turn',(130,113,214,225))]
    plate=Image.new('RGB',(1200,1110),PAPER)
    label(plate,(24,16),'SAME REGIONS / REFERENCE - v0.7 - v0.8',27)
    for row,(title,box) in enumerate(regions):
        y=65+row*347
        label(plate,(24,y),title,19)
        for col,im in enumerate([source,before,after]):
            native=box if col==0 else tuple(v*2 for v in box)
            tile=im.crop(native)
            scale=min(366/tile.width,304/tile.height)
            tile=tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.LANCZOS)
            plate.paste(tile,(col*400+(400-tile.width)//2,y+33+(304-tile.height)//2))
    plate.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='iteration-output')
    parser.add_argument('--replay')
    args=parser.parse_args()
    if args.replay:
        canvas,brush,_=replay(args.replay)
        print(json.dumps({'replay':'PASS',**brush.diagnostics()}))
        return
    programs=make_programs()
    canvas,brush,frames=render(programs)
    data=save_run(args.output,canvas,brush,frames,programs)
    source=Image.open('reference-copy/zhao_rock_crop.png').convert('RGB')
    before=Image.open('reference-final/reference_r2_tooth.png').convert('RGB')
    make_plate(source,before,canvas.to_image(),Path(args.output)/'comparison.png')
    make_steps(frames,Path(args.output)/'steps.png')
    make_details(source,before,canvas.to_image(),Path(args.output)/'details.png')
    print(json.dumps({'image_sha256':data['image_sha256'],**brush.diagnostics()},ensure_ascii=False))


if __name__=='__main__':main()
