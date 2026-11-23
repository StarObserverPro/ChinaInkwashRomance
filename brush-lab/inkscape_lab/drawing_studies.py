"""Three actual contacts: sparse, authored stone drawing studies.

Every route is authored once. No contour offsets, hatching, wash, point clouds,
face fills or stochastic route perturbations are added. Material texture is
the only seeded variation. Coordinates are study choices, not recovered
historical hand trajectories.
"""
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .brush import InkCanvas
from .studies import label

PAPER = (247, 244, 235)


def stone_spec(revision=0):
    """Each node is a real hand update and has a concrete form reason.

    Node tuple: x, y, load, speed, reason. The first is the touch destination;
    the last is release, with target load zero.
    """
    specs = [
        {'role': '主势', 'start': [231, 245], 'nodes': [
            (239, 234, .38, 115, '左肩入势'),
            (273, 205, .53, 155, '左肩斜升'),
            (318, 184, .34, 135, '峰前坡转缓'),
            (354, 190, .30, 150, '峰顶折向右肩'),
            (397, 222, .54, 145, '右肩由平转落'),
            (421, 266, .71, 115, '侧壁下落'),
            (415, 285, 0., 120, '侧缘收住，底不封口')],
         'why': '由左肩跨峰至右侧，一笔交代外势，不沿外缘补描。'},
        {'role': '侧足', 'start': [235, 251], 'nodes': [
            (228, 268, .69, 120, '承左肩，落入侧壁'),
            (225, 301, .78, 125, '侧壁承重'),
            (249, 326, .62, 140, '石足折向内'),
            (285, 333, .30, 170, '底缘由实转虚'),
            (306, 329, 0., 180, '向内提收，底部留白')],
         'why': '只补左侧承重与短底脚，不闭合底边。'},
        {'role': '转面', 'start': [264, 216], 'nodes': [
            (277, 232, .31, 155, '由肩内侧接入'),
            (313, 253, .39, 175, '斜向交代顶面前缘'),
            (348, 248, .34, 155, '层缘短折，面向改变'),
            (360, 281, .61, 125, '棱脊折下，顶面转为侧面'),
            (382, 311, 0., 155, '侧面中提收，停在底脚之前')],
         'why': '唯一内界，由顶缘折入侧面；白面本身保留，不增加伴随线。'},
    ]
    if revision >= 1:
        specs[0].update(start=[218,251],nodes=[
            (226,239,.28,135,'左肩轻入'),
            (257,224,.56,165,'肩坡短促上行'),
            (279,197,.30,135,'肩峰局部陡起'),
            (329,178,.12,170,'峰顶向右展开，提笔变轻'),
            (350,185,.20,170,'峰顶小折'),
            (397,224,.48,155,'右肩转向外侧'),
            (414,268,.78,120,'侧壁下行承重'),
            (405,285,0.,130,'侧缘提收，底口开放')])
        specs[1].update(start=[224,251],nodes=[
            (216,268,.50,120,'左肩下承'),
            (210,294,.85,120,'左侧承重'),
            (232,327,.48,150,'石足向内折'),
            (264,333,.16,180,'足部平出，渐虚'),
            (288,329,0.,180,'提收，不封底')])
        specs[2].update(start=[257,224],nodes=[
            (268,237,.18,155,'由左肩转入顶面前缘'),
            (307,252,.27,170,'顶面向前展开'),
            (343,246,.19,170,'凸脊处换向'),
            (340,270,.44,120,'短棱折下，转到前面'),
            (327,290,0.,165,'面内提收，不沿外侧伴行')])
    if revision >= 2:
        specs[0].update(start=[222,250],nodes=[
            (232,238,.23,130,'左肩轻入'),
            (289,195,.37,160,'长肩坡上行，建立主势'),
            (350,201,.13,170,'顶面展开，长行轻提'),
            (394,235,.49,150,'右肩折落'),
            (420,311,.76,135,'外侧承重落到右足'),
            (397,332,0.,145,'右足向内短收，底不闭合')])
        specs[2].update(start=[247,326],nodes=[
            (258,315,.20,140,'由左足内侧进入'),
            (293,288,.34,165,'前侧交界向石腹上行'),
            (326,280,.20,160,'石腹层缘换向'),
            (342,260,0.,155,'短折提收于面内，保留大块白面')],
            why='从左足向石腹引出唯一内界，在局部层缘改向后即收；不复制外缘。')
    return specs


def _program(spec, previous=None):
    start=list(spec['start'])
    first=spec['nodes'][0]
    angle=math.atan2(first[1]-start[1], first[0]-start[0])
    p={'start':start, 'initial':{'load':0.,'angle':angle,'tilt':.18},
       'max_dt':1/140, 'spatial_step':.45, 'tag':spec['role'], 'actions':[]}
    if previous is not None:
        p['start']=[previous['hand_x'],previous['hand_y']]
        p['initial']={'load':previous['pressure_target'],
                      'angle':previous['angle_target'],'tilt':previous['tilt_target']}
        p['actions'].append({'kind':'airmove','duration':max(.12,math.dist(p['start'],start)/750),
                             'to':start,'load':0.,'angle':angle,'tilt':.18,
                             'form_event':'离纸移到下一结构落点；不补墨'})
    current=start
    for i,(x,y,load,speed,reason) in enumerate(spec['nodes']):
        dest=[x,y]
        kind='touch' if i==0 else 'release' if i==len(spec['nodes'])-1 else 'turn'
        duration=max(.035,math.dist(current,dest)/speed)
        p['actions'].append({'kind':kind,'duration':duration,'to':dest,'load':load,
                             'angle':math.atan2(y-current[1],x-current[0]),
                             'tilt':.18 if spec['role']=='主势' else .32,
                             'form_event':reason})
        current=dest
    return p


def render_drawing(output, revision=0):
    from .drawing import draw_mark
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    canvas=InkCanvas(700,500,seed=505,paper_color=PAPER)
    brush={'width':12.,'tip_tau':.010,'pressure_tau':.023,'angle_tau':.021,'tilt_tau':.022}
    ink={'ink_capacity':28000.,'concentration':.82,'k_transfer':.045,'tau_ink':.025,
         'initial_water_load':.62,'dry':.12,'water_transfer':.012,'evaporation':.006}
    if revision>=1:
        brush['width']=14.
        ink.update(ink_capacity=23500.,initial_water_load=.80,dry=.02,water_transfer=.006,
                   evaporation=.004,k_transfer=.044)
    frames=[]; state=None
    for order,spec in enumerate(stone_spec(revision),1):
        program=_program(spec,state)
        record=draw_mark(canvas,program,brush=brush,ink=ink,paper={'spread':.24},seed=505,
                         initial_state=state,feedback={'enabled':False},
                         context={'role':spec['role'],'brush_id':'one_brush','stroke_number':order,
                                  'form_reason':spec['why'],'no_parallel_copy':True})
        state=record['final_state']; frames.append(canvas.to_image())
    image=f'drawing_r{revision}.png'; canvas.save(output/image)
    ledger={'width':700,'height':500,'recipe':{'seed':505},'paper_color':list(PAPER),
            'renderer_version':'0.5.0','image':image,'revision':revision,
            'metadata':{'actual_contacts':3,'full_ledger':'drawing_strokes.json',
                        'title':'三笔勾石：主势、侧足、转面'},'strokes':canvas.strokes}
    raw=json.dumps(ledger,ensure_ascii=False,separators=(',',':')).encode()
    (output/f'drawing_r{revision}_strokes.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    if revision==2:
        canvas.save(output/'drawing_final.png')
        ledger['image']='drawing_final.png'
        (output/'drawing_strokes.json').write_text(json.dumps(ledger,ensure_ascii=False,separators=(',',':')))
        steps=Image.new('RGB',(1500,450),PAPER)
        label(steps,(30,20),'THREE CONTACTS / ONE STONE',25)
        for i,frame in enumerate(frames):
            crop=frame.crop((150,130,490,370)).resize((442,312),Image.Resampling.LANCZOS)
            steps.paste(crop,(30+i*500,72))
            label(steps,(45+i*500,399),['1 / main contour','2 / left support','3 / change of plane'][i],20)
        steps.save(output/'drawing_steps.png')
        _turn_diagram(frames[-1],stone_spec(revision),output/'drawing_turns.png')
    return canvas


def _turn_diagram(image,specs,path):
    im=image.copy(); d=ImageDraw.Draw(im)
    colors=[(159,71,44),(60,113,117),(86,94,164)]
    label(im,(28,25),'DISCRETE ACTION NODES / ANALYSIS OVERLAY',21)
    for i,spec in enumerate(specs):
        for j,node in enumerate(spec['nodes']):
            x,y=node[:2]
            d.ellipse((x-3,y-3,x+3,y+3),fill=colors[i])
            label(im,(x+5,y-18),f'{i+1}.{j+1}',12,colors[i])
    label(im,(28,450),'Dots mark hand updates, not additional brush contacts.',17)
    im.save(path)


def make_drawing_studies(output,revision=2):
    c=render_drawing(output,revision)
    print(f'drawing_r{revision}.png: {len(c.strokes)} actual contacts')
