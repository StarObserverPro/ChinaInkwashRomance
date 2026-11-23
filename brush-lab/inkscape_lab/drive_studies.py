"""Counterfactual experiments for semantic actions and independent followers.

These are engineering hypotheses for stroke construction, not recovered
motions of historical painters. Rendered marks are generated only by code.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .brush import InkCanvas
from .gesture import compile_gesture, gesture_stroke
from .studies import label

PAPER = (244, 240, 228)
TARGET = (168, 90, 68)
ACTUAL = (48, 82, 82)
INK = dict(k_transfer=.035, tau_ink=.065, concentration=.68, ink_capacity=23040.,
           initial_water_load=.55, dry=.25)
BRUSH = dict(width=24., pressure_tau=.065, tip_tau=.045,
             angle_tau=.09, tilt_tau=.065)


def action_program(scale=1., offset=(0., 0.)):
    """One contact, seven editable actions; a fold is not a new stroke."""
    p = {'start': [42., 100.], 'initial': {'angle': -.25, 'tilt': .45},
         'max_dt': 1 / 120, 'spatial_step': .6, 'tag': 'action/fold',
         'events': [
             {'kind': 'touch', 'duration': .13, 'load': .55},
             {'kind': 'drag', 'duration': .72, 'to': [202., 69.], 'load': .42},
             {'kind': 'dwell', 'duration': .16, 'load': .86},
             {'kind': 'turn', 'duration': .26, 'to': [219., 91.], 'angle': 1.03, 'load': .64},
             {'kind': 'drag', 'duration': .55, 'to': [304., 175.], 'load': .44},
             {'kind': 'drag', 'duration': .43, 'to': [385., 151.], 'angle': -.27, 'load': .28},
             {'kind': 'lift', 'duration': .21, 'to': [412., 145.]}]}
    def move(q):
        return [float(q[0] * scale + offset[0]), float(q[1] * scale + offset[1])]
    p['start'] = move(p['start'])
    for event in p['events']:
        if 'to' in event:
            event['to'] = move(event['to'])
    return p


def _plot(im, box, rows, fields, colors, ymax=None, title=''):
    x, y, w, h = box
    d = ImageDraw.Draw(im)
    tmax = max(r['t'] for r in rows)
    ymax = ymax or max(max(float(r.get(k, 0.)) for r in rows) for k in fields) * 1.05
    ymax = max(.01, ymax)
    d.line((x, y, x, y+h, x+w, y+h), fill=(170, 170, 159), width=1)
    for k, color in zip(fields, colors):
        pts = [(x+w*r['t']/tmax, y+h*(1-min(ymax, max(0.,float(r.get(k,0.))))/ymax)) for r in rows]
        d.line(pts, fill=color, width=2)
    label(im, (x, y-25), title, 15)
    label(im, (x, y+h+5), '0 s', 12)
    label(im, (x+w-43, y+h+5), f'{tmax:.2f} s', 12)


def _driver_path(im, driver, shift=(0, 0)):
    pts = [(r['hand_x']+shift[0], r['hand_y']+shift[1]) for r in driver['samples']]
    d = ImageDraw.Draw(im)
    for i in range(0, len(pts)-1, 14):
        d.line(pts[i:min(i+7, len(pts))], fill=(175, 117, 95), width=1)


def mechanics(out):
    p = action_program()
    driver = compile_gesture(p)
    im = Image.new('RGB', (1440, 850), PAPER)
    label(im, (40, 25), '08 / DISCRETE ACTIONS, CONTINUOUS FOLLOWERS', 30)
    label(im, (40, 70), 'One command history. Only mechanical relaxation times change; ink, paper and seeds stay fixed.', 17)
    label(im, (40, 112), 'TOUCH   >   DRAG   >   DWELL   >   TURN (settle / rotate / depart)   >   DRAG   >   DRAG   >   LIFT', 16)
    timings = [(.008, .012, .015), (.045, .065, .09), (.14, .24, .28)]
    records = []
    for i, (tip, force, angle) in enumerate(timings):
        brush = dict(BRUSH, tip_tau=tip, pressure_tau=force, angle_tau=angle, tilt_tau=force)
        c = InkCanvas(445, 260, seed=919)
        r = gesture_stroke(c, p, brush=brush, ink=INK, paper={'spread': .25}, seed=719)
        records.append(r)
        x = 38+i*470
        label(im, (x, 162), ['A / Fast response', 'B / Medium response', 'C / Slow response'][i], 21)
        label(im, (x, 197), f'tip {tip:.3f}s  load {force:.3f}s  angle {angle:.3f}s', 15)
        pic = c.to_image()
        _driver_path(pic, driver)
        im.paste(pic, (x, 233))
        _plot(im, (x+18, 543, 396, 130), r['trace'], ['pressure_target', 'pressure'], [TARGET, ACTUAL], 1.05,
              'Load target (red) / contact load (green)')
        lag = max(math.hypot(t['hand_x']-t['tip_x'], t['hand_y']-t['tip_y']) for t in r['trace'])
        label(im, (x+18, 715), f'Max hand-tip separation: {lag:.1f} px', 16)
    label(im, (40, 770), 'Dashed red = hand command; ink = following tip. No state resets at the dwell or fold.', 17)
    label(im, (40, 801), 'These time constants are uncalibrated model parameters. More lag is not automatically better brushwork.', 15)
    im.save(out/'drive_01_mechanics.png')
    return records


def ink_paper(out):
    p = action_program()
    im = Image.new('RGB', (1440, 920), PAPER)
    label(im, (40, 25), '09 / SAME MECHANICS: INK SUPPLY AND PAPER SPREAD', 29)
    label(im, (40, 70), 'Rows change supply relaxation; columns change the normalized paper kernel. No optical gain adjustment.', 16)
    records = []
    for row, tau in enumerate([.008, .36]):
        for col, spread in enumerate([0., 1.4, 3.0]):
            x, y = 38+col*470, 129+row*355
            c = InkCanvas(445, 245, seed=919)
            r = gesture_stroke(c, p, brush=BRUSH, ink=dict(INK, tau_ink=tau), paper={'spread':spread}, seed=719)
            records.append(r)
            label(im, (x,y), f'Supply tau {tau:.3f}s / spread {spread:.1f}', 19)
            im.paste(c.to_image(), (x,y+40))
            b = r['diagnostics']['pigment_budget']
            label(im, (x+5,y+289), f'Transfer {b["transferred"]:.1f}  /  remaining {b["final"]:.1f}', 15)
    label(im, (40, 851), 'All six retain exactly the same tip, load and orientation traces. Paper changes neither supply nor brush inventory.', 16)
    label(im, (40, 880), 'Paper here is a local spread proxy, not a time-evolving wet-paper solver.', 15)
    im.save(out/'drive_02_layers.png')
    return records


def cun_program(style, x, y, length, width, revision, rng):
    """Form-conditioned action grammar; each call creates an explicit contact."""
    duration = length / (130 if style != 'axe' else 155)
    jitter = float(rng.uniform(.85, 1.15)) if revision else 1.
    p = {'start':[x,y], 'initial':{'angle':math.pi/2, 'tilt':.25},
         'max_dt':1/120, 'spatial_step':.6, 'tag':style+'/grammar', 'events':[]}
    events = p['events']
    events.append({'kind':'touch','duration':.11 if revision == 0 else .055,'load':.58*jitter})
    if style == 'pima':
        for j, (dx, frac, load) in enumerate([(4,.28,.60),(-5,.63,.37),(3,.89,.66)]):
            events.append({'kind':'drag','duration':duration*(.28 if j==0 else .33),
                           'to':[x+dx,y+length*frac], 'load':load*jitter,'angle':math.pi/2+(j-1)*.10})
        events.append({'kind':'lift','duration':duration*.22,'to':[x+6,y+length]})
    elif style == 'zhedai':
        p['initial']['angle'] = .08
        ledge = length * (.46 if revision == 0 else float(rng.uniform(.30,.44)))
        events.extend([
            {'kind':'drag','duration':duration*.37,'to':[x+ledge,y+3], 'load':.50*jitter},
            {'kind':'dwell','duration':.17 if revision == 0 else .035,'load':.93 if revision == 0 else .62*jitter},
            {'kind':'turn','duration':.24 if revision == 0 else .12,
             'to':[x+ledge+2,y+12], 'load':.66*jitter,'angle':1.35},
            {'kind':'drag','duration':duration*.57,'to':[x+ledge+9,y+length*.65],'load':.45*jitter},
            {'kind':'lift','duration':duration*.19,'to':[x+ledge+18,y+length*.70]}])
    else:
        p['initial'].update(angle=2.15,tilt=1.05)
        events.extend([
            {'kind':'press','duration':.12 if revision == 0 else .045,'load':1.0 if revision == 0 else .78},
            {'kind':'drag','duration':duration*.45,'to':[x-length*.15,y+length*.22], 'load':.63},
            {'kind':'drag','duration':duration*.20,'to':[x-length*.38,y+length*.48], 'load':.28},
            {'kind':'lift','duration':duration*.15,'to':[x-length*.50,y+length*.64]}])
    return p


def cun(out, revision=0):
    im = Image.new('RGB', (1500, 880), PAPER)
    label(im, (40,25), f'10 / CUN AS AN ACTION GRAMMAR - STUDY {revision}', 30)
    label(im, (40,70), 'Stroke placement follows a shared rock face; each mark expands into timed actions. Seeds are fixed.', 17)
    rows = []
    for col, style in enumerate(['pima','zhedai','axe']):
        c = InkCanvas(445, 600, seed=627)
        rng = np.random.default_rng(713)
        shape = [(32,518),(69,380),(103,313),(144,167),(218,92),(258,190),(295,281),(365,399),(412,525)]
        c.wash(shape, ink=.08,water=.7,tag=style+'/form',fade_bottom=90)
        for i in range(1, len(shape)-1):
            if i%3:
                c.stroke(shape[i-1:i+1],width=1.3,ink=.30,water=.2,dry=.6,tag=style+'/contour')
        combined = None
        for k in range(24):
            group, lane = k//8, k%8
            if revision == 0:
                x=126+group*77+lane*3.1
                y=[254,175,331][group]+lane*9
                length=88.
            else:
                # Grouping changes with face position; variations share a seed.
                x=[111,193,267][group]+lane*4.5+float(rng.normal(0,1.8))
                y=[284,145,354][group]+lane*float(rng.uniform(6,13))
                length=float(rng.uniform(58,108))
            width={'pima':4.5,'zhedai':5.7,'axe':17.}[style]
            program=cun_program(style,x,y,length,width,revision,rng)
            # For the axe group place the contact on the right of each wedge.
            if style == 'axe':
                program=deepcopy(program)
                program['start'][0]+=16
                for event in program['events']:
                    if 'to' in event: event['to'][0]+=16
            if combined is None:
                combined=deepcopy(program)
                combined['tag']=style+'/24-contacts-2-reloads'
            else:
                if k%8==0:
                    combined['events'].append({'kind':'reload','duration':.3,'ink_load':1.,'water_load':.22})
                combined['events'].append({'kind':'airmove','duration':.16,'to':program['start'],
                                           'angle':program['initial']['angle'],'tilt':program['initial']['tilt']})
                combined['events'].extend(program['events'])
        brush=dict(BRUSH,width=width,tip_tau=.018,pressure_tau=.026,angle_tau=.055)
        ink=dict(INK,ink_capacity=65*width**2, k_transfer=.18 if style!='axe' else .06,
                 dry=.45,initial_water_load=.22)
        r=gesture_stroke(c,combined,brush=brush,ink=ink,paper={'spread':.18},seed=450)
        rows.append(r)
        im.paste(c.to_image(),(40+col*493,133))
        label(im,(45+col*493,755),['Pima / successive load targets','Zhedai / dwell, rotate, depart','Axe / press, drag, unload'][col],20)
        label(im,(45+col*493,790),['No global sine wave is required','The same brush crosses every action','A wedge follows from a timed release'][col],15)
    label(im,(40,837),'Each panel: one brush history, 24 contacts, 2 explicit reloads. These are engineering recipes, not historical reconstructions.',15)
    im.save(out/f'drive_03_cun_r{revision}.png')
    return rows


def ledger(out):
    c=InkCanvas(1200,760,seed=901)
    for i,tau in enumerate([.01,.06,.16]):
        p=action_program(scale=2.15,offset=(30,-135+i*210))
        # Keep the waveform identical up to a known canvas translation.
        p['tag']=['fast/one-contact','medium/one-contact','slow/one-contact'][i]
        gesture_stroke(c,p,brush=dict(BRUSH,width=30,tip_tau=tau,pressure_tau=tau*1.5,angle_tau=tau*1.8),
                       ink=dict(INK,k_transfer=.065,ink_capacity=40*30**2),paper={'spread':.3},seed=400)
    c.save(out/'drive_04_ledger.png')
    data={'width':c.width,'height':c.height,'recipe':{'seed':901},'paper_color':list(c.paper_color),
          'renderer_version':'0.3.0','image':'drive_04_ledger.png','strokes':c.strokes}
    (out/'drive_strokes.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    return data


def make_drive_studies(out,revision=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if revision==0:
        records=mechanics(out)+ink_paper(out)
        (out/'drive_experiments.json').write_text(json.dumps({'strokes':records},ensure_ascii=False,separators=(',',':')),encoding='utf8')
        ledger(out)
    records=cun(out,revision)
    # Keep editable grammar programs and resolved follower parameters without
    # duplicating every numeric trace already illustrated in the large ledger.
    recipes=[{k:r[k] for k in ('program','brush','ink','paper','seed','driver_hash','diagnostics')} for r in records]
    (out/f'drive_cun_r{revision}_recipes.json').write_text(json.dumps(recipes,ensure_ascii=False,indent=2),encoding='utf8')
    print('Action / response studies written to',out)
