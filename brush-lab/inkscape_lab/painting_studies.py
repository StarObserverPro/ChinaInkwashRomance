"""Painting a shared stone: every mark has a role, a place and a history.

The geometric scaffold is explicit and hand-designed. Pressure feedback reads
the evolving optical-density canvas. Neither is an inferred painter's mind.
"""
from copy import deepcopy
import json
import gzip
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .brush import InkCanvas
from .studies import label

PAPER = (244, 240, 228)


def _unit(v):
    v=np.asarray(v,dtype=float)
    return v/max(float(np.linalg.norm(v)),1e-9)


def _angle(a,b):
    return math.atan2(b[1]-a[1],b[0]-a[0])


def stroke_program(points, role, *, revision=1, load=.6, speed=130.,
                   entry='moving', ending='lift', tilt=.3):
    """A local recipe: contact and movement share the same spatial path.

    A turn is only scheduled where the path actually changes direction. The
    turn phase unloads slightly; no fixed dark stamp is added at corners.
    """
    pts=[np.asarray(p,dtype=float) for p in points]
    pts=[p for i,p in enumerate(pts) if i==0 or np.linalg.norm(p-pts[i-1])>1e-6]
    if len(pts)<2:
        raise ValueError('a painted mark needs two distinct spatial points')
    start=pts[0].copy();first_direction=_unit(pts[1]-pts[0]);last_direction=_unit(pts[-1]-pts[-2])
    entry_angle=_angle(pts[0],pts[1])
    program={'start':start.tolist(),'initial':{'load':0.,'angle':entry_angle,'tilt':tilt},
             'max_dt':1/100,'spatial_step':.8,'tag':role,'events':[]}
    events=program['events']
    if revision==0:
        events.append({'kind':'touch','duration':.10,'load':load})
        current=start
        for p in pts[1:]:
            events.append({'kind':'drag','duration':max(.025,float(np.linalg.norm(p-current))/speed),
                           'to':p.tolist(),'load':load,'angle':_angle(current,p),'tilt':tilt})
            current=p
        events.append({'kind':'lift','duration':.12})
        return program
    if entry=='reverse':
        program['start']=(start+first_direction*4).tolist()
        events.append({'kind':'touch','duration':.055,'to':start.tolist(),'load':load*.5})
        current=start
    elif entry=='pressed':
        events.append({'kind':'touch','duration':.035,'load':load*.86})
        current=start
    else:
        current=start+first_direction*min(5.,float(np.linalg.norm(pts[1]-start))*.6)
        events.append({'kind':'touch','duration':.055,'to':current.tolist(),'load':load*.58})
    path=pts[1:]
    if ending=='lift':
        # Move while releasing, but do not discard an entire long last segment.
        tail=min(8.,float(np.linalg.norm(pts[-1]-pts[-2]))*.55)
        path=pts[1:-1]+[pts[-1]-last_direction*tail]
    traveled=0.
    total=sum(float(np.linalg.norm(b-a)) for a,b in zip(pts[:-1],pts[1:]))
    previous_direction=first_direction
    for j,p in enumerate(path):
        delta=p-current;distance=float(np.linalg.norm(delta))
        if distance<1e-5:
            continue
        direction=_unit(delta)
        turn=math.acos(float(np.clip(np.dot(previous_direction,direction),-1,1)))
        target=load*(.92-.24*min(1.,traveled/max(total,1.)))
        if turn>.45 and role not in ('tone','moss'):
            departure=current+direction*min(4.,distance*.3)
            events.append({'kind':'turn','duration':.055+.035*min(turn,1.4),
                           'to':departure.tolist(),'load':target*.59,
                           'angle':_angle(current,p),'tilt':tilt+.12})
            current=departure;distance=float(np.linalg.norm(p-current))
        events.append({'kind':'drag','duration':max(.025,distance/speed),
                       'to':p.tolist(),'load':target,'angle':_angle(current,p),'tilt':tilt})
        traveled+=distance;current=p;previous_direction=direction
    if ending=='return':
        events.append({'kind':'lift','duration':.105,'to':(pts[-1]-last_direction*5).tolist()})
    elif ending=='blunt':
        events.append({'kind':'lift','duration':.065})
    else:
        events.append({'kind':'lift','duration':max(.06,float(np.linalg.norm(pts[-1]-current))/(speed*1.15)),
                       'to':pts[-1].tolist()})
    return program


class RolePainter:
    """A few named brush histories; changing a picture role does not dip ink."""
    def __init__(self, canvas, revision=1, feedback=True):
        self.canvas,self.revision,self.feedback=canvas,revision,feedback
        self.states={}
        self.materials={
            'line':({'width':5.2,'tip_tau':.012,'pressure_tau':.023,'angle_tau':.035,'tilt_tau':.025},
                    {'ink_capacity':8500.,'concentration':.78,'k_transfer':.065,'tau_ink':.025,
                     'initial_water_load':.36,'dry':.45}),
            'texture':({'width':4.2,'tip_tau':.016,'pressure_tau':.026,'angle_tau':.04,'tilt_tau':.03},
                       {'ink_capacity':6000.,'concentration':.62,'k_transfer':.062,'tau_ink':.04,
                        'initial_water_load':.30,'dry':.62}),
            'tone':({'width':29.,'tip_tau':.024,'pressure_tau':.045,'angle_tau':.06,'tilt_tau':.035},
                    {'ink_capacity':42000.,'concentration':.20,'k_transfer':.095,'tau_ink':.08,
                     'initial_water_load':.85,'dry':.10}),
            'moss':({'width':11.,'tip_tau':.015,'pressure_tau':.02,'angle_tau':.03,'tilt_tau':.025},
                    {'ink_capacity':8500.,'concentration':.80,'k_transfer':.20,'tau_ink':.025,
                     'initial_water_load':.28,'dry':.50}),
        }
        if revision>=2:
            self.materials['texture'][0]['width']=6.5
            self.materials['texture'][1].update(ink_capacity=7200.,concentration=.48,k_transfer=.052,dry=.48)
            self.materials['tone'][0]['width']=51.
            self.materials['tone'][1].update(ink_capacity=60000.,concentration=.075,k_transfer=.12)

    def draw(self, points, *, role, face, load=.6, speed=130., entry='moving',
             ending='lift', tilt=.3, explanation=None, brush_name=None, ceiling=.8):
        from .painting import paint_mark
        brush_name=brush_name or ('line' if role in ('contour','ridge','crevice','foot') else role)
        brush,ink=deepcopy(self.materials[brush_name])
        p=stroke_program(points,role,revision=self.revision,load=load,speed=speed,entry=entry,ending=ending,tilt=tilt)
        state=self.states.get(brush_name)
        if state is not None:
            target=p['start'];angle=p['initial']['angle'];pose=p['initial']['tilt']
            p['start']=[state['hand_x'],state['hand_y']]
            p['initial']={'load':state['pressure_target'],'angle':state['angle_target'],'tilt':state['tilt_target']}
            prefix=[]
            if state['ink_load']<.28 or state['water_load']<.12:
                prefix.append({'kind':'reload','duration':.24,'ink_load':1.,'water_load':ink['initial_water_load']})
            distance=math.dist(p['start'],target)
            prefix.append({'kind':'airmove','duration':max(.12,distance/1200.),'to':target,
                           'load':0.,'angle':angle,'tilt':pose,'ramp':'smooth'})
            p['events']=prefix+p['events']
        context={'role':role,'face':face,'brush_id':brush_name,
                 'entry_recipe':entry,'ending_recipe':ending,**(explanation or {})}
        p['tag']=role+'/'+face
        rec=paint_mark(self.canvas,p,brush=brush,ink=ink,paper={'spread':.33 if role!='tone' else 1.5},
                       seed={'line':804,'texture':805,'tone':806,'moss':807}[brush_name],
                       initial_state=state,feedback={'enabled':bool(self.feedback and self.revision>0),
                                                     'density_ceiling':ceiling,'strength':1.35,
                                                     'min_multiplier':.16,'lookahead':6.,'multiplier_tau':.025},
                       context=context)
        self.states[brush_name]=rec['final_state']
        return rec


def _write_ledger(canvas,out,name,image,extra=None):
    data={'width':canvas.width,'height':canvas.height,'recipe':{'seed':canvas.seed},
          'paper_color':list(canvas.paper_color),'renderer_version':'0.4.0','image':image,
          'strokes':canvas.strokes,**(extra or {})}
    stored=data
    if name.endswith('.gz') and data.get('revision',2)<2:
        stored=replay_input_ledger(data)
    encoded=json.dumps(stored,ensure_ascii=False,separators=(',',':')).encode('utf8')
    if name.endswith('.gz'):
        (out/name).write_bytes(gzip.compress(encoded,compresslevel=5,mtime=0))
    else:
        (out/name).write_bytes(encoded)
    return data


def replay_input_ledger(data):
    """Keep all reconstruction inputs, omit large regenerable trace arrays."""
    stored={k:v for k,v in data.items() if k!='strokes'}
    stored.update(derived_trace_omitted=True,strokes=[])
    for r in data['strokes']:
        row={k:v for k,v in r.items() if k not in ('trace','driver','points')}
        row['points']=[r['program']['start']]+[e['to'] for e in r['program']['events'] if 'to' in e]
        stored['strokes'].append(row)
    return stored


def _at(points,u):
    p=np.asarray(points,dtype=float)
    a=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    return [float(np.interp(u*a[-1],a,p[:,j])) for j in (0,1)]


def _piece(points,a,b):
    pts=np.asarray(points,dtype=float)
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    u=lengths/lengths[-1]
    return [_at(points,a)]+[p.tolist() for p,v in zip(pts,u) if a<v<b]+[_at(points,b)]


def form_plate(out,form):
    im=Image.new('RGB',(1000,940),PAPER);d=ImageDraw.Draw(im)
    label(im,(42,26),'11 / WHERE A MARK BELONGS',30)
    label(im,(42,72),'Shared planes, turning ridges, fissures and reserved paper. This is a diagram, not the painting.',16)
    shift=np.array([50.,35.])
    colors={'top':(226,225,207),'front':(210,218,209),'side':(187,201,196)}
    for face_id,face in form.faces.items():
        poly=[tuple(np.array(p)+shift) for p in face['polygon']]
        d.polygon(poly,fill=colors[face_id])
        for p in form.sample_points(face_id,15,seed=185,margin=20,min_spacing=26):
            v=np.array(form.tangent(face_id,p))*20;start=np.array(p)+shift;end=start+v
            d.line([tuple(start),tuple(end)],fill=(93,118,108),width=2)
            side=np.array([-v[1],v[0]])*.15
            d.polygon([tuple(end),tuple(end-v*.35+side),tuple(end-v*.35-side)],fill=(93,118,108))
    for e in form.edges.values():
        color=(137,67,47) if e['kind']=='ridge' else (80,83,69)
        d.line([tuple(np.array(p)+shift) for p in e['points']],fill=color,width=3 if e['kind']=='ridge' else 2)
    for e in form.exclusions:
        d.polygon([tuple(np.array(p)+shift) for p in e['polygon']],fill=PAPER)
    label(im,(450,330),'LIGHT / reserve',20)
    label(im,(243,624),'FRONT',22)
    label(im,(655,606),'SIDE',22)
    label(im,(45,806),'Every stroke is assigned a role: contour / plane turn / texture / tone / foot / moss.',17)
    label(im,(45,841),'The scaffold constrains the act of drawing; it is never pasted into the final image.',16)
    label(im,(45,876),'Existing ink affects load during deposition. Face geometry determines where a mark can run.',16)
    im.save(out/'painting_00_form.png')
    (out/'painting_form.json').write_text(json.dumps(form.to_dict(),ensure_ascii=False,indent=2),encoding='utf8')


def render_stone(out,revision=0,feedback=True):
    from .form import StoneForm
    form=StoneForm()
    canvas=InkCanvas(900,850,seed=924)
    painter=RolePainter(canvas,revision,feedback)
    rng=np.random.default_rng(414)
    stages=[]
    # Establish volume through mark relationships before applying any tone.
    for name,edge in form.edges.items():
        role={'fissure':'crevice'}.get(edge['kind'],edge['kind'])
        faces='+'.join(edge['faces'])
        if revision==0:
            spans=[(0.,1.)]
        elif name=='crown':
            spans=[(.015,.22),(.255,.50),(.56,.73),(.76,.97)]
        elif name=='left_foot':
            spans=[(.0,.25),(.42,.68)]
        elif name in ('left_ridge','right_ridge','spine'):
            spans=[(.0,.40),(.46,.84)]
        elif name in ('front_shelf','top_fissure'):
            spans=[(.05,.55),(.73,.89)]
        else:
            spans=[(.0,.95)]
        if revision>=2:
            if name=='crown': spans=[(.01,.235),(.248,.52),(.535,.75),(.764,.99)]
            elif name=='left_foot': spans=[(.0,.34),(.40,.78)]
            elif name in ('left_ridge','right_ridge'): spans=[(.0,.45),(.475,.96)]
            elif name=='spine': spans=[(.0,.61),(.64,.97)]
        for a,b in spans:
            points=_piece(edge['points'],a,b)
            painter.draw(points,role=role,face=faces,
                         load=.70 if revision==0 else (.26+.45*edge['emphasis'] if revision>=2 else .34+.45*edge['emphasis']),
                         speed=140 if role!='foot' else 105,
                         entry='reverse' if role=='ridge' else 'moving',
                         ending='return' if role=='foot' else 'lift',tilt=.35,
                         ceiling=1.0 if role in ('ridge','foot') else .7,
                         explanation={'start_reason':'沿已有石面关系进入：'+edge['role'],
                                      'turn_reason':'轨迹在结构折点转向；先略提再转锋，局部保持材料历史',
                                      'end_reason':'在光面/虚边提收；底脚局部回收，保留遮挡重量',
                                      'edge_id':name})
    stages.append(('Structure / no wash',canvas.to_image()))
    # Texture begins near actual ridges and fissures, continuing their slopes.
    anchor_sets=[('front','left_ridge',19),('side','right_ridge',22),
                 ('front','front_fissure',12),('side','side_fissure',16),
                 ('front','front_shelf',8)]
    if revision>=2:
        anchor_sets=[('front','left_ridge',9),('side','right_ridge',11),
                     ('front','front_fissure',11),('side','side_fissure',14),('front','front_shelf',6)]
    for face,edge_id,count in anchor_sets:
        edge=form.edges[edge_id]
        for k in range(count):
            u=(k+.35)/count
            if revision>0: u=float(np.clip(u+rng.normal(0,.016),.025,.975))
            if revision>=2 and edge['kind']=='ridge':
                centers=[.13,.43,.81] if face=='front' else [.19,.52,.85]
                u=float(np.clip(centers[k%3]+rng.normal(0,.048),.025,.975))
            base=np.array(_at(edge['points'],u))
            direction=np.array(form.tangent(face,base))
            transverse=np.array([-direction[1],direction[0]])
            offset=3.5 if revision==0 else float(rng.uniform(3.,10.))
            start=base+direction*offset
            if edge['kind']=='fissure':
                start+=transverse*float(rng.uniform(-18,18))
                if revision>=2: start+=direction*float(rng.uniform(0,24))
            length=96. if revision==0 else float(rng.uniform(34.,116.))*(.8+.5*form.darkness(start,face))
            if revision>=2: length*=float(rng.uniform(.45,.90))
            points=form.trace(face,start,length,step=7.,margin=3.)
            if len(points)<4: continue
            density=form.darkness(start,face)
            painter.draw(points,role='texture',face=face,load=.66 if revision==0 else .31+.43*density,
                         speed=118.+float(rng.uniform(-12,15)),tilt=.30 if face=='front' else .65,
                         ceiling=.58 if face=='front' else .78,
                         explanation={'start_reason':'从'+edge_id+'邻近处生出，与骨架接续',
                                      'turn_reason':'行进方向随该石面的局部坡向与裂隙改变',
                                      'end_reason':'到面界、留白前或该笔长度终点提收；不裁切已落墨',
                                      'edge_id':edge_id})
    if revision>=2:
        # Short internal marks are staggered across the face, rather than all
        # hanging from its upper ridge. Existing density controls their load.
        for face,n in [('front',17),('side',25)]:
            for k,start in enumerate(form.sample_points(face,n,seed=713,margin=15,min_spacing=24)):
                density=form.darkness(start,face)
                pts=form.trace(face,start,float(rng.uniform(18,62)),step=6.,margin=7.)
                if len(pts)<4: continue
                painter.draw(pts,role='texture',face=face,load=.26+.34*density,
                             speed=120.+float(rng.uniform(-15,25)),tilt=.55 if face=='front' else .9,
                             entry='pressed' if k%4==0 else 'moving',ceiling=.6 if face=='front' else .8,
                             explanation={'start_reason':'面内短皴与接脊长笔错层分布，避免所有笔排成垂帘',
                                          'turn_reason':'顺所属面局部方向，已有浓墨处减载',
                                          'end_reason':'长度分层，留出面内空隙；不把每笔延至底边'})
    # A few upper bedding traces cross the broad plane rather than copy the
    # downward pattern of the shaded faces.
    for k,p in enumerate(form.sample_points('top',9,seed=190,margin=17,min_spacing=43)):
        pts=form.trace('top',p,65+float(rng.uniform(0,65)),step=7.,margin=9.)
        if len(pts)>=4:
            painter.draw(pts,role='texture',face='top',load=.36,speed=155.,tilt=.18,ceiling=.30,
                         explanation={'start_reason':'上面层理稀疏处轻入，避开保留纸白',
                                      'turn_reason':'顺上面延伸，方向区别于前面和侧面',
                                      'end_reason':'到留白边界前渐提，亮面不以等密度填满'})
    stages.append(('Structure + texture / no wash',canvas.to_image()))
    # Tone is also painted by broad wet contacts. No polygon face fill or
    # post-render eraser is used in this study.
    for face,count in ([('top',3),('front',22),('side',34)] if revision>=2 else [('top',4),('front',13),('side',21)]):
        for p in form.sample_points(face,count,seed=382,margin=20,min_spacing=23):
            pts=form.trace(face,p,float(rng.uniform(65,135) if revision>=2 else rng.uniform(50,105)),step=10.,margin=17.)
            if len(pts)<4: continue
            painter.draw(pts,role='tone',face=face,load={'top':.20,'front':.35,'side':.54}[face],
                         speed=105,tilt=.90,ceiling={'top':.15,'front':.34,'side':.55}[face],
                         explanation={'start_reason':'在该面的空疏处以湿笔轻入',
                                      'turn_reason':'宽笔仍顺面运行；读取既有墨色并限制继续加深',
                                      'end_reason':'在面内卸载，不用面多边形一次填色'})
    # Sparse moss accents are tied to seams, not distributed over the paper.
    for edge_id,count in [('left_ridge',5),('right_ridge',7),('right_foot',7)]:
        edge=form.edges[edge_id]
        for k in range(count):
            p=np.array(_at(edge['points'],float(rng.uniform(.08,.94))))
            p+=rng.normal(0,3,2)
            delta=np.array([float(rng.uniform(1,4)),float(rng.uniform(2,5))])
            painter.draw([p.tolist(),(p+delta).tolist()],role='moss',face='+'.join(edge['faces']),
                         load=.58,speed=70,entry='pressed',ending='blunt',tilt=.7,ceiling=1.8,
                         explanation={'start_reason':'只在脊或底脚作少量浓墨标记',
                                      'turn_reason':'短笔按入即收，保持不等距与不等大小',
                                      'end_reason':'原处提离形成厚实端部，不强制每笔都尖细'})
    name=f'painting_01_r{revision}.png'
    canvas.save(out/name)
    stages.append(('Final / all tone painted',canvas.to_image()))
    full_name=f'painting_r{revision}_strokes.json.gz'
    data=_write_ledger(canvas,out,full_name,name,
                  {'title':f'画的过程 · 三面石第{revision}轮','form':'painting_form.json','revision':revision})
    if revision>0:
        # The viewer is sampled, while the full trace is retained losslessly.
        compact=compact_display_ledger(data,full_name)
        (out/'painting_strokes.json').write_text(json.dumps(compact,ensure_ascii=False,separators=(',',':')),encoding='utf8')
        plate=Image.new('RGB',(1500,700),PAPER)
        label(plate,(36,24),'12 / ONE PAINTING HISTORY: STRUCTURE, TEXTURE, TONE',28)
        label(plate,(36,66),'Three actual checkpoints of one painting history. Broad tones are brush contacts, never face fills.',16)
        for i,(title,pic) in enumerate(stages):
            # All columns use exactly the same crop and scale.
            crop=pic.crop((80,215,820,725)).resize((470,324),Image.Resampling.LANCZOS)
            plate.paste(crop,(30+i*500,159))
            label(plate,(35+i*500,527),title,18)
        label(plate,(36,603),'Contour, turning ridge and cun share spatial relations. White paper participates in the plane change.',17)
        label(plate,(36,642),'This authored stone is a construction study; material constants and style still require comparison with real brushwork.',15)
        plate.save(out/'painting_02_process.png')
    return canvas,stages


def compact_display_ledger(data,full_name):
    """Keep replay inputs intact, decimate only derived viewer arrays."""
    result={k:v for k,v in data.items() if k!='strokes'}
    result.update(full_ledger=full_name,display_trace_decimated=True,strokes=[])
    for source in data['strokes']:
        row={k:v for k,v in source.items() if k not in ('trace','driver','points')}
        trace=source['trace']
        keep={0,len(trace)-1}
        for i,s in enumerate(trace):
            if s.get('event_start') or s.get('event_end'): keep.add(i)
        keep.update(int(i) for i in np.linspace(0,len(trace)-1,min(96,len(trace))))
        row['trace']=[{k:v for k,v in trace[i].items() if k=='event_start' or (not k.endswith('_mid') and not k.endswith('_start'))}
                      for i in sorted(keep)]
        row['driver']={k:v for k,v in source['driver'].items() if k!='samples'}
        row['points']=[[s['tip_x'],s['tip_y']] for s in row['trace']]
        row['display_trace_decimated']=True
        result['strokes'].append(row)
    return result


def make_painting_studies(out,revision=0):
    from .form import StoneForm
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if revision==0: form_plate(out,StoneForm())
    canvas,_=render_stone(out,revision)
    if revision>=2:
        supplementary_studies(out)
        comparison_plate(out)
    print(f'Painting study {revision}: {len(canvas.strokes)} mark histories written to {out}')


def _curve(im,box,trace,keys,colors,maximum=1.,title=''):
    x,y,w,h=box;d=ImageDraw.Draw(im)
    first,last=trace[0]['t'],trace[-1]['t']
    label(im,(x,y-24),title,14)
    d.line([(x,y),(x,y+h),(x+w,y+h)],fill=(156,160,150),width=1)
    for key,color in zip(keys,colors):
        points=[(x+w*(r['t']-first)/max(last-first,1e-9),y+h*(1-min(maximum,max(0,r.get(key,0)))/maximum)) for r in trace]
        d.line(points,fill=color,width=2)
    label(im,(x,y+h+7),'0 s',12);label(im,(x+w-45,y+h+7),f'{last-first:.2f} s',12)


def boundary_examples(out):
    """The same ledge with different spatial entry/turn/release recipes."""
    im=Image.new('RGB',(1440,930),PAPER)
    label(im,(38,25),'13 / HOW THIS LEDGE IS PAINTED',30)
    label(im,(38,73),'The main path and materials are shared. Entry, turning and release are actual timed movements.',16)
    points=[[45,100],[112,69],[204,76],[231,130],[350,150]]
    variants=[(0,'moving','blunt','A / Equal load, stationary ends'),
              (1,'moving','lift','B / Enter in motion, unload to turn'),
              (1,'reverse','return','C / Reverse entry, return in release')]
    records=[]
    for i,(revision,entry,ending,title) in enumerate(variants):
        c=InkCanvas(420,225,seed=924)
        painter=RolePainter(c,revision,feedback=False)
        r=painter.draw(points,role='ridge',face='top+side',load=.78,speed=125,
                       entry=entry,ending=ending,explanation={'start_reason':entry,'turn_reason':'the ledge folds into the side','end_reason':ending})
        records.append(r)
        x=36+i*472
        label(im,(x,125),title,18)
        im.paste(c.to_image(),(x,170))
        # These magnifications are explicit crops, not invented brush caps.
        for j,(box,caption) in enumerate([((28,79,82,124),'Entry'),((180,54,249,113),'Turn'),((320,127,372,170),'Release')]):
            crop=c.to_image().crop(box).resize((123,109),Image.Resampling.LANCZOS)
            im.paste(crop,(x+j*139,421));label(im,(x+j*139+5,545),caption,14)
        _curve(im,(x+9,629,402,133),r['trace'],['nominal_pressure_target','pressure'],[(171,91,67),(52,86,81)],1.05,
               'Nominal load (red) / contact load (green)')
        label(im,(x+9,814),f'{r["diagnostics"]["commanded_contact_episodes"]} contact / no restart at the turn',15)
    label(im,(38,877),'A lifted brush leaves no new ink. The visible end can still be pointed, blunt or folded back.',17)
    im.save(out/'painting_03_boundaries.png')
    (out/'painting_boundary_records.json.gz').write_bytes(gzip.compress(json.dumps(records,ensure_ascii=False,separators=(',',':')).encode(),mtime=0))


def feedback_example(out):
    from .painting import paint_mark
    im=Image.new('RGB',(1440,1020),PAPER)
    label(im,(40,25),'14 / EXISTING INK PARTICIPATES IN THE NEXT MOMENT',29)
    label(im,(40,73),'Both trials use the same prior mark and the same nominal stroke. Only density feedback is toggled.',16)
    records=[]
    program=stroke_program([[40,145],[255,126],[440,140],[643,116]],'ridge',revision=1,load=.66,speed=180)
    brush={'width':16.,'pressure_tau':.028,'tip_tau':.014,'angle_tau':.04}
    ink={'ink_capacity':18000.,'concentration':.64,'k_transfer':.07,'initial_water_load':.50,'dry':.30}
    for i,enabled in enumerate([False,True]):
        c=InkCanvas(680,260,seed=912)
        under=stroke_program([[334,59],[355,211]],'tone',revision=1,load=.8,speed=110,tilt=1.)
        paint_mark(c,under,brush={'width':29},ink={'ink_capacity':24000.,'concentration':.75,'k_transfer':.12},seed=617)
        before=c.density.copy()
        rec=paint_mark(c,program,brush=brush,ink=ink,paper={'spread':.4},seed=618,
                       feedback={'enabled':enabled,'density_ceiling':.32,'strength':1.6,'min_multiplier':.12,'lookahead':7.},
                       context={'role':'ridge','start_reason':'same nominal entry','turn_reason':'same nominal path','end_reason':'same lift'})
        records.append({'canvas_seed':912,'width':680,'height':260,'strokes':c.strokes})
        x=33+i*710
        label(im,(x,129),'A / Feedback off' if not enabled else 'B / Observe, reduce load, continue',22)
        im.paste(c.to_image(),(x,173))
        label(im,(x+7,449),'New mark only (density difference, prior mark removed)',14)
        delta=InkCanvas(680,260,seed=912);delta.density[:]=c.density-before
        im.paste(delta.to_image().crop((0,79,680,193)),(x,485))
        _curve(im,(x+17,667,635,142),rec['trace'],['nominal_pressure_target','pressure_target','pressure'],
               [(176,109,78),(106,137,84),(40,70,77)],.9,
               'Nominal / feedback target / actual contact load')
        minimum=rec['diagnostics']['minimum_feedback_multiplier']
        label(im,(x+17,866),f'Minimum load multiplier: {minimum:.3f}',17)
        label(im,(x+17,899),f'Contact episodes: {rec["diagnostics"]["commanded_contact_episodes"]}',16)
    label(im,(40,958),'The local change follows a reduction of contact load and ink supply. It adds no lift; prior ink remains intact.',16)
    im.save(out/'painting_04_feedback.png')
    (out/'painting_feedback_records.json.gz').write_bytes(gzip.compress(json.dumps(records,ensure_ascii=False,separators=(',',':')).encode(),mtime=0))


def supplementary_studies(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    boundary_examples(out)
    feedback_example(out)


def comparison_plate(out):
    out=Path(out)
    if not all((out/f'painting_01_r{i}.png').is_file() for i in (0,1,2)):
        return
    im=Image.new('RGB',(1500,660),PAPER)
    label(im,(38,25),'15 / THREE ROUNDS: MARKS BECOME PART OF THE STONE',28)
    label(im,(38,73),'Same authored form. All three versions use the final contact-coupled ink model.',17)
    captions=['R0 / closed outlines, hanging lines','R1 / varied entry and turn, too many gaps','R2 / staggered cun, thinner broad tones']
    for i in (0,1,2):
        with Image.open(out/f'painting_01_r{i}.png') as src:
            crop=src.crop((78,226,820,722)).resize((474,317),Image.Resampling.LANCZOS)
        im.paste(crop,(24+i*497,161));label(im,(27+i*497,525),captions[i],17)
    label(im,(38,590),'R2 improves grouping and continuity. The main lines still look schematic; it is a study, not a finished landscape.',16)
    im.save(out/'painting_05_comparison.png')
