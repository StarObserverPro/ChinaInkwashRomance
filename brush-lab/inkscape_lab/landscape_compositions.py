"""Four hand-authored rock studies, assembled from the same brush actions.

Coordinates describe an invented practice composition, not a trace or a
pixel-matched reconstruction of a historical painting.
"""
import math
import numpy as np
from .landscape_catalog import stroke,curve,load,rinse,refill,checkpoint

def compositions(round_id=0):
    result=[]
    def add(sid,title,en,ops,width,loading,refs):
        result.append(dict(id=sid,title=title,en=en,family='组合练习',width_mm=width,size_mm=[24,18],loading=loading,
            operations=ops,reference_ids=refs,status='study',note='据真实画例归纳组织关系后编写的练习，非原画路径反演。',
            check='勾、皴、擦、点、染共用纸面状态，皴线服从体块与留白',settle_s=.35))
    # Rounded hemp mountains: distant face, main ridges, grouped fibres, moss.
    ops=[stroke(curve((12,10.3),(15,2.3),(20.3,11.7)),depth=[.025,.13,.015],tilt=25,duration=.34),
         stroke(curve((14,10.8),(16.4,4.3),(19.1,11.6)),depth=[.025,.11,.02],tilt=35,duration=.26),rinse(load(.35,25))]
    for pts in [[(3.4,13.4),(4.7,10.2),(5.5,9.7),(6.5,6.2),(8,5.2),(9,3.6),(9.8,3.3)],
                [(9.9,3.4),(11.1,4.6),(12.3,5),(13,7.6),(14.8,9.2),(16.1,13.5)],
                [(6.6,6.2),(8.2,7.4),(9.2,10.5),(9.5,13.5)],
                [(12.3,5.2),(11.7,8.1),(12.5,10.4)]]:
        ops.append(stroke(pts,depth=[.025,.105,.075,.025],tilt=28,azimuth=[70,90,135],duration=.28))
    ops[-1]=checkpoint(ops[-1],'勾：主脊与转面')
    fibres=[((7.4,6.2),(5.9,8),(5.1,11.8)),((7.8,6.8),(6.3,10),(6,12.9)),((8.2,7.7),(7.2,10),(7.2,13)),
        ((6.1,9.6),(5.7,10.9),(5.8,12.7)),((9.4,4.7),(8.6,5.9),(9.1,8.7)),((10,5.2),(10.1,8.5),(11.2,12.4)),
        ((10.5,6),(10.9,9.1),(12.1,13)),((11,7),(11.9,9.4),(12.8,12.9)),((13.1,8.8),(14.1,10.4),(14.9,12.8)),
        ((13.3,10.3),(13.5,12),(14,13.2)),((8.1,10.5),(8.2,12.2),(8.9,13.3)),((4.6,11.5),(4.1,12.4),(4.1,13.6))]
    ops.append(rinse(load(.23,22)))
    for i,abc in enumerate(fibres):
        ops.append(stroke(curve(*abc),depth=[.035,.115 if i%3 else .085,.018],tilt=33,azimuth=15+7*(i%3),duration=.14+.02*(i%3)))
    ops[-1]=checkpoint(ops[-1],'皴：长短披麻分组')
    ops.append(rinse(load(.20,9)))
    for a,b in [((6.2,9.6),(5.8,12.8)),((11,9.5),(12.7,13)),((14,10.7),(15,13))]:
        ops.append(stroke([a,b],depth=[.19,.03],tilt=78,azimuth=90,duration=.13))
    ops.append(rinse(load(.32,52)))
    for x,y,dx in [(7.7,5.4,.35),(8.1,5.15,.45),(8.65,4.45,.27),(9,4.1,.33),(11.1,4.7,.35),(11.6,4.9,.28),(12.2,5.2,.38),(13.6,8.6,.35),(14.1,9,.24),(4.8,10.2,.3),(5.2,9.8,.37)]:
        ops.append(stroke([(x,y),(x+dx,y+.04)],depth=[.12,.025],tilt=65,duration=.035))
    ops[-1]=checkpoint(ops[-1],'点：苔点沿山脊聚散')
    ops.append(rinse(load(.31,20)))
    for pts in [[(3.2,14.5),(5,13.4),(7.1,14),(8.2,15.1)],[(8.8,15.1),(11.1,14.2),(14.8,14.8),(16.5,15.4)],[(18.5,15.2),(18,13),(18.4,10.5)],[(18.1,13.3),(16.9,12.4),(16.1,11.8)],[(18.2,12.4),(19.4,11.8),(20.5,10.9)]]:
        ops.append(stroke(pts,depth=[.025,.12,.02],tilt=25,duration=.16))
    add('S01','披麻山石与疏树','Hemp hills',ops,.82,load(.33,7),['Met-Shen-Zhou','hemp'])

    # Angular rock faces, varied oblique entry. Waterfall is unpainted paper.
    ops=[stroke([(4,12.7),(5.2,5.4),(8.7,3.3),(11.4,4.8),(12.2,7.7)],depth=[.03,.13,.09,.07,.018],tilt=33,azimuth=[90,110,70],duration=.36),
         stroke([(8.8,3.5),(8.2,7.1),(7.6,10.8),(8.1,13.5)],depth=[.035,.12,.10,.02],tilt=42,azimuth=75,duration=.28),
         stroke([(13.4,6.1),(16.3,5.5),(18.4,7.5),(20.5,8.2),(21.6,13.2)],depth=[.03,.13,.11,.07,.02],tilt=40,duration=.30),
         stroke([(16.2,5.7),(15.2,8.7),(15.7,12.4)],depth=[.04,.13,.02],tilt=48,duration=.22)]
    ops[-1]=checkpoint(ops[-1],'勾：断续山脊与瀑布两岸')
    ops.append(rinse(load(.29,13,42,.65)))
    cuts=[(6.1,5.7,-.75,3.0),(7.3,5.1,-.65,3.7),(7.0,8.2,-1.2,3.1),(9.8,5.1,1.0,2.7),(10.2,7.3,1.1,3.2),(8.9,9.2,.7,3.4),
          (16.9,7.1,1,2.7),(18.4,8.1,1.2,3),(16.6,9.1,.6,3.6),(20,10.4,.7,2.4)]
    for i,(x,y,dx,dy) in enumerate(cuts):
        ops.append(stroke([(x-.25,y-.18),(x+.2,y+.3),(x+dx*.6,y+dy*.56),(x+dx,y+dy)],depth=[.065,.30,.18,.02],tilt=[69,76,63,40],azimuth=100+11*(i%3),roll=[10,35,50,60],duration=.15+.015*(i%3)))
    ops[-1]=checkpoint(ops[-1],'皴：斜落、侧拖、提收')
    ops.append(rinse(load(.21,13)))
    for a,b in [((5.7,9),(5.1,12.7)),((9.8,10.2),(10.6,13.4)),((17.7,10.2),(18.6,13.4))]:
        ops.append(stroke([a,b],depth=[.19,.025],tilt=79,azimuth=100,duration=.16))
    ops.append(rinse(load(.28,8)))
    for x,dx in [(12.4,.2),(13.3,-.15),(14.1,.1)]:ops.append(stroke([(x,8.2),(x+dx,11),(x+.15,14.8)],depth=[.018,.05,.015],tilt=12,duration=.18))
    ops.append(rinse(load(.30,48)))
    for x,y in [(5.5,5.6),(6.1,5.2),(7.2,4.5),(8.1,3.8),(14.2,6),(15.4,5.7),(16,5.6),(18.5,7.5),(19.2,7.9)]:
        ops.append(stroke([(x,y),(x+.4,y+.05)],depth=[.13,.025],tilt=66,duration=.04))
    ops.append(rinse(load(.25,26)))
    for pts in [[(3,14.4),(5.5,13.5),(7.2,14.3),(9.3,14.8)],[(15.8,14.5),(18.4,13.8),(21.7,14.8)],[(16.8,15.4),(19,15.1),(21.8,15.4)]]:
        ops.append(stroke(pts,depth=[.035,.13,.02],tilt=40,duration=.18))
    add('S02','斧劈山石与瀑布','Axe-cut waterfall',ops,1.0,load(.30,26),['axe','HKMoA-Pamphlet'])

    # Mi-family: dots are authored by ridge/cluster, never a uniform scatter.
    ops=[]
    layers=[(load(.70,5),[(6,4.9,1.1),(7.3,4.1,1.2),(8.5,4.5,1.4),(5.5,5.8,1.7),(7.4,5.4,1.8),(9.5,5.6,1.5),(4.4,6.8,1.7),(6.8,6.6,1.6),(9.2,6.8,1.6),(11,6.9,1.1)]),
       (load(.65,11),[(14.7,5.8,1),(15.8,5.3,1.4),(17.3,5.9,1.2),(13.3,6.9,1.5),(15.1,6.8,1.7),(17.6,7.1,1.5),(12.4,7.9,1.5),(14.5,7.9,1.8),(16.8,8.1,1.6),(18.7,8.2,1)]),
       (load(.58,18),[(4.5,10.7,1),(5.5,10.1,1.6),(7.1,10.5,1.6),(3.1,11.9,1.3),(4.6,11.6,1.8),(6.8,11.7,1.7),(8.6,11.9,1.6),(2.9,13.1,1.6),(5.1,13,1.7),(7.5,13.2,1.9),(9.5,13.4,1.5)]),
       (load(.50,29),[(15.1,11.9,1.4),(16.2,11.2,1.6),(18,11.7,1.5),(13.9,13,1.7),(16.1,12.7,1.7),(18.2,13,1.7),(20,13.6,1),(13.3,14.3,1.4),(15.5,14.1,1.7),(17.7,14.3,1.9),(19.8,14.6,1.6)])]
    for j,(loading,marks) in enumerate(layers):
        if j:ops.append(rinse(loading))
        for i,(x,y,l) in enumerate(marks):
            ops.append(stroke([(x,y),(x+l*.4,y-.08),(x+l,y+.03)],depth=[.07,.22+.03*(i%2),.025],tilt=74,azimuth=90,roll=i*13,duration=.065+.005*(i%3)))
        ops[-1]=checkpoint(ops[-1],f'第 {j+1} 组湿点与云间留白')
    add('S03','米点云山','Mi-dot cloud hills',ops,1.25,layers[0][0],['mi','HKMoA-Pamphlet'])

    # Folded-belt rock terraces and bare branches: horizontal planes turn down.
    ops=[stroke([(3.4,12.8),(5.1,9.4),(6.1,5.2),(10,4.3),(11.2,6.1),(14.7,6.5),(17.8,10),(20.5,12.8)],depth=[.025,.11,.08,.025],tilt=30,duration=.38),
         stroke([(6.2,5.4),(8.8,5.5),(9.7,6.2),(9.2,8.2)],depth=[.025,.10,.16,.02],tilt=55,azimuth=[90,90,160,180],duration=.20)]
    planes=[(5.6,8.4,4.5),(5.1,10.2,5.4),(4.4,12.1,5.1),(9.7,7.8,3.3),(10.7,9.7,4.2),(10.2,11.8,5.2),(14.2,10.3,3.2),(15.8,12.4,3.7)]
    ops.append(rinse(load(.23,24)))
    for i,(x,y,w) in enumerate(planes):
        ops.append(stroke([(x,y),(x+w*.84,y+.04),(x+w,y+.34),(x+w-.18,y+1.0)],depth=[.025,.10,.18,.02],tilt=[45,60,64,45],azimuth=[90,90,155,170],roll=[0,0,20,30],duration=.16+.015*(i%2)))
    ops[-1]=checkpoint(ops[-1],'折带：水平层面转折')
    ops.append(rinse(load(.22,11)))
    for a,b in [((6.4,7.1),(6.9,9.7)),((11.8,8.2),(12.8,10.8)),((15.5,11.2),(16.3,13.4))]:
        ops.append(stroke([a,b],depth=[.15,.02],tilt=78,azimuth=90,duration=.12))
    ops.append(rinse(load(.30,34)))
    for pts in [[(19.2,13.8),(19,10.9),(19.5,7.4)],[(19.1,11.2),(17.8,10.1),(17,8.3)],[(19.3,9.3),(20.8,8.8),(21.4,7.3)],[(17.9,10.1),(17,10),(16.3,9)],[(20.7,8.9),(21.7,9.1),(22.2,8.3)]]:
        ops.append(stroke(pts,depth=[.12,.08,.02],tilt=20,duration=.16))
    for x,y in [(6.4,5.2),(7.2,5),(9.3,4.6),(10,4.7),(13,6.5),(13.7,6.6),(5.3,9.5),(11,10)]:
        ops.append(stroke([(x,y),(x+.35,y+.06)],depth=[.14,.02],tilt=62,duration=.035))
    ops.append(rinse(load(.32,9)))
    for i in range(3):ops.append(stroke(curve((4+i,15+i*.48),(10,14.8+i*.55),(17-i*.8,15.4+i*.4)),depth=[.015,.055,.012],tilt=12,duration=.24))
    add('S04','折带坡岸与枯树','Folded banks',ops,.76,load(.28,21),['fold','Met-Shen-Zhou'])
    if round_id>=1:improve_compositions(result)
    return result

def improve_compositions(recipes):
    """R0 inspection: S01 lacked face tone; S04 lost the vertical folds."""
    for r in recipes:
        if r['id']=='S01':
            ops=[];phase='distant';fibres=0
            for op in r['operations']:
                if op['kind']=='rinse_load':
                    c=op['loading']['body_concentration_kg_m3']
                    if c==25:op['loading']=load(.38,32);phase='outline'
                    elif c==22:op['loading']=load(.31,30);phase='fibres'
                    elif c==9:op['loading']=load(.36,9);phase='rub'
                    elif c==52:op['loading']=load(.25,34);phase='moss'
                    elif c==20:op['loading']=load(.35,27);phase='tree'
                elif op['kind']=='stroke':
                    if phase=='fibres':
                        if fibres and fibres%3==0:ops.append(refill(load(.32,30)))
                        fibres+=1
                    if phase in ('outline','fibres','tree'):
                        for k in op['gesture']['keyframes'][1:-1]:k['compression_mm']=max(.05,k['compression_mm']*1.18)
                        op['gesture']['keyframes'][-1]['t_s']=op['gesture']['keyframes'][-2]['t_s']+.018
                ops.append(op)
                if op.get('checkpoint','').startswith('勾：'):
                    ops.append(rinse(load(.6,6)))
                    for j,(a,b) in enumerate([((6.9,7.3),(5.2,12.6)),((7.4,8.0),(6.1,13.0)),((10.0,6.5),(12,12.7)),((10.8,7.0),(13.2,13.3)),((13.2,10.0),(14.6,13.2))]):
                        if j:ops.append(refill(load(.60,6)))
                        mid=((a[0]+b[0])*.5,(a[1]+b[1])*.5)
                        ops.append(stroke([a,mid,b],depth=[.10,.35,.06],tilt=79,azimuth=90,duration=.18))
                    ops[-1]=checkpoint(ops[-1],'染：先给坡面一层淡调子')
            r['operations']=ops;r['revision_notes']=['R0 线条太淡、坡面缺体积，苔点又过重。按三笔一组补液，增加局部淡染，降低苔点浓度。']
        elif r['id']=='S04':
            ops=[];phase='outline';plane=0
            for op in r['operations']:
                if op['kind']=='rinse_load':
                    c=op['loading']['body_concentration_kg_m3']
                    if c==24:op['loading']=load(.31,28);phase='planes'
                    elif c==11:op['loading']=load(.32,13);phase='rub'
                    else:phase='other'
                elif op['kind']=='stroke' and phase=='planes':
                    if plane and plane%2==0:ops.append(refill(load(.31,28)))
                    plane+=1;ks=op['gesture']['keyframes'];ks[-2]['compression_mm']=.07;ks[-1]['t_s']=ks[-2]['t_s']+.018
                ops.append(op)
                if op.get('checkpoint','').startswith('折带：'):
                    ops.append(rinse(load(.32,15)))
                    for x,y in [(6.3,6.2),(6.9,8.8),(7.7,10.7),(11.8,8.2),(12.8,10.0),(16.2,11.2)]:
                        ops.append(stroke([(x,y),(x-.2,y+.6),(x+.25,y+1.4)],depth=[.04,.13,.055],tilt=61,azimuth=90,duration=.095))
            r['operations']=ops;r['revision_notes']=['R0 横折之后的立面断墨，成为悬空横线。延长立面接触并按组补液，加少量侧皴交代转面。']
        elif r['id']=='S02':
            ops=[];phase='outline';n=0
            for op in r['operations']:
                if op['kind']=='rinse_load':
                    c=op['loading']['body_concentration_kg_m3']
                    if c==13 and op['loading']['tip_concentration_kg_m3']==42:op['loading']=load(.29,10,30,.7);phase='cut'
                    elif c==13:op['loading']=load(.40,10);phase='rub'
                    elif c==48:op['loading']=load(.26,32);phase='moss'
                    else:phase='other'
                elif op['kind']=='stroke':
                    if phase=='cut':
                        k=op['gesture']['keyframes'];x,y=k[1]['x_mm'],k[1]['y_mm'];xe,ye=k[-2]['x_mm'],k[-2]['y_mm']
                        if n and n%3==0:ops.append(refill(load(.31,10,30,.7)))
                        n+=1;op=stroke([(x+.6,y-.12),(x,y+.04),(x-.08,y+.6),(xe,ye)],depth=[.016,.13,.30,.06],tilt=[80,77,68,40],azimuth=[35,72,95,110],roll=[65,32,12,5],duration=.17,entry=.008,exit=.02)
                    elif phase=='rub':
                        ks=op['gesture']['keyframes'];a=ks[1];b=ks[-2]
                        op=stroke([(a['x_mm'],a['y_mm']),((a['x_mm']+b['x_mm'])/2,(a['y_mm']+b['y_mm'])/2),(b['x_mm'],b['y_mm'])],depth=[.07,.31,.06],tilt=78,azimuth=95,duration=.17)
                ops.append(op)
            r['operations']=ops;r['revision_notes']=['R0 斧劈像孤立深色长点。采用 R2 的轻触横切再侧拖动作；侧擦补足灰面，减轻苔点。']
        elif r['id']=='S03':
            # Contract only the vertical spacing within each ridge group; its
            # position in the landscape and the broad cloud gaps are retained.
            loading_levels=[load(.68,3),load(.64,6),load(.60,11),load(.56,17)]
            anchors=[5.55,6.95,11.70,12.95];ops=[];group=0;i=0
            r['width_mm']=1.45;r['loading']=loading_levels[0]
            for old in r['operations']:
                if old['kind']=='rinse_load':
                    group+=1;ops.append(rinse(loading_levels[group]));i=0;continue
                ks=old['gesture']['keyframes'];x=ks[1]['x_mm'];y=anchors[group]+.62*(ks[1]['y_mm']-anchors[group]);length=(ks[-2]['x_mm']-x)*1.32
                if i and i%4==0:ops.append(refill(loading_levels[group]))
                op=stroke([(x,y),(x+length*.4,y-.045),(x+length,y+.015)],depth=[.04,.19+.02*(i%2),.05],tilt=76,azimuth=90,roll=i*13,duration=.060,entry=.012,exit=.018)
                if old.get('checkpoint'):op['checkpoint']=old['checkpoint']
                ops.append(op);i+=1
            r['operations']=ops;r['revision_notes']=['R0 湿点像四组孤立黑珠。减浓、延长横卧短拖、缩小组内竖向间距，让湿点相接，同时保留云隙。']
