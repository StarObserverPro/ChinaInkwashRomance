"""Changes motivated by saved, actually inspected R0/R1 output boards."""
import math
import numpy as np
from .landscape_catalog import stroke,curve,load,rinse,refill,checkpoint

def round_one(recipes):
    d={r['id']:r for r in recipes}
    def setops(sid,ops,reason,**kwargs):
        d[sid]['operations']=ops;d[sid].update(kwargs);d[sid]['revision_notes']=[reason]
    # R0 C01-C04 were comb-like parallel rows. Author shared slopes and gaps.
    fibres=[((5.9,2.1),(4.2,3.8),(2.8,7.3)),((6.2,2.5),(5.1,4.5),(4.2,7.8)),((6.5,3.1),(5.8,5),(5.6,8.1)),
      ((5,4.2),(4,5.7),(3.8,7.3)),((7.7,2.6),(8,4.5),(9.5,7.8)),((8.2,3.1),(9,4.9),(10.5,7.5)),
      ((9.6,4.4),(10.6,6),(11.8,7.4)),((7.8,4.8),(7.8,6.5),(8.6,8))]
    setops('C01',[stroke(curve(*abc),depth=[.025,.11+.02*(i%2),.02],tilt=25+3*(i%3),azimuth=20,duration=.15+.018*(i%3)) for i,abc in enumerate(fibres)],
        'R0 等距同长；改为两片坡面上错落的长短线群。',loading=load(.25,26),width_mm=.56)
    short=[]
    for i,(x,y,dx,dy) in enumerate([(4.8,2.5,-.5,1.8),(5.7,2.1,-.4,2.1),(6.4,3.1,-.3,2),(4.2,4.1,-.6,2),(5.4,4.6,-.25,2.2),(6.5,5.5,.1,1.9),(3.2,6,-.25,1.5),(4.5,6.2,-.1,1.8),
       (8.6,3,.3,1.7),(9.4,3.5,.5,1.8),(8.4,5,.5,1.9),(10.4,5.1,.7,1.8),(9.4,6.6,.4,1.3)]):
        short.append(stroke(curve((x,y),(x+dx*.3-.15,y+dy*.6),(x+dx,y+dy)),depth=[.025,.10+.02*(i%3),.018],tilt=30,azimuth=20,duration=.10+.012*(i%3)))
    setops('C02',short,'R0 两排孤立短点；改为沿坡相接、错位的短线群。',width_mm=.50,loading=load(.25,27))
    ropes=[];ox=[]
    for i,abc in enumerate(fibres):
        pts=curve(*abc,n=15);t=np.linspace(0,1,len(pts));pts[:,0]+=(.18+.07*(i%3))*np.sin(t*(2.2+.3*(i%3))*math.pi+i*.7)*np.sin(math.pi*t)
        ropes.append(stroke(pts,depth=[.025,.12,.03],tilt=33,azimuth=20,duration=.17+.017*(i%3)))
        for j in range(2):
            pp=pts.copy();pp[:,0]+=(j-.5)*.28;pp[:,1]+=.20*j
            ox.append(stroke(pp,depth=[.02,.075+.012*(i%2),.014],tilt=22,azimuth=20,duration=.16))
    setops('C03',ropes,'R0 各自正弦摆动；改为坡向相关、相邻线有缠绕关系。',width_mm=.53,loading=load(.25,25))
    setops('C04',ox,'R0 稀疏且与解索难分；缩细并成簇加密，保留中间空隙。',width_mm=.35,loading=load(.27,25))
    lotus=[]
    for i,abc in enumerate([((6.4,2.0),(4.6,3.8),(2.6,7.7)),((6.8,2.35),(5.4,4.9),(4.1,8.0)),((6.7,3.0),(6.1,5.3),(5.8,8.1)),((7.2,2.7),(7.3,4.8),(8.2,7.3)),((7.4,3.1),(9.5,4.8),(11.8,6.8)),((8.3,4.1),(9.2,6.1),(10,8)),((5.0,4.8),(4.3,6.1),(3.5,7.6))]):
        lotus.append(stroke(curve(*abc),depth=[.025,.11+.015*(i%2),.02],tilt=28,azimuth=25,duration=.17+.014*(i%3)))
    setops('C05',lotus,'R0 对称扇形；分叉起点错位，并加入次级坡脉。',width_mm=.54,loading=load(.26,26))
    cloud=[]
    for i,(cx,cy,rx,ry,ang) in enumerate([(4.6,3.5,1.55,1.35,.5),(7.1,3.0,1.35,1.0,-.3),(8.7,4.8,1.7,1.4,.3),(5.3,6.2,1.7,1.2,.1),(9.1,7.1,1.4,.8,-.2)]):
        t=np.linspace(-.9*math.pi+ang,.75*math.pi+ang,15);pts=np.c_[cx+rx*np.cos(t),cy+ry*np.sin(t)]
        cloud.append(stroke(pts,depth=[.025,.12,.07,.03],tilt=37,azimuth=np.linspace(20,170,len(t)),duration=.20))
    setops('C06',cloud,'R0 重复螺旋符号；改为大小不一、互相承接的回旋体块。',width_mm=.57,loading=load(.26,27))
    folded=[]
    for i,(x,y,w,drop) in enumerate([(4.4,2.2,4.1,1.5),(3.1,3.8,4.7,1.4),(6.7,4.5,3.9,1.3),(2.4,5.8,4.7,1.7),(6.2,6.6,4.9,1.5)]):
        folded.append(stroke([(x,y),(x+w*.8,y+.03),(x+w,y+.35),(x+w-.12,y+drop)],depth=[.025,.11,.17,.02],tilt=[45,58,63,40],azimuth=[90,90,155,175],roll=[0,0,20,30],duration=.20))
    setops('C07',folded,'R0 等距小钩；折后的立面延长，各层前后错落。',width_mm=.62,loading=load(.25,25))
    cuts=[]
    for i,(x,y,dx,dy) in enumerate([(3.8,2.6,-.6,3.7),(5.2,2.1,-.35,4.2),(7.1,3.3,.55,4.1),(8.4,4.2,.8,3.5),(10,3.1,1.0,3.4)]):
        cuts.append(stroke([(x-.2,y-.15),(x+.12,y+.35),(x+dx*.7,y+dy*.65),(x+dx,y+dy)],depth=[.055,.30,.16,.02],tilt=[68,75,61,37],azimuth=92+10*(i%3),roll=[0,25,40,55],duration=.16+.012*(i%2)))
    setops('C08',cuts,'R0 圆头蝌蚪；斜落后按入、横向铺毫再侧拖，减轻墨头。',width_mm=1.20,loading=load(.28,12,38,.65))
    small=[]
    for i,(x,y,l) in enumerate([(4.5,2.3,2),(5.4,2.1,2.6),(6.6,3,2),(3.7,4.1,2.4),(4.9,4.6,2.7),(6.1,5.2,2.4),(3.2,6.2,1.8),(8.2,3.5,2.1),(9.5,4.1,2.4),(8.4,5.4,2.6),(10.2,6,1.8)]):
        dx=.3+.15*(i%3);small.append(stroke([(x-.15,y),(x+.14,y+.3),(x+dx,y+l)],depth=[.055,.23,.02],tilt=[66,73,40],azimuth=90+10*(i%2),roll=[0,20,40],duration=.10+.01*(i%3)))
    setops('C09',small,'R0 两排同形刀片；按两片坡面交搭，斜入轻收。',width_mm=.70,loading=load(.28,13,37,.55))
    marks=[(5.3,2.0),(6.2,2.4),(4.4,3.1),(5.5,3.5),(6.7,3.7),(3.5,4.5),(4.7,4.9),(5.9,5.1),(3.0,6.1),(4.1,6.7),(5.3,6.8),
           (8.8,3.2),(9.7,3.7),(8.2,4.6),(9.4,5.2),(10.6,5.1),(8.3,6.2),(9.5,6.9),(10.8,6.7)]
    for sid,kind in [('C10','rain'),('C11','bean'),('C12','mi'),('C13','nail')]:
        ops=[]
        for i,(x,y) in enumerate(marks):
            z=.86+.14*(i%3)
            if kind=='rain':pts=[(x,y),(x+.10,y+.65*z)];dep=[.095,.035];tilt=35;az=0;dur=.035
            elif kind=='bean':pts=[(x,y),(x+.3*z,y+.34*z)];dep=[.17,.07];tilt=55;az=80;dur=.045
            elif kind=='mi':pts=[(x-.15,y),(x+.35,y-.04),(x+1.05*z,y+.025)];dep=[.065,.20,.035];tilt=73;az=90;dur=.064
            else:pts=[(x-.08,y),(x+.05,y+.18),(x+.09,y+.8*z)];dep=[.065,.20,.025];tilt=48;az=20;dur=.07
            ops.append(stroke(pts,depth=dep,tilt=tilt,azimuth=az,roll=i*11,duration=dur))
        widths={'rain':.42,'bean':.63,'mi':.95,'nail':.52};loads={'rain':load(.28,28),'bean':load(.29,28),'mi':load(.62,9,20),'nail':load(.26,32)}
        setops(sid,ops,'R0 网格撒点；改为两组坡面，分别保留竖点、豆瓣、横卧湿点、钉头的动作差异。',width_mm=widths[kind],loading=loads[kind])
    horse=[]
    for i,(x,y,w,h) in enumerate([(3,5.2,1.9,2.7),(4.5,3.8,2.0,2.6),(6.1,5.8,2.3,3.3),(8.2,4.7,2.0,2.6),(9.6,6.5,2.1,2.8)]):
        horse.append(stroke([(x,y+1.8),(x+.10,y-h*.55),(x+w*.42,y-h),(x+w,y-h*.35),(x+w*.8,y+1.1)],depth=[.02,.085,.12,.08,.018],tilt=35,azimuth=[20,20,70,150,150],duration=.22))
    setops('C14',horse,'R0 等高拱门；尖棱按前后层次相接，减少满边轮廓。',width_mm=.48,loading=load(.25,26))
    walls=[]
    for i,(x,y,end) in enumerate([(3,3.2,7.7),(3.8,2.5,7.4),(4.6,2.3,7.9),(5.6,2.8,8.1),(6.4,3.7,7.5),(8,3.3,7.9),(8.9,3.0,7.6),(9.7,3.5,8),(10.7,4.2,7.4)]):
        walls.append(stroke([(x,y),(x+.1,y+.65),(x+.3,end)],depth=[.09,.23,.025],tilt=[67,72,43],azimuth=90,duration=.18+.015*(i%2)))
    setops('C15',walls,'R0 垂直栅栏；墙头成组错落，上部侧锋接触向下减弱。',width_mm=.80,loading=load(.30,13,33,.4))
    ghost=[]
    for i,(cx,cy,rx,ry) in enumerate([(4.9,3.7,2.1,1.6),(8.0,4.0,1.9,1.9),(6.3,6.4,2.4,1.4)]):
        t=np.linspace(-.25,1.7*math.pi,21);pts=np.c_[cx+rx*np.cos(t)*(1+.20*np.sin(3*t+i)),cy+ry*np.sin(t)*(1+.15*np.cos(2*t))]
        ghost.append(stroke(pts,depth=[.025,.11,.06,.12,.02],tilt=38,azimuth=np.linspace(20,210,len(t)),duration=.26))
    setops('C16',ghost,'R0 四个孤立椭圆；不规则空腔互相挤接，保留连通空白。',width_mm=.56,loading=load(.25,24))
    stripped=[]
    for i,(x,y,dx) in enumerate([(4,2.8,-.5),(5.2,2.1,-.1),(6.6,3.1,.4),(8.7,2.9,.6),(10,4.3,.5)]):
        pts=[(x,y),(x-.35,y+1.1),(x+.22,y+2.1),(x-.18,y+3.2),(x+dx,7.8)]
        stripped.append(stroke(pts,depth=[.025,.11,.18,.07,.02],tilt=65,azimuth=[75,95,75,100,90],duration=.21))
        if i in (0,2,3):
            stripped.append(stroke([(x+.38,y+.4),(x+.6,y+1.9),(x+.3,y+3.1),(x+dx+.5,7.5)],depth=[.02,.08,.10,.02],tilt=28,azimuth=20,duration=.19))
    setops('C17',stripped,'R0 五根孤立线；加入粗细相邻的骨架与剥离边，仍属形态练习。',width_mm=.65,loading=load(.23,27))
    messy=[]
    for i,abc in enumerate(fibres):
        pts=curve(*abc,n=12);t=np.linspace(0,1,12);pts[:,0]+=.45*np.sin(math.pi*t*1.4+i*.8)
        messy.append(stroke(pts,depth=[.025,.10,.025],tilt=30,azimuth=20+10*i,duration=.17))
        if i%2==0:
            pts2=pts.copy();pts2[:,0]+=.3-.5*t;pts2[:,1]+=.4
            messy.append(stroke(pts2,depth=[.02,.085,.02],tilt=35,azimuth=30+10*i,duration=.16))
    setops('C18',messy,'R0 全部挤向中心成沙漏；相邻线交搭，保留两片主坡。',width_mm=.48,loading=load(.24,26))
    twigs=[]
    for i,pts in enumerate([[(3,3.3),(5.4,2.4),(6.4,4.8)],[(4,4.7),(6.1,3.5),(7.3,6)],[(2.9,6),(4.7,5.3),(6,7.9)],[(4.5,7.3),(5.4,6.1),(7.2,7.5)],[(7.8,3.1),(9.3,4.8),(8.1,6.7)],[(9.3,3.8),(10.5,5.6),(9.2,7.3)],[(7,5.1),(8.6,5.8),(10.7,7.8)]]):
        twigs.append(stroke(pts,depth=[.025,.12,.02],tilt=40,azimuth=45+i*17,duration=.18))
    setops('C19',twigs,'R0 平行横扫；改为相交的硬折线组，方向随岩面变化。',width_mm=.52,loading=load(.25,27))
    alum=[]
    for i,(x,y,r) in enumerate([(5.9,2.9,1.0),(4.3,4.2,1.3),(7.9,4.3,1.35),(5.8,5.8,1.4),(9.5,6.1,1.2),(3.5,6.4,1.0)]):
        alum.append(stroke([(x-r,y+.7),(x-.7*r,y-r*.5),(x+.1*r,y-r),(x+r,y-.1),(x+.8*r,y+.8)],depth=[.02,.09,.12,.07,.02],tilt=30,duration=.18))
        alum.append(stroke([(x-.15,y-.2),(x+.15,y+.4),(x+.4,y+1.0)],depth=[.025,.08,.02],tilt=35,duration=.09))
    setops('C20',alum,'R0 五个小拱形；大小错落的矾头相接，并加内部短皴。',width_mm=.50,loading=load(.27,25))
    # Short marks: avoid regular rows and make the lateral action visible.
    dot_groups=[(3.3,3.2),(4.15,2.7),(5.3,2.8),(6.1,3.4),(4.2,4.0),(8.5,5.6),(9.5,5.3),(10.4,6.2),(8.8,6.8)]
    for sid,vertical in [('B01',False),('B02',True)]:
        ops=[]
        for i,(x,y) in enumerate(dot_groups):
            dx,dy=(.10,.85+.12*(i%2)) if vertical else (1.05+.18*(i%3),.06)
            ops.append(stroke([(x,y),(x+dx*.4,y+dy*.4),(x+dx,y+dy)],depth=[.07,.19,.03],tilt=52 if vertical else 68,azimuth=0 if vertical else 90,duration=.07))
        setops(sid,ops,'R0 点的长轴不明显、排布均匀；延长短拖过程并聚散成组。',loading=load(.31,38))
    accum=[]
    for j,c in enumerate([9,19,35]):
        if j:accum.extend([dict(kind='wait',seconds=.25),dict(kind='dry_limit'),rinse(load(.34,c))])
        accum.append(checkpoint(stroke(curve((2.3+j*.9,4.1+j*.2),(6.4,2.7+j*.3),(11.8-j*.7,5.8-j*.25)),depth=[.06,.32-j*.025,.025],tilt=70,duration=.25),f'第 {j+1} 层，覆盖同一坡面'))
    setops('D08',accum,'R0 三层几乎分离；层间只小幅错位，让浓淡真正叠在同一坡面。',loading=load(.4,9))
    # R0 wet wash exhausted its reservoir into separated dry-looking bands.
    wash=[dict(kind='prewet',saturation=.012)]
    for i,y in enumerate([3.3,4.05,4.8,5.55,6.3]):
        if i:wash.append(refill(load(.90,4)))
        pts=curve((2.8+.18*i,y),(6.8,y-.8),(11.5-.12*i,y+.25))
        if i%2:pts=pts[::-1].copy()
        wash.append(stroke(pts,depth=[.055,.29,.045],tilt=77,azimuth=90,duration=.27))
    setops('D03',wash,'R0 三条湿染互不铺接且耗液后变干；湿纸、逐笔补液、反向铺接。',loading=load(.9,4),width_mm=1.9)
    clouds=[dict(kind='prewet',saturation=.008)]
    for i,(a,b,c) in enumerate([((2.7,3.5),(6,1.4),(11.5,3.0)),((3.1,4.15),(6.6,2.3),(11.3,3.65)),((2.8,6.7),(7,5.6),(11.7,7.4)),((3.4,7.4),(7,6.3),(11,8.0))]):
        if i:clouds.append(refill(load(.85,4)))
        clouds.append(stroke(curve(a,b,c),depth=[.045,.28,.025],tilt=77,azimuth=90,duration=.28))
    setops('E06',clouds,'R0 三根刷带不能形成云隙；将上下染区铺接，保留中间连续纸白。',loading=load(.85,4),width_mm=1.9)
    leaves=[]
    for i,(cx,cy,rx,ry) in enumerate([(3.4,3.3,.65,1),(5.4,2.8,.5,1.05),(7.8,3.5,.65,1.2),(10.3,3.0,.5,.9),(4.5,6.7,.6,1.1),(7,6.6,.7,1.0),(9.8,6.8,.65,1.1)]):
        leaves += [stroke(curve((cx,cy+ry),(cx-rx,cy),(cx+.12,cy-ry)),depth=[.045,.075,.055],tilt=12,duration=.11),
                   stroke(curve((cx+.12,cy-ry),(cx+rx,cy),(cx,cy+ry)),depth=[.04,.075,.05],tilt=12,duration=.11)]
    setops('E03',leaves,'R0 双钩末段断得太早成 V 形；两半按相反方向接合，端点保留接触。',width_mm=.38,loading=load(.34,26))
    needles=[]
    for k,(cx,cy) in enumerate([(3.7,3.5),(9.2,3.4),(6.6,6.9)]):
        for j,a in enumerate([-2.82,-2.38,-2.02,-1.62,-1.10,-.63,-.32]):
            a+=.08*k;l=1.55+.24*((j+k)%3);r=.26+.07*(j%2)
            needles.append(stroke([(cx+r*math.cos(a),cy+r*math.sin(a)),(cx+l*math.cos(a),cy+l*math.sin(a))],depth=[.045,.012],tilt=10,azimuth=math.degrees(a),duration=.060))
    setops('E04',needles,'R0 六针重压同一点形成黑球；松针在枝端附近错开起点并减轻落笔。',width_mm=.35,loading=load(.30,30))
    foliage=[]
    for k,(cx,cy) in enumerate([(3.5,3.5),(9.3,3.3),(6.4,7.1)]):
        for j,(dx,dy) in enumerate([(-1.8,-.5),(-.3,-2.2),(1.7,-.8)]):
            a=np.array([cx+.18*(j-1),cy+.12*(j%2)]);b=np.array([cx+dx,cy+dy]);mid=.42*a+.58*b
            foliage.append(stroke([b,mid,a],depth=[.012,.14,.015],tilt=[35,61,30],azimuth=math.degrees(math.atan2(dy,dx))+90,roll=12*k,duration=.11))
    setops('E08',foliage,'R0 五笔共压一处变成黑团；三笔错位起收，叶中按入、叶端提起。',width_mm=.56,loading=load(.38,24,42,.35))
    return recipes

def round_two(recipes):
    d={r['id']:r for r in recipes}
    r=d['C02'];r['width_mm']=.42;r['loading']=load(.31,26)
    for op in r['operations']:
        ks=op['gesture']['keyframes'];n=len(ks)-2
        for j,k in enumerate(ks[1:-1]):k['compression_mm']=.067+.019*math.sin(math.pi*j/max(n-1,1))
        ks[-1]['t_s']=ks[-2]['t_s']+.018
    r['revision_notes'].append('R1 短披麻仍像叶形点；以较稳定的小压入行笔，减小中段鼓起。')
    r=d['C06'];r['width_mm']=.51
    for i,(a,b,c) in enumerate([((3.9,3.8),(5.5,2.0),(6.0,4.6)),((7.8,5.2),(9.2,3.5),(10.0,5.7)),((4.8,6.4),(6,5.4),(7,7.4))]):
        r['operations'].append(stroke(curve(a,b,c),depth=[.03,.085,.04],tilt=32,azimuth=[60,110,160],duration=.16))
    r['revision_notes'].append('R1 开口圆弧仍像符号；加内层起伏，让回旋层次相接。')
    r=d['C07'];ops=[]
    for i,op in enumerate(r['operations']):
        if i and i%2==0:ops.append(refill(load(.30,25)))
        ks=op['gesture']['keyframes'];ks[-2]['compression_mm']=.07;ks[-1]['t_s']=ks[-2]['t_s']+.02
        ops.append(op)
    r['operations']=ops;r['revision_notes'].append('R1 折后的竖面过早断墨；维持末段接触，并按组补液。')
    for sid,large in [('C08',True),('C09',False)]:
        r=d[sid];ops=[]
        for i,old in enumerate(r['operations']):
            k=old['gesture']['keyframes'];x,y=k[1]['x_mm'],k[1]['y_mm'];xe,ye=k[-2]['x_mm'],k[-2]['y_mm'];h=ye-y
            dx=.70 if large else .43
            pts=[(x+dx,y-.12),(x,y+.06),(x-.06,y+.60 if large else y+.37),(xe,ye)]
            ops.append(stroke(pts,depth=[.016,.13 if large else .095,.31 if large else .20,.035],tilt=[80,77,68,38],
                azimuth=[35,72,95,110],roll=[65,32,12,5],duration=.17 if large else .105,entry=.008,exit=.018))
        r['operations']=ops;r['loading']=load(.29,11 if large else 13,32 if large else 31,.72);r['width_mm']=1.18 if large else .68
        r['revision_notes'].append('R1 落笔边仍圆；先轻触横切，再转入侧拖，避免直接压下形成圆头。')
    r=d['C12'];r['loading']=load(.58,6,13);r['width_mm']=.85
    for i,op in enumerate(r['operations']):
        k=op['gesture']['keyframes'];x,y=k[1]['x_mm'],k[1]['y_mm'];length=1.45+.18*(i%3)
        op.clear();op.update(stroke([(x,y),(x+.45,y-.025),(x+length,y+.02)],depth=[.035,.16,.04],tilt=76,azimuth=90,roll=i*13,duration=.055,entry=.015,exit=.018))
    r['revision_notes'].append('R1 湿点头部偏黑偏圆；降低浓度、延长横卧接触，保留组内搭接。')
    return recipes

def apply_revisions(recipes,round_id):
    if round_id>=1:recipes=round_one(recipes)
    if round_id>=2:recipes=round_two(recipes)
    return recipes
