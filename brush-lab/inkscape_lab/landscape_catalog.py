"""Authored landscape vocabulary, compiled to continuous brush actions.

The recipes approximate named visual/gestural families. They are not recovered
historical trajectories, nor a claim of exhaustive art-historical taxonomy.
"""
from copy import deepcopy
import math
import numpy as np
from .technique_studies import frame

DEMOS={'nail':'5SFhaXYO2-k','rain':'p6-nkVBGffI','bean':'tROpQ_z-KF0','mi':'ZVwzv4dEjJM',
 'hemp':'ZAMisiskqcA','rope':'wOot4m6J9hk','cloud':'7eUVUH1XKUk','lotus':'iFZswA8-B6g',
 'ox':'cq1AbVmd_yA','fold':'yHtYFFTYvNw','axe':'qBByol3Bh1I','rub':'js26X_OrqtQ',
 'horse':'T0f5pEk9SZc','wall':'40NRqMwCOf4','ghost':'euo0BEUgJbM','stripped':'Xa2CmfJ5Vz4'}

def load(w=.35,c=32,tip=None,contrast=0):
    return dict(body_saturation=w,tip_saturation=w,body_concentration_kg_m3=c,
        tip_concentration_kg_m3=c if tip is None else tip,side_contrast=contrast,side_axis_deg=90)

def stroke(points,depth=.18,tilt=35,azimuth=90,roll=0,duration=None,name='stroke',entry=.03,exit=.04,times=None):
    points=np.asarray(points,float);n=len(points)
    if n<2:raise ValueError('stroke requires two points')
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);total=float(lengths.sum())
    if total<1e-8:points[-1,0]+=.02;lengths=np.linalg.norm(np.diff(points,axis=0),axis=1);total=float(lengths.sum())
    u=np.r_[0,np.cumsum(np.maximum(lengths,.001))];u/=u[-1]
    duration=max(.08,total/32.) if duration is None else duration
    clock=duration*u if times is None else np.asarray(times,float)
    if len(clock)!=n or clock[0]!=0 or np.any(np.diff(clock)<=0):raise ValueError('invalid motion times')
    duration=float(clock[-1])
    def values(v):
        if np.isscalar(v):return np.full(n,v)
        v=np.asarray(v,float)
        return v if len(v)==n else np.interp(u,np.linspace(0,1,len(v)),v)
    dep,til,azi,rol=[values(v) for v in [depth,tilt,azimuth,roll]]
    d0=(points[1]-points[0])/max(lengths[0],1e-8);d1=(points[-1]-points[-2])/max(lengths[-1],1e-8)
    start=points[0]-.24*d0;end=points[-1]+.30*d1
    k=[frame(0,*map(float,start),-.38,float(til[0]),float(azi[0]),float(rol[0]))]
    for i,p in enumerate(points):k.append(frame(float(entry+clock[i]),*map(float,p),float(dep[i]),float(til[i]),float(azi[i]),float(rol[i])))
    k.append(frame(entry+duration+exit,*end,-.40,float(til[-1]),float(azi[-1]),float(rol[-1])))
    return dict(kind='stroke',gesture=dict(name=name,keyframes=k))

def curve(a,b,c,n=9):
    t=np.linspace(0,1,n)[:,None];return (1-t)**2*np.array(a)+2*t*(1-t)*np.array(b)+t*t*np.array(c)

def rinse(loading):return dict(kind='rinse_load',loading=loading)
def refill(loading):return dict(kind='load',loading=loading)
def checkpoint(op,label):op['checkpoint']=label;return op

def catalog(round_id=0):
    out=[]
    def add(sid,title,en,family,ops,*,width=.85,loading=None,reference=None,check='',status='gesture',note='',size=(14,10),ink=None):
        r=dict(id=sid,title=title,en=en,family=family,width_mm=width,size_mm=list(size),loading=loading or load(),operations=ops,
            reference_ids=[reference] if reference else ['MFA-Sue-Yang'],check=check,status=status,note=note,settle_s=.35)
        if reference in DEMOS:r['reference_url']='https://www.youtube.com/watch?v='+DEMOS[reference]
        if ink:r['ink']=ink
        out.append(r)
    # A: kinematic controls; paired gestures share the same brush and paper.
    add('A01','中锋勾线','Centered outline','用笔',[
        stroke(curve((2,3),(6.5,2),(12,3)),depth=[.06,.13,.08],tilt=8),
        stroke(curve((2,6),(7,7.4),(12,6)),depth=[.07,.14,.06],tilt=10)],width=.65,check='线条两边相对均衡，转向时不变成平头刷带')
    add('A02','侧锋铺面','Side stroke','用笔',[stroke([(2,3),(5,3.3),(8,2.8),(12,3)],depth=[.12,.35,.29,.06],tilt=74),
        stroke([(2,6),(5,6.2),(9,6.3),(12,6)],depth=[.1,.30,.25,.04],tilt=65,azimuth=110)],width=1.3,loading=load(.4,16,50,.55),check='笔腹与笔尖浓淡有别；不能只把中锋线加粗')
    add('A03','顺锋行笔','Following stroke','用笔',[stroke([(2,3),(6,3.3),(12,3)],depth=[.10,.20,.06],tilt=58,azimuth=5),stroke([(2,6),(6,6.3),(12,6)],depth=[.10,.20,.06],tilt=58,azimuth=5)],check='用与逆锋相同的纸、墨、路径比较方向影响')
    add('A04','逆锋行笔','Against the tip','用笔',[stroke([(2,3),(6,3.3),(12,3)],depth=[.10,.20,.06],tilt=58,azimuth=185),stroke([(2,6),(6,6.3),(12,6)],depth=[.10,.20,.06],tilt=58,azimuth=185)],check='运动反向与弯曲历史影响墨迹；仍是降阶接触近似')
    add('A05','露锋起收','Exposed tip','用笔',[stroke([(2,3),(4,3),(8,3),(12,3)],depth=[.015,.18,.16,.015],tilt=[12,30,30,10]),stroke([(2,6),(5,6),(9,6),(12,6)],depth=[.012,.14,.17,.01],tilt=20)],width=.8,check='起收由轻入轻出形成，不后处理削尖')
    add('A06','藏锋回护','Concealed entry','用笔',[stroke([(2.4,3),(2,2.8),(2.2,3.1),(6,3),(11.7,3),(11.5,3.25)],depth=[.05,.14,.20,.16,.12,.04],tilt=25,azimuth=[20,170,90,90,180,220]),stroke([(2.3,6),(2,5.8),(2.1,6.1),(7,6),(11.5,6)],depth=[.05,.13,.18,.14,.035],tilt=25)],check='逆入、折回、回护是实际路径，不加端部印章')
    add('A07','提按起伏','Lift and press','用笔',[stroke([(2,3),(4,3),(7,3),(10,3),(12,3)],depth=[.03,.10,.40,.1,.02],tilt=45),stroke([(2,6),(4,6),(6,6),(8,6),(10,6),(12,6)],depth=[.03,.27,.06,.30,.09,.02],tilt=40)],width=1.05,check='粗细与接触滞后连续变化')
    add('A08','顿挫停行','Dwell and advance','用笔',[stroke([(2,3),(5,3),(5.03,3),(7.6,3.2),(7.63,3.2),(12,3)],depth=[.06,.12,.34,.12,.32,.04],tilt=40,times=[0,.14,.24,.35,.45,.55]),stroke([(2,6),(5,6),(5.03,6),(9,6),(12,6)],depth=[.06,.13,.27,.14,.02],tilt=40,times=[0,.14,.25,.39,.5])],check='停顿与压入分开定义，观察局部供墨和接触展开')
    add('A09','圆转','Rounded turn','用笔',[stroke(curve((2,3),(10,.8),(11,7)),depth=[.08,.20,.10],tilt=30,azimuth=[70,110,170]),stroke(curve((3,6),(4,2.8),(8.2,6)),depth=[.07,.18,.04],tilt=28,azimuth=[20,80,160])],width=.75,check='转向不断笔，不在拐弯重置笔毛')
    add('A10','折转','Angular turn','用笔',[stroke([(2,2.5),(9,2.5),(9.4,3),(9.4,6.4),(12,6.5)],depth=[.06,.18,.28,.16,.04],tilt=45,azimuth=[90,90,150,170,80]),stroke([(2.5,6),(5.6,6),(5.8,7.7),(8,7.8)],depth=[.08,.2,.18,.04],tilt=45)],width=.75,check='折角保持连续毛束状态；避免圆球角')
    add('A11','绞转换面','Rolling the brush','用笔',[stroke([(2,3),(5,3),(8,3),(12,3)],depth=[.08,.28,.24,.035],tilt=60,roll=[0,40,100,150]),stroke([(2,6),(6,6),(10,6),(12,6)],depth=[.08,.28,.15,.02],tilt=60,roll=[160,90,25,0])],width=1.1,loading=load(.32,12,55,.7),check='同一毛身份随滚转换面，不随机切换刷头')
    add('A12','散锋破笔','Separated dry tuft','用笔',[stroke([(2,3),(5,3.2),(8,2.8),(12,3)],depth=[.12,.35,.28,.08],tilt=75,roll=[0,20,-15,25]),stroke([(2,6),(6,6.3),(10,6),(12,6.2)],depth=[.13,.32,.26,.06],tilt=70,roll=[35,0,50,0])],width=1.5,loading=load(.13,65),check='分离来自低供液和离散毛束接触，不能变成喷点')
    # B: dotting and rubbing.
    positions=[(2.4,2.5),(5.2,2.2),(8.1,2.9),(11.4,2.4),(3.5,5.5),(6.5,6.2),(9.5,5.8),(11.5,7.3)]
    add('B01','横点苔','Horizontal moss dots','点擦',[stroke([(x,y),(x+.7,y+.05)],depth=[.22,.10],tilt=65,duration=.055) for x,y in positions],width=.95,loading=load(.37,48),check='每个点是短按与拖动；疏密形成组团')
    add('B02','竖点苔','Vertical moss dots','点擦',[stroke([(x,y),(x+.05,y+.7)],depth=[.2,.09],tilt=55,azimuth=0,duration=.055) for x,y in positions],width=.75,loading=load(.36,46),check='点的长轴来自行笔与姿态')
    dots=[(2.8,3),(3.6,2.5),(4,3.6),(3.2,4.1),(8.8,5.8),(9.8,5.1),(10.6,6.3),(9.6,7)]
    add('B03','攒点／聚散','Grouped dots','点擦',[stroke([(x,y),(x+.3,y+.22)],depth=[.17+.035*(i%3),.045],tilt=48,duration=.045) for i,(x,y) in enumerate(dots)],width=.85,loading=load(.3,54),check='局部聚合、组间留空；不均匀撒点')
    add('B04','破笔散点','Broken tuft dots','点擦',[stroke([(x,y),(x+.42,y+.13)],depth=[.26,.06],tilt=73,roll=i*29,duration=.06) for i,(x,y) in enumerate(positions)],width=1.35,loading=load(.14,60),check='看得到接触方向和断续，避免悬空圆点')
    add('B05','横擦','Lateral dry rubbing','点擦',[stroke([(2,3),(5,2.8),(9,3.3),(12,3)],depth=[.15,.34,.29,.08],tilt=80,azimuth=90),stroke([(2.5,5),(6,5.5),(11.5,5.2)],depth=[.15,.32,.08],tilt=78),stroke([(3,7),(7,7.3),(11,7)],depth=[.12,.26,.07],tilt=76)],width=1.45,loading=load(.16,42),reference='rub',check='擦是低液量、铺毫接触的宽断片')
    add('B06','直擦皴','Vertical rubbing','点擦',[stroke([(x,2),(x+.2,4.5),(x-.15,7.8)],depth=[.11,.27,.06],tilt=75,azimuth=10,roll=15*i) for i,x in enumerate([3,5.4,8,10.5])],width=1.25,loading=load(.18,36),reference='rub',check='四次连续擦笔，允许供墨渐耗')
    # C: texture-stroke organizations, including the CUHK museum's 16 types.
    def lines(kind,n=7):
        ops=[]
        for i in range(n):
            x=2.1+i*1.35;y=2+.45*math.sin(i*1.7);end=7.5-.5*math.cos(i*1.1)
            if kind=='hemp':pts=curve((x,y),(x-.6,4.8),(x+.8,end));depth=[.07,.17,.05];tilt=28
            elif kind=='rope':
                t=np.linspace(0,1,14);pts=np.c_[x+.42*np.sin(t*3*math.pi+i*.6)+t*.7,y+t*(end-y)];depth=[.06,.15,.08];tilt=35
            elif kind=='ox':
                t=np.linspace(0,1,10);pts=np.c_[x+.30*np.sin(t*2.5*math.pi+i),y+t*(end-y)];depth=[.04,.11,.025];tilt=24
            elif kind=='wall':pts=[(x,y),(x+.12,y+1),(x+.35,end)];depth=[.20,.25,.045];tilt=64
            else:raise ValueError(kind)
            ops.append(stroke(pts,depth=depth,tilt=tilt,azimuth=15,duration=.18+.014*(i%3)))
        return ops
    add('C01','长披麻皴','Long hemp fibre','皴法',lines('hemp'),width=.65,loading=load(.29,29),reference='hemp',check='长线顺坡分组，疏密与长短有节奏')
    short=[]
    for j in range(2):
        for i in range(5):
            x=2.4+1.9*i+.4*j;y=2.1+j*3+.3*math.sin(i)
            short.append(stroke(curve((x,y),(x-.25,y+1),(x+.4,y+2)),depth=[.06,.16,.04],tilt=30,azimuth=15,duration=.11))
    add('C02','短披麻皴','Short hemp fibre','皴法',short,width=.62,loading=load(.28,30),reference='hemp',check='短线交错相接，不排成等距梳齿')
    add('C03','解索皴','Unravelled rope','皴法',lines('rope'),width=.62,loading=load(.3,28),reference='rope',check='曲线有共同坡向，同时有错位和回旋')
    add('C04','牛毛皴','Ox hair','皴法',lines('ox',8),width=.45,loading=load(.26,27),reference='ox',check='细密弯曲短线保留呼吸空间，避免毛线团')
    lotus=[stroke(curve((6.5+.17*i,2+.12*i),(4+1.1*i,4.2),(2.3+1.5*i,8-.25*abs(i-3))),depth=[.06,.16,.03],tilt=32,azimuth=25,duration=.19) for i in range(7)]
    add('C05','荷叶皴','Lotus leaf','皴法',lotus,width=.62,loading=load(.3,31),reference='lotus',check='从山脊向坡脚分流，不能只是放射线图')
    cloud=[]
    for i in range(5):
        x=2.3+1.85*i;y=2.2+.4*(i%2);t=np.linspace(-.7*math.pi,1.15*math.pi,15)
        pts=np.c_[x+.85*np.cos(t),y+1.4+1.15*np.sin(t)];pts=np.vstack([pts,[x+.5,6.6+.3*math.sin(i)]])
        cloud.append(stroke(pts,depth=[.07,.2,.13,.04],tilt=45,azimuth=np.linspace(20,200,len(pts)),duration=.22))
    add('C06','卷云皴','Roiling cloud','皴法',cloud,width=.70,loading=load(.30,30),reference='cloud',check='回旋线组织成凸凹体块，保留内部空隙')
    folded=[]
    for i in range(5):
        x=2.1+i*.8;y=2+i*1.3
        folded.append(stroke([(x,y),(x+5.4-.6*i,y+.1),(x+5.8-.6*i,y+.7),(x+5.6-.6*i,y+1.1)],depth=[.04,.14,.22,.025],tilt=58,azimuth=[85,85,160,175],roll=[0,0,30,40],duration=.18))
    add('C07','折带皴','Folded belt','皴法',folded,width=.7,loading=load(.24,29),reference='fold',check='横折与侧转形成层面，层间留白')
    axe=[]
    for i,(x,y,l) in enumerate([(2.2,2.4,3.4),(6.1,2.0,4),(9.6,2.6,3.6),(3.5,5.7,2.5),(7.8,5.4,2.9)]):
        axe.append(stroke([(x,y),(x+.4,y+.45),(x+.8,y+l)],depth=[.28,.40,.025],tilt=[72,74,45],azimuth=15,roll=[0,15,25],duration=.14))
    add('C08','大斧劈皴','Large axe cut','皴法',axe,width=1.30,loading=load(.32,18,58,.75),reference='axe',check='重落侧拖轻收，墨面有明确受力边')
    small=[]
    for j in range(2):
        for i in range(5):
            x=2.1+i*1.85+.4*j;y=2.2+j*3+.25*math.sin(i*1.5)
            small.append(stroke([(x,y),(x+.4,y+.4),(x+.7,y+1.7)],depth=[.20,.28,.025],tilt=67,azimuth=15,duration=.08))
    add('C09','小斧劈皴','Small axe cut','皴法',small,width=.82,loading=load(.30,20,50,.6),reference='axe',check='小面交错搭接，避免均一刀片阵列')
    def dotfield(kind):
        ops=[]
        for j in range(3):
            for i in range(4):
                x=2.4+2.7*i+.35*math.sin(i+j);y=2+j*2.1+.2*math.cos(i*1.7)
                if kind=='rain':dx,dy,dep,til=.13,.58,.18,48
                elif kind=='bean':dx,dy,dep,til=.32,.48,.27,58
                elif kind=='mi':dx,dy,dep,til=.80,.05,.23,72
                elif kind=='nail':dx,dy,dep,til=.15,.90,.26,50
                ops.append(stroke([(x,y),(x+dx,y+dy)],depth=[dep,.025 if kind=='nail' else dep*.55],tilt=til,azimuth=0 if kind in ('rain','nail') else 85,duration=.045+.008*((i+j)%2)))
        return ops
    for sid,title,en,key,w,loading in [('C10','雨点皴','Raindrop','rain',.55,load(.29,36)),('C11','豆瓣皴','Bean dot','bean',.86,load(.33,31)),('C12','米点皴','Mi dots','mi',1.05,load(.55,18,31)),('C13','钉头皴','Nail head','nail',.64,load(.28,40))]:
        add(sid,title,en,'皴法',dotfield(key),width=w,loading=loading,reference=key,check='短接触由按、移、提形成；各点按坡面聚散')
    horse=[]
    for i in range(6):
        x=2.2+i*1.65;y=2.2+.65*(i%2)
        horse.append(stroke([(x,y+4.8),(x-.1,y+1),(x+.45,y),(x+1.15,y+1.6),(x+.9,y+4)],depth=[.04,.13,.18,.13,.03],tilt=40,azimuth=[20,20,80,150,150],duration=.20))
    add('C14','马牙皴','Horse tooth','皴法',horse,width=.6,loading=load(.28,31),reference='horse',check='尖棱形体以转折勾皴衔接')
    add('C15','雨淋墙头皴','Rain on wall','皴法',lines('wall'),width=.95,loading=load(.30,18,52,.45),reference='wall',check='上实下虚，垂直面逐步拖散')
    ghost=[]
    for i,(cx,cy,rx,ry) in enumerate([(4,3.3,1.5,1.2),(8.3,3.4,1.9,1.5),(5.1,6.6,1.8,1),(10,6.6,1.1,1.4)]):
        t=np.linspace(.2,2*math.pi-.45,18);pts=np.c_[cx+rx*np.cos(t)*(1+.18*np.sin(3*t)),cy+ry*np.sin(t)]
        ghost.append(stroke(pts,depth=[.06,.17,.1,.19,.03],tilt=45,azimuth=np.linspace(0,220,len(t)),duration=.22))
    add('C16','鬼脸皴','Ghost face','皴法',ghost,width=.7,loading=load(.28,28),reference='ghost',check='不规则空腔与缠绕边界相接；非面部图标')
    stripped=[]
    for i in range(5):
        x=2.5+2*i;pts=[(x,2.1),(x-.4,3.2),(x+.15,4.3),(x-.45,5.2),(x+.4,7.6)]
        stripped.append(stroke(pts,depth=[.03,.12,.22,.10,.025],tilt=60,azimuth=[10,25,-10,30,5],duration=.22))
    add('C17','抽筋剥皮皴','Stripped tendons','皴法',stripped,width=.85,loading=load(.24,32),reference='stripped',check='瘦硬骨架与侧擦交织，保留剥落空隙')
    messy=[]
    for i in range(8):
        x=2.2+1.25*i;y=2+.3*(i%3)
        messy.append(stroke(curve((x,y),(7+.5*math.sin(i),4.5),(x+.8*math.cos(i),7.5)),depth=[.05,.13,.04],tilt=35,azimuth=15+12*i,duration=.17))
    add('C18','乱麻皴','Tangled hemp','皴法',messy,width=.54,loading=load(.24,29),reference='Met-Shen-Zhou',check='线群交搭但仍有主坡向')
    wood=[stroke([(2+.4*i,2+.7*i),(5.5+.7*(i%2),3.5+.45*i),(10.8-.4*i,2.6+.75*i)],depth=[.07,.15,.035],tilt=42,azimuth=75+i*12,duration=.17) for i in range(7)]
    add('C19','乱柴皴','Broken twigs','皴法',wood,width=.62,loading=load(.25,32),reference='Met-Shen-Zhou',check='硬折、交错、疏密；不随机放置无关系短线')
    alum=[]
    for x,y,r in [(3.5,3,.9),(6,2.4,.8),(8.6,3,1.0),(5.1,5.1,1.1),(8,6,.9)]:
        alum.append(stroke([(x-r,y+.6),(x-.6*r,y-r),(x+.3*r,y-1.1*r),(x+r,y),(x+.65*r,y+.7)],depth=[.035,.12,.15,.10,.025],tilt=35,duration=.17))
    add('C20','矾头组织','Alum-head forms','皴法',alum,width=.62,loading=load(.28,31),reference='EDB-Chinese-Painting',check='矾头是山石组织单元，结合勾与短皴，不视为单根特殊笔刷')
    # D: controlled ink preparation and same-sheet interactions.
    shades=[]
    for i,c in enumerate([100,65,38,13,3]):
        if i:shades.append(rinse(load(.42,c)))
        shades.append(checkpoint(stroke([(2.3+i*2.3,2.5),(2.1+i*2.3,4.5),(2.6+i*2.3,7.5)],depth=[.10,.25,.035],tilt=50,azimuth=15,duration=.18),f'墨色 {i+1}'))
    add('D01','墨色五阶','Five ink densities','墨法',shades,width=.95,loading=load(.42,100),reference='HKMoA-Ink',check='浓度改变，水量与动作相同；五阶不是五种物理墨水定律')
    add('D02','焦墨枯笔','Dense dry ink','墨法',[stroke([(2,3),(6,3.4),(12,3)],depth=[.09,.23,.05],tilt=63),stroke([(2,6),(7,6.4),(12,6)],depth=[.1,.25,.035],tilt=65)],width=1.15,loading=load(.13,105),reference='HKMoA-Ink',check='浓而少水，须同时看到墨黑和接触断续')
    wash=[stroke(curve((2,3),(7,2),(12,3)),depth=[.14,.34,.10],tilt=76),stroke(curve((2,4.7),(7,3.7),(12,4.7)),depth=[.15,.33,.08],tilt=75),stroke(curve((3,6.5),(7,5.5),(11.8,6.5)),depth=[.12,.30,.06],tilt=72)]
    add('D03','淡墨湿染','Dilute wet wash','墨法',wash,width=1.8,loading=load(.85,6),reference='HKMoA-Ink',check='淡墨铺接，水分和颜料在同一纸层传输')
    def broken(base,next_load):
        return [checkpoint(stroke([(2,3.3),(6,3),(12,3.6)],depth=[.16,.4,.10],tilt=75,duration=.32),'第一层湿墨'),rinse(next_load),
            checkpoint(stroke([(4,2),(6,4),(8,6.8)],depth=[.12,.27,.05],tilt=50,duration=.22),'第二层入湿纸')]
    add('D04','浓破淡','Dark into light','墨法',broken(None,load(.6,65)),width=1.55,loading=load(.85,8),reference='HKMoA-Ink',check='两次笔触共用纸水状态，交界非图层透明叠加')
    add('D05','淡破浓','Light into dark','墨法',broken(None,load(.9,5)),width=1.55,loading=load(.65,58),reference='HKMoA-Ink',check='淡液只能输运和稀释移动颜料，不能涂白擦掉固定墨')
    add('D06','水破墨','Water into ink','墨法',broken(None,load(.95,0)),width=1.55,loading=load(.68,42),reference='HKMoA-Ink',check='清水不含白颜料；固定颜料不可凭空消失')
    add('D07','墨破水','Ink into water','墨法',broken(None,load(.5,55)),width=1.55,loading=load(.95,0),reference='HKMoA-Ink',check='先画清水，随后墨笔进入实际润湿区')
    accum=[]
    for j,c in enumerate([10,24,44]):
        if j:accum.extend([dict(kind='wait',seconds=.25),dict(kind='dry_limit'),rinse(load(.32,c))])
        accum.append(checkpoint(stroke(curve((2+j*.5,2.8+j*1.5),(6,2+j*1.6),(11.7-j*.6,4.3+j*1.5)),depth=[.08,.30,.05],tilt=65,duration=.24),f'第 {j+1} 层'))
    add('D08','积墨：干层极限','Layered ink / dry limit','墨法',accum,width=1.65,loading=load(.4,10),reference='DPM-Chen-Shizeng',status='approximation',note='干层状态保持颜料位置并把水记入蒸发账本；未求解完整干燥期间迁移。',check='旧层保留，新层有深浅与交叠')
    add('D09','泼写／没骨','Broad wet boneless','墨法',[stroke([(3,2.8),(6,3.4),(10.5,3)],depth=[.22,.5,.12],tilt=80,duration=.25),stroke([(3.5,4.8),(7,5.4),(11,5)],depth=[.20,.43,.08],tilt=78,duration=.26),stroke([(4.5,6.5),(7.3,7),(10,6.7)],depth=[.17,.38,.06],tilt=76,duration=.23)],width=2.15,loading=load(.9,13,30,.55),reference='HKMoA-Ink',status='approximation',note='覆盖阔笔泼写与没骨块面；未模拟倾倒、空中液滴、飞溅。',check='以墨块直接成形，保留未着墨空隙')
    add('D10','宿墨：粗粒近似','Aged ink proxy','墨法',[stroke([(2,3),(6,3.3),(12,3)],depth=[.08,.28,.06],tilt=65),stroke([(2,6),(7,6.3),(12,6)],depth=[.1,.28,.035],tilt=68)],width=1.15,loading=load(.35,45),ink={'fine_fraction':.30,'viscosity_pa_s':.0045},reference='HKMoA-Ink',status='approximation',note='仅改变颗粒比例与黏度；未模拟真实老化絮凝，不声称为校准的宿墨。',check='与普通墨并列检查，不把颗粒变粗等同于已实现宿墨')
    # E: landscape elements, composed exclusively from the same strokes.
    branches=[[(6.3,8),(6,5.5),(6.5,2)],[(6.1,5.9),(4.2,4.3),(3,2.5)],[(6.2,5.4),(8.6,4.1),(10.8,2.4)],[(4.4,4.5),(3.1,4.4),(2.2,3.2)],[(8.4,4.3),(8.5,2.8),(9,1.7)],[(6.4,3.8),(5.1,2.8),(4.9,1.6)]]
    add('E01','鹿角枝','Upward branches','树水云', [stroke(pts,depth=[.20 if i==0 else .11,.13,.025],tilt=25,azimuth=30,duration=.2) for i,pts in enumerate(branches)],width=.85,loading=load(.34,33),reference='Met-Shen-Zhou',check='枝干由主干向上分叉，分枝处不悬空')
    crab=[[(6.3,8),(6,5.3),(6.6,2)],[(6.1,5.4),(4,4.1),(2.4,5.7)],[(6.4,3.7),(8.8,3.1),(11.4,4.7)],[(4.3,4.3),(3.7,3.8),(2.8,4.7)],[(8.8,3.2),(10,2.9),(11,3.9)],[(6.3,6.6),(8,5.6),(9.8,7)]]
    add('E02','蟹爪枝','Downturned branches','树水云',[stroke(curve(*pts),depth=[.20 if i==0 else .1,.14,.025],tilt=32,azimuth=[30,70,130],duration=.2) for i,pts in enumerate(crab)],width=.82,loading=load(.32,34),reference='Met-Shen-Zhou',check='枝条下垂转折，与鹿角枝的组织清晰区别')
    leaves=[]
    for i,(cx,cy) in enumerate([(3,3),(5.2,2.5),(7.8,3),(10.4,2.7),(4.2,6),(7,6.5),(9.8,6)]):
        leaves += [stroke(curve((cx,cy+1.1),(cx-.65,cy),(cx+.12,cy-.8)),depth=[.06,.10,.025],tilt=15,duration=.09),stroke(curve((cx,cy+1.1),(cx+.8,cy+.2),(cx+.12,cy-.8)),depth=[.06,.10,.02],tilt=15,duration=.09)]
    add('E03','夹叶双钩','Outlined leaves','树水云',leaves,width=.48,loading=load(.3,35),reference='Met-Shen-Zhou',check='双钩围出叶内留白，线端相接')
    needles=[]
    for cx,cy in [(3.7,3.5),(9.2,3.2),(6.6,6.8)]:
        for a in np.linspace(-2.8,-.3,6):needles.append(stroke([(cx,cy),(cx+1.9*math.cos(a),cy+1.9*math.sin(a))],depth=[.10,.014],tilt=12,azimuth=math.degrees(a),duration=.06))
    add('E04','松针','Pine needles','树水云',needles,width=.42,loading=load(.32,37),reference='Met-Shen-Zhou',check='针束围绕枝端组织，不能成为均匀星芒印章')
    waves=[]
    for i in range(5):
        t=np.linspace(0,1,18);waves.append(stroke(np.c_[2+10*t,2+i*1.3+.15*np.sin(t*2*math.pi+i*.4)],depth=[.035,.09,.025],tilt=15,azimuth=90,duration=.28))
    add('E05','水纹勾线','Water ripples','树水云',waves,width=.5,loading=load(.32,15),reference='Met-Shen-Zhou',check='横向平缓线的长短和间距，不用噪声模拟水')
    clouds=[stroke(curve((2,3.2),(7,1),(12,2.8)),depth=[.10,.34,.04],tilt=78,duration=.28),stroke(curve((2.6,6.4),(7,4.9),(12,6.8)),depth=[.10,.30,.04],tilt=75,duration=.28),stroke(curve((3.8,8),(8,7.1),(11.8,8.1)),depth=[.08,.26,.03],tilt=75,duration=.22)]
    add('E06','云气烘染留白','Cloud reserve','树水云',clouds,width=1.8,loading=load(.78,8),reference='HKMoA-Ink',check='云由周围着墨留下纸白；没有白色覆盖层')
    falls=[stroke([(3,2),(4.4,3.2),(4.6,5),(4.9,8)],depth=[.08,.24,.20,.03],tilt=63,azimuth=15),stroke([(8.6,2),(7.8,3.4),(7.6,5.8),(7.7,8)],depth=[.08,.25,.17,.03],tilt=63,azimuth=170),rinse(load(.32,10)),stroke([(5.2,3),(5.5,5),(5.7,8)],depth=[.04,.07,.02],tilt=12),stroke([(6.8,3.3),(6.5,5.9),(6.5,8.1)],depth=[.035,.07,.02],tilt=12)]
    add('E07','瀑布留白与水线','Waterfall reserve','树水云',falls,width=.95,loading=load(.4,28),reference='Met-Shen-Zhou',check='两侧山石夹出流路，水线保持淡细')
    foliage=[]
    for i,(cx,cy) in enumerate([(3.5,3.2),(9.1,3.1),(6.3,6.8)]):
        for j,(dx,dy) in enumerate([(-1.1,-1.4),(0,-1.7),(1.2,-1.2),(-1.3,.2),(1.2,.3)]):
            foliage.append(stroke([(cx,cy),(cx+dx,cy+dy)],depth=[.20,.02],tilt=55,azimuth=math.degrees(math.atan2(dy,dx))+90,duration=.055))
    add('E08','介字／个字点叶','Grouped leaf marks','树水云',foliage,width=.65,loading=load(.42,35),reference='MFA-Sue-Yang',check='叶笔围绕枝端疏密交错，不是复制图章')
    from .landscape_compositions import compositions
    return revise(out,round_id)+compositions(round_id)

def revise(recipes,round_id):
    """Only actual later visual feedback is added here, in versioned blocks."""
    from .landscape_revisions import apply_revisions
    return apply_revisions(recipes,round_id)
