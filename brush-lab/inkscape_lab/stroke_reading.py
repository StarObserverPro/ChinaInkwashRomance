"""v0.20: explicit stroke readings -> constrained contact programmes.

The ROIs and seed paths are human annotations. Measurements describe visible
marks, not uniquely recovered historical hand motion. Images never enter the
brush/paper solver; it receives only keyframes and finite brush loading.
"""
from pathlib import Path
import json, hashlib, copy
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, percentile_filter, map_coordinates
from .landscape_engine import ROOT
from .technique_studies import frame
from .landscape_catalog import stroke, load
from .mountain_policy import CONTROLS

VERSION = 'stroke-reading-0.20.0'

def reference_scale(reference_id):
    # Image extents were manually checked in the restored full scans. Mounting
    # is excluded. The Shen Zhou scan is a scroll DETAIL: use height, not the
    # 6.407-m whole-scroll width. These are approximate photographic scales.
    info={
      'T01':dict(object_id=40020,image_mm=[410.,1029.],image_bounds_px=[305,179,1755,3815],copy_ratio=.30,uncertainty_fraction=.03),
      'T07':dict(object_id=40086,image_mm=[260.,251.],image_bounds_px=[602,327,3040,2672],copy_ratio=1.,uncertainty_fraction=.03),
      'C00':dict(object_id=45683,image_mm=[6407.163,206.375],image_bounds_px=[0,95,1961,603],copy_ratio=.40,uncertainty_fraction=.05,height_only=True),
    }[reference_id].copy()
    x,y,x1,y1=info['image_bounds_px'];sx=info['image_mm'][0]/(x1-x);sy=info['image_mm'][1]/(y1-y)
    info['reference_mm_per_px']=sy if info.get('height_only') else (sx+sy)/2
    info['source']='https://www.metmuseum.org/art/collection/search/'+str(info['object_id'])
    info['note']='画心边界由扫描图人工估计，非实物尺规标定；笔宽仍需由墨迹与接触标定联合选择。'
    return info

def annotations():
    return [
      dict(id='turn',title='弘仁：转折勾勒',reference_id='T01',
        crop=[460,436,620,535],mm_per_px=.085,role='勾骨',family='turn',legacy='outline',
        observations=['轮廓细而连续，长行段相对均匀；右端由横行转为下行。','转角没有一个明显膨大的圆墨团；附近两道短横墨迹另算两笔。'],
        hypothesis='从左向右行，临转处略提、换向后继续；轻提离纸。它是待验证的动作假设。',
        uncertainty=['静态成品不能证明唯一行笔方向或精确压力。','左侧竖段与横段是否连续一笔不清，本次仅圈选上沿至右转段。'],
        paths=[dict(family='turn',points=[[480,460],[510,469],[540,471],[561,465],[578,456],[585,455],[593,462],[600,484]],radius=7),
               dict(family='dab',points=[[592,509],[605,506]],radius=5),
               dict(family='dab',points=[[593,522],[608,521]],radius=5)],confidence='中：轮廓角色清楚，时序为假设'),
      dict(id='axe',title='马远：侧锋斜向斧劈',reference_id='T07',
        crop=[272,414,351,532],mm_per_px=.105,role='皴面',family='axe',legacy='axe',
        observations=['短笔上段较实、较宽，斜向下拖出分叉和尖细尾段。','多笔靠拢成一组，长短有别；底材的规则织纹不能当作飞白。'],
        hypothesis='侧锋落下并短按，笔腹接纸后向下斜拖；随行渐提并收立，使尾部变窄。',
        uncertainty=['绢本的吸墨与本项目纸模型不同，不能将边缘扩散直接一比一拟合。','有重叠的尾段，三笔分割是人工假设，不作为真值笔序。'],
        paths=[dict(family='axe',points=[[302,429],[309,451],[319,486],[329,516]],radius=12),
               dict(family='axe',points=[[282,449],[288,470],[301,502]],radius=7),
               dict(family='axe',points=[[320,441],[329,474],[340,507]],radius=7)],confidence='中：斧劈候选有形态和馆藏上下文支持'),
      dict(id='dab',title='沈周：短横点苔',reference_id='C00',
        crop=[347,34,418,93],mm_per_px=.165,role='点苔',family='dab',legacy='moss',
        observations=['点呈短横或微斜的扁形，具有明确长轴。','三处浓淡、大小和间距不同，彼此保留纸白，不是均匀散点。'],
        hypothesis='斜置笔锋，短按铺开，作少量侧移后提离；一组内续用剩余墨。',
        uncertainty=['浓淡不直接等于按压力，可能同时受上墨和复笔影响。','先后次序在静态图中不可确定，本轮顺序是可复演的工作假设。'],
        paths=[dict(family='dab',points=[[357,42],[368,43]],radius=5),
               dict(family='dab',points=[[384,59],[403,62]],radius=6),
               dict(family='dab',points=[[379,80],[401,85]],radius=7)],confidence='较高：短横点的几何角色清楚；真实笔序未知')]

def sample_path(points,n=64):
    p=np.asarray(points,float);u=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u/=u[-1]
    q=np.linspace(0,1,n)
    return np.c_[np.interp(q,u,p[:,0]),np.interp(q,u,p[:,1])],u

def image_alpha(im):
    a=np.asarray(im.convert('RGB'),float)/255
    g=a@np.array([.2126,.7152,.0722]);g=gaussian_filter(g,.65)
    bg=percentile_filter(g,88,size=25)
    od=np.maximum(0,np.log((bg+.01)/(g+.01)))
    return np.clip(od/max(float(np.percentile(od,96)),.05)*.68,0,.90)

def profile(alpha,points,radius):
    p,_=sample_path(points);v=np.gradient(p,axis=0);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-12)
    normals=np.c_[-v[:,1],v[:,0]];r=np.linspace(-radius,radius,49)
    xy=p[:,None,:]+normals[:,None,:]*r[None,:,None]
    a=map_coordinates(alpha,[xy[:,:,1],xy[:,:,0]],order=1,mode='constant')
    mass=a.sum(axis=1)+1e-9;centroid=(a*r).sum(axis=1)/mass
    width=2.355*np.sqrt((a*(r-centroid[:,None])**2).sum(axis=1)/mass)
    return dict(u=np.linspace(0,1,len(p)).tolist(),width_px=width.tolist(),
                integrated_alpha=(mass*(r[1]-r[0])).tolist(),
                lateral_asymmetry=((a[:,25:].sum(axis=1)-a[:,:24].sum(axis=1))/mass).tolist(),
                center_alpha=map_coordinates(alpha,[p[:,1],p[:,0]],order=1).tolist())

def prepare():
    out=ROOT/'stroke20';out.mkdir(exist_ok=True);rows=annotations()
    for row in rows:
        row['scale_calibration']=reference_scale(row['reference_id'])
        row['mm_per_px']=row['scale_calibration']['reference_mm_per_px']*row['scale_calibration']['copy_ratio']
        im=Image.open(ROOT/f'references19/{row["reference_id"]}-reference.png').convert('RGB')
        x,y,x1,y1=row['crop'];patch=im.crop(row['crop']);patch.save(out/f'{row["id"]}-reference.png')
        alpha=image_alpha(patch);np.save(out/f'{row["id"]}-target.npy',alpha)
        row['source_sha256']=hashlib.sha256((ROOT/f'references19/{row["reference_id"]}-reference.png').read_bytes()).hexdigest()
        row['size_mm']=[(x1-x)*row['mm_per_px']+4,(y1-y)*row['mm_per_px']+4]
        for p in row['paths']:
            p['local_points']=(np.array(p['points'])-[x,y]).tolist()
            p['measured']=profile(alpha,p['local_points'],p['radius'])
            p['points_mm']=(np.array(p['local_points'])*row['mm_per_px']+2).tolist()
    (out/'readings.json').write_text(json.dumps(dict(version=VERSION,method='人工圈选与读笔假设；程序测量截面。不是自动笔触识别器。',cases=rows,abstention=dict(reference_id='C00',crop=[180,194,357,301],reason='浅淡皴线与擦染重叠，不能可靠拆成一笔。暂不交给单笔编译器。')),ensure_ascii=False,indent=2))
    return rows

def phase_gesture(path,params):
    """Different grammars retain independent travel, press, tilt and release."""
    f=path['family'];points=np.asarray(path['points_mm']);dense,_=sample_path(points,101)
    length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
    scale=params.get('pressure',1.);speed=params.get('speed',1.)
    if f=='turn':
        u=np.array([0,.035,.15,.35,.58,.70,.80,.88,1.])
        depths=np.array([.035,.080,.077,.076,.073,.043,.041,.065,.010])
        tilts=np.array([13,16,17,18,18,12,12,18,8.])
        tt=np.array([0,.05,.17,.35,.56,.69,.83,.91,1.])
        phases=['触纸','稳锋','行','行','临转','提','换向','再行','提收']
        azdelta=np.zeros(len(u))+35;duration=max(.12,length/34)
    elif f=='axe':
        u=np.array([0,.008,.05,.22,.5,.78,1.])
        depths=np.array([.11,.24,.29,.24,.145,.07,.008])
        tilts=np.array([70,74,74,68,58,38,16.])
        tt=np.array([0,.13,.24,.39,.61,.81,1.])
        phases=['侧锋触纸','短按','铺腹','斜拖','渐提','收立','提收']
        azdelta=np.array([78,78,72,64,46,30,15]);duration=max(.12,length/35)
    elif f=='dab':
        u=np.array([0,.015,.10,.50,.85,1.])
        depths=np.array([.045,.17,.205,.20,.12,.018])
        tilts=np.array([58,69,74,70,53,30.])
        tt=np.array([0,.22,.40,.68,.85,1.])
        phases=['触纸','按','铺开','短移','提','离纸']
        azdelta=np.array([80,80,80,75,70,60]);duration=.065+length/100
    else:raise ValueError('unread stroke family '+f)
    if params.get('sustain',False):
        if f=='turn': depths=np.maximum(depths,.068);depths[-1]=.035
        elif f=='dab':
            u=np.array([0,.12,.30,.58,.88,1.]);tt=np.array([0,.12,.30,.58,.88,1.]);depths=np.array([.09,.19,.205,.20,.18,.07])
        elif f=='axe': depths=np.array([.08,.16,.26,.25,.21,.13,.025]);tt=np.array([0,.035,.10,.28,.54,.80,1.])
    widths=np.interp(u,path['measured']['u'],path['measured']['width_px'])
    widths=np.clip(widths/max(np.median(widths),1e-9),.6,1.5)
    depths*=scale*(.8+.2*widths)
    tilts=np.clip(tilts+params.get('tilt_delta',0),0,82)
    q=np.c_[np.interp(u,np.linspace(0,1,101),dense[:,0]),np.interp(u,np.linspace(0,1,101),dense[:,1])]
    tangent=np.gradient(dense,axis=0);angles=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))*180/np.pi
    azimuth=np.interp(u,np.linspace(0,1,101),angles)+azdelta+params.get('azimuth_delta',0)
    roll=params.get('roll',0)+np.linspace(0,18 if f=='axe' else 3,len(u))
    # Entry/exit are explicit vertical approach/lift, not shared diagonal caps.
    enter=.018*params.get('entry',1.);duration/=speed
    k=[frame(0,*q[0],-.35,tilts[0],azimuth[0],roll[0])]
    for i in range(len(u)):k.append(frame(enter+tt[i]*duration,*q[i],depths[i],tilts[i],azimuth[i],roll[i]))
    k.append(frame(enter+duration+.020,*q[-1],-.38,tilts[-1],azimuth[-1],roll[-1]))
    return dict(name=f,keyframes=k),[dict(phase=p,u=float(a),time_s=float(enter+t*duration)) for p,a,t in zip(phases,u,tt)]

def build_task(row,round_id,params=None,tag=None,single=False):
    params=params or {};ops=[]
    canvas_scale=params.get('canvas_scale',1.)
    row=copy.deepcopy(row)
    row['size_mm']=(np.array(row['size_mm'])*canvas_scale).tolist()
    for p in row['paths']:p['points_mm']=(np.array(p['points_mm'])*canvas_scale).tolist()
    widths=dict(fine=.40,side=.85,wash=1.3)
    widths['fine']*=params.get('width',1.);widths['side']*=params.get('width',1.)
    for i,path in enumerate(row['paths'][:1] if single else row['paths']):
        f=path['family'];legacy=dict(turn='outline',axe='axe',dab='moss')[f]
        points=np.array(path['points_mm']);length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        tool='fine' if f=='turn' else 'side'
        if round_id==0:
            _,depth,tilt,water,c,speed,_=CONTROLS[legacy]
            tangent=np.diff(points,axis=0);theta=np.rad2deg(np.unwrap(np.arctan2(tangent[:,1],tangent[:,0])))+90
            op=stroke(points,depth=[depth*.48,depth,depth*.18],tilt=tilt,azimuth=np.r_[theta,theta[-1]],roll=0,duration=max(.035,length/speed),name=f)
            op['loading']=load(water,c);phases=[dict(phase='统一轻重轻包络',u=.5,time_s=0)]
        else:
            stroke_params={**params,**params.get('per_stroke',{}).get(str(i),{})}
            gesture,phases=phase_gesture(path,stroke_params);op=dict(kind='stroke',gesture=gesture)
            base=dict(turn=(.23,23,32,0),axe=(.27,16,46,.60),dab=(.26,30,48,.30))[f]
            w,body,tip,contrast=base;w*=stroke_params.get('water',1.);body*=stroke_params.get('concentration',1.);tip*=stroke_params.get('concentration',1.)
            if i==0 or tool!=ops[-1]['tool']:
                op['loading']=dict(body_saturation=min(w,1),tip_saturation=min(w*.93,1),body_concentration_kg_m3=body,tip_concentration_kg_m3=tip,side_contrast=contrast,side_axis_deg=90)
        op.update(tool=tool,checkpoint=f'第 {i+1} 笔',reading=dict(case=row['id'],stroke=i,role=row['role'],family=f,phases=phases,measurement=path['measured'],inference=row['hypothesis'],confidence=row['confidence']))
        ops.append(op)
    name=tag or f'r{round_id}/{row["id"]}'
    return dict(round=round_id,seed=20,widths_mm=widths,output=f'stroke20/{name}/physical',
                recipe=dict(id=row['id'],title=row['title'],family='先读笔再落笔',size_mm=row['size_mm'],operations=ops,settle_s=.04),parameters=params,reading_version=VERSION,
                scale=dict(reference_calibration=row['scale_calibration'],canvas_factor=canvas_scale,mm_per_reference_pixel=row['mm_per_px']*canvas_scale,grid_um=50,
                           note='尺度对照中的换笔仅改变横向笔束宽度与压入行程；毛径、毛长和储墨容量保持固定，不宣称是完整毛笔的几何相似缩放。'))

def render(task):
    from . import mountain_session,mountain_paper,mountain_paper_fast
    hashes=mountain_paper_fast.verify_source_pair()
    mountain_paper.entry_native=mountain_paper_fast.entry_native
    mountain_session.WIDTHS=task['widths_mm']
    out=ROOT/task['output'];out.parent.mkdir(parents=True,exist_ok=True)
    (out.parent/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2))
    result=mountain_session.run(task)
    (out.parent/'implementation.json').write_text(json.dumps(dict(version=VERSION,backend_hashes=hashes,reading_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    return result

if __name__=='__main__':prepare()
