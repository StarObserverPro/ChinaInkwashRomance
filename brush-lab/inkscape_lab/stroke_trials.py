"""Bounded actual-physics comparisons of stroke interpretation and size."""
import json,argparse,copy
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt
from .landscape_engine import ROOT
from .stroke_reading import prepare,build_task,render,profile

def evaluate(task,row):
    out=ROOT/task['output'];z=np.load(out/'fields.npz');a=-np.expm1(-z['density'])
    cs=task['scale']['canvas_factor'];mpp=task['scale']['mm_per_reference_pixel']
    target=np.load(ROOT/f'stroke20/{row["id"]}-target.npy')
    yy,xx=np.indices(target.shape);sample=map_coordinates(a,[(yy*mpp+2*cs)/.05,(xx*mpp+2*cs)/.05],order=1,mode='constant')
    mask=np.zeros_like(target,bool);narrow=np.zeros_like(target,bool);profiles=[];terms=[]
    paths=row['paths'][:len(task['recipe']['operations'])]
    for path in paths:
        p=np.array(path['local_points']);q,_=__import__('inkscape_lab.stroke_reading',fromlist=['sample_path']).sample_path(p,160)
        seed=np.ones_like(mask);ix=np.rint(q[:,0]).astype(int);iy=np.rint(q[:,1]).astype(int);valid=(ix>=0)&(ix<mask.shape[1])&(iy>=0)&(iy<mask.shape[0]);seed[iy[valid],ix[valid]]=False
        dist=distance_transform_edt(seed);mask|=dist<path['radius'];narrow|=dist<path['radius']*.55
        p0=path['measured'];p1=profile(sample,path['local_points'],path['radius'])
        w0=np.array(p0['width_px']);w1=np.array(p1['width_px']);m0=np.array(p0['integrated_alpha']);m1=np.array(p1['integrated_alpha'])
        width=float(np.mean(abs(w1-w0))/max(np.mean(w0),1e-9))
        shape=float(np.mean(abs(m1/max(m1.mean(),1e-9)-m0/max(m0.mean(),1e-9))))
        terms.append((width,shape));profiles.append(dict(target=p0,actual=p1))
    # Adjacent unannotated strokes are not forced into a chosen stroke's mask.
    l1=float(np.mean(abs(sample[mask]-target[mask])));spill=float(np.mean(sample[~mask]))
    width=float(np.mean([x[0] for x in terms]));shape=float(np.mean([x[1] for x in terms]));mass=float(abs(sample[mask].mean()-target[mask].mean()))
    loss=.38*l1+.20*mass+.18*width+.12*shape+.12*spill
    stats=dict(loss=loss,masked_alpha_mae=l1,width_profile_error=width,longitudinal_profile_error=shape,mean_ink_error=mass,spill=spill,profiles=profiles,
               note='人工圈选附近的轮廓/截面代理分数；不等于艺术质量或真实动作识别精度。')
    (out.parent/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    Image.fromarray(np.uint8(np.clip(1-sample,0,1)*255)).save(out.parent/'aligned-ink.png')
    return stats

def job(task,row):
    r=render(task);m=evaluate(task,row)
    return dict(id=row['id'],output=task['output'],seconds=r['seconds'],loss=m['loss'],parameters=task['parameters'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,default=0);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--search',action='store_true');ap.add_argument('--scale',action='store_true');a=ap.parse_args()
    rows=prepare();tasks=[]
    if a.scale:
        row=next(r for r in rows if r['id']=='axe')
        variants=[('base',dict()),('half_same_pen',dict(canvas_scale=.5)),('double_same_pen',dict(canvas_scale=2.)),('double_wider_pen',dict(canvas_scale=2.,width=2.,pressure=2.))]
        for name,p in variants:tasks.append((build_task(row,1,p,tag='scale/'+name,single=True),row))
    elif a.search:
        rng=np.random.default_rng(2002)
        for row in rows:
            candidates=[dict()]
            for i in range(7):
                candidates.append(dict(width=float(rng.uniform(.65,1.1)),pressure=float(rng.uniform(.62,1.1)),water=float(rng.uniform(.55,1.05)),concentration=float(rng.uniform(.85,1.35)),speed=float(rng.uniform(.85,1.3)),tilt_delta=float(rng.uniform(-8,5))))
            for i,p in enumerate(candidates):tasks.append((build_task(row,2,p,tag=f'search/{row["id"]}-{i:02}',single=True),row))
    else:
        for row in rows:tasks.append((build_task(row,a.round),row))
    summary=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(job,t,r):t for t,r in tasks}
        for f in as_completed(fs):
            v=f.result();summary.append(v);print(json.dumps(v,ensure_ascii=False),flush=True)
    tag='scale' if a.scale else ('search' if a.search else f'r{a.round}')
    (ROOT/f'stroke20/{tag}-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
