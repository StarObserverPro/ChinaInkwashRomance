"""Local, background-corrected cross sections; never fed into raster physics.
Reports geometry and optical differences separately. No single winner score.
"""
import json
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter1d
from .landscape_engine import ROOT
from .stroke_reading import sample_path

def cross_sections(alpha,points,radius,background=True):
 p,_=sample_path(points,80);v=np.gradient(p,axis=0);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-9);normal=np.c_[-v[:,1],v[:,0]]
 r=np.linspace(-radius,radius,81);xy=p[:,None,:]+normal[:,None,:]*r[None,:,None]
 z=map_coordinates(alpha,[xy[:,:,1],xy[:,:,0]],order=1,mode='constant');z=gaussian_filter1d(z,.7,axis=1)
 bg=np.quantile(np.c_[z[:,:12],z[:,-12:]],.35,axis=1) if background else np.zeros(len(z))
 z=np.maximum(0,z-bg[:,None]);dr=r[1]-r[0]
 widths=[];masses=[];centers=[];peaks=[]
 for a in z:
  threshold=max(.025,float(a.max())*.22);active=a>threshold
  # Pick the nearest peak-connected interval, excluding separate adjacent marks.
  starts=np.flatnonzero(active&~np.r_[False,active[:-1]]);ends=np.flatnonzero(active&~np.r_[active[1:],False])
  if not len(starts):widths.append(0);masses.append(0);centers.append(0);peaks.append(0);continue
  dist=[0 if l<=40<=h else min(abs(l-40),abs(h-40)) for l,h in zip(starts,ends)];j=int(np.argmin(dist));lo,hi=starts[j],ends[j]
  if dist[j]>24:widths.append(0);masses.append(0);centers.append(0);peaks.append(0);continue
  w=a[lo:hi+1];rr=r[lo:hi+1];mass=w.sum()*dr
  widths.append((hi-lo+1)*dr);masses.append(mass);centers.append(float(w@rr/max(w.sum(),1e-9)));peaks.append(w.max())
 return dict(width=np.array(widths),mass=np.array(masses),center=np.array(centers),peak=np.array(peaks))

def metrics(target,actual,points,radius,mpp):
 a=cross_sections(target,points,radius);b=cross_sections(actual,points,radius,False)
 usable=a['peak']>.05;core=usable.copy();core[:8]=False;core[-8:]=False
 if not core.any():core=usable
 w0=np.median(a['width'][core]) if core.any() else 0;w1=np.median(b['width'][core]) if core.any() else 0
 segments={}
 for name,sl in [('head',slice(0,16)),('body',slice(24,56)),('tail',slice(64,80))]:
  segments[name]=dict(target_width_mm=float(np.median(a['width'][sl])*mpp),actual_width_mm=float(np.median(b['width'][sl])*mpp),target_alpha=float(np.mean(a['peak'][sl])),actual_alpha=float(np.mean(b['peak'][sl])))
 return dict(width_ratio=float(w1/max(w0,1e-9)),width_relative_error=float(abs(w1-w0)/max(w0,1e-9)),coverage=float(np.mean(b['peak'][usable]>.06)) if usable.any() else 0.,center_error_mm=float(np.mean(abs(b['center'][usable]-a['center'][usable]))*mpp) if usable.any() else 0.,target_width_mm=float(w0*mpp),actual_width_mm=float(w1*mpp),segments=segments)

def evaluate21(task,row):
 out=ROOT/task['output'];density=np.load(out/'fields.npz')['density'];alpha=-np.expm1(-density)
 target=np.load(ROOT/f'stroke20/{row["id"]}-target.npy');yy,xx=np.indices(target.shape);cs=task['scale']['canvas_factor'];mpp=task['scale']['mm_per_reference_pixel']
 sample=map_coordinates(alpha,[(yy*mpp+2*cs)/.05,(xx*mpp+2*cs)/.05],order=1,mode='constant')
 result=dict(strokes=[metrics(target,sample,p['local_points'],p['radius'],mpp) for p in row['paths']],limits='Separate local proxies; scan/paper optical calibration and manual annotation remain uncertain. No scalar artistic score.')
 (out.parent/'metrics21.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return result
