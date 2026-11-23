"""Frozen v0.21 R2 action compiler and its keyframe factory."""

import numpy as np


def frame(t,x,y,d,tilt=50,azimuth=85,roll=0):
    return dict(t_s=t,x_mm=x,y_mm=y,compression_mm=d,tilt_deg=tilt,azimuth_deg=azimuth,roll_deg=roll)


def gesture(path,variant=1,case=''):
 p=np.array(path['points_mm']);d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];L=d[-1];vertex=d/L;f=path['family']
 if f=='turn':
  u0=np.array([0,.05,.30,.58,.70,.82,.94,1.]);depth=np.array([.105,.140,.14,.135,.118,.125,.13,.11]);tilt=np.full(8,18.);delta=35.;speed=28.;release=.012
 elif f=='axe':
  u0=np.array([0,.06,.18,.45,.70,.88,1.]);depth=np.array([.18,.29,.31,.26,.20,.10,.03]);tilt=np.array([69,71,71,66,52,32,18]);delta=75.;speed=31.;release=.012
 else:
  u0=np.array([0,.10,.28,.60,.85,1.]);depth=np.array([.15,.22,.225,.225,.205,.13]);tilt=np.array([65,67,67,67,65,58]);delta=25.;speed=24.;release=.008
  if variant>=2:delta=50.;speed=23.
  if case=='turn':depth*=.65;speed=19.
 u=np.unique(np.r_[np.linspace(0,1,25),u0,vertex]);q=np.c_[np.interp(u,vertex,p[:,0]),np.interp(u,vertex,p[:,1])]
 duration=max(.055,L/speed);t=u*duration
 compression=np.interp(u,u0,depth);tilts=np.interp(u,u0,tilt)
 # All hand-annotated corners are retained. Uniform nine-key sampling used
 # to shortcut the final turn. Pose still interpolates continuously.
 tangent=np.gradient(q,t,axis=0);angles=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))*180/np.pi
 if variant>=2:
  # Bounded feed-forward for the already present first-order tip response.
  # tau is a controller coefficient, not a new paper/brush physical constant.
  lead=tangent*.010;lim=.28 if f=='turn' else .22
  lead*=np.minimum(1,lim/np.maximum(np.linalg.norm(lead,axis=1),1e-9))[:,None]
  q=q+lead
  if f=='dab':compression*=1.12
  if f=='axe':compression*=.82
 k=[frame(0,*q[0],-.35,tilts[0],angles[0]+delta,0)]
 for a,xy,dd,tt,ang in zip(t,q,compression,tilts,angles):k.append(frame(.012+float(a),*xy,float(dd),float(tt),float(ang+delta),0))
 k.append(frame(.012+duration+release,*q[-1],-.40,tilts[-1],angles[-1]+delta,0))
 return dict(name=f'v021-{f}',keyframes=k)
