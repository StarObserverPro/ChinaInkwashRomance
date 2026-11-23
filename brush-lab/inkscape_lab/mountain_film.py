"""Local pressure solves for spatially disconnected surface pools.

Uses the existing conservative film equation and timestep. This changes the
weak precursor-pressure coupling across large dry gaps, so it is an explicitly
validated numerical approximation, not bitwise identity with the joint solve.
"""
import numpy as np
from scipy.ndimage import binary_dilation,label,find_objects
from .meniscus import MeniscusBrush
def implicit_components(self,dt):
 mask=self.film_h>1e-12
 labs,n=label(binary_dilation(mask,iterations=4))
 if n<2:return MeniscusBrush._implicit_film(self,dt)
 H,W=mask.shape;boxes=[]
 for sl in find_objects(labs):
  if sl:boxes.append([max(0,sl[0].start-4),min(H,sl[0].stop+4),max(0,sl[1].start-4),min(W,sl[1].stop+4)])
 changed=True
 while changed:
  changed=False;merged=[]
  for b in boxes:
   match=next((a for a in merged if a[0]<b[1] and b[0]<a[1] and a[2]<b[3] and b[2]<a[3]),None)
   if match is not None:
    match[:]=[min(match[0],b[0]),max(match[1],b[1]),min(match[2],b[2]),max(match[3],b[3])];changed=True
   else:merged.append(b)
  boxes=merged
 if len(boxes)<2:return MeniscusBrush._implicit_film(self,dt)
 yy,xx=np.nonzero(mask);area=(yy.max()-yy.min()+5)*(xx.max()-xx.min()+5)
 if sum((b[1]-b[0])*(b[3]-b[2]) for b in boxes)>.72*area:return MeniscusBrush._implicit_film(self,dt)
 names=['film_h','film_mass','film_z','theta','perm','paper_water','paper_mobile'];saved={k:getattr(self,k) for k in names};dims=(self.canvas.height,self.canvas.width);total=0.
 try:
  for y0,y1,x0,x1 in boxes:
   for k in names:setattr(self,k,saved[k][y0:y1,x0:x1])
   self.canvas.height=y1-y0;self.canvas.width=x1-x0
   total+=MeniscusBrush._implicit_film(self,dt)
   self.component_solves=getattr(self,'component_solves',0)+1
 finally:
  for k,v in saved.items():setattr(self,k,v)
  self.canvas.height,self.canvas.width=dims
 return total
