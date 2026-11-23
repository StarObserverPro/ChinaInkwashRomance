"""Tutorial priors + adaptive regional counts + receding-horizon MC beams.

This is stochastic search over a calibrated low-cost surrogate. It is not UCT,
neural training, inverse reconstruction of historical hand motions, or a
guarantee of optimality. Actual accepted copies are rendered by mountain_session.
"""
from pathlib import Path
from collections import Counter
import json,math,time,argparse,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,map_coordinates
from .landscape_engine import ROOT
from .landscape_catalog import stroke,load
from .mountain_policy import FAMILIES,NAMES,CONTROLS,prior,explain,RULES
VERSION='tutorial-mountain-driver-0.19.5-morphology'

def prepare(row,side=128,physical_side=18.):
 im=Image.open(ROOT/row['reference']).convert('RGB');ratio=(side-20)/max(im.size)
 w,h=[max(12,round(z*ratio)) for z in im.size];im=im.resize((w,h),Image.Resampling.LANCZOS)
 rgb=np.asarray(im)/255.;luma=rgb@np.array([.2126,.7152,.0722]);hi,lo=np.percentile(luma,[96.5,2.])
 a=.76*np.clip((hi-luma)/max(.12,hi-lo),0,1);a=gaussian_filter(a,.45);a=np.pad(a,10)
 D=-np.log1p(-a);gy,gx=np.gradient(gaussian_filter(a,.6));xx=gaussian_filter(gx*gx,1.5);yy=gaussian_filter(gy*gy,1.5);xy=gaussian_filter(gx*gy,1.5)
 theta=.5*np.arctan2(2*xy,xx-yy)+math.pi/2;coh=np.hypot(xx-yy,2*xy)/(xx+yy+1e-9)
 detail=a-gaussian_filter(a,2.5);dy,dx=np.gradient(gaussian_filter(detail,.45));tx=gaussian_filter(dx*dx,1.1);ty=gaussian_filter(dy*dy,1.1);txy=gaussian_filter(dx*dy,1.1)
 texture_theta=.5*np.arctan2(2*txy,tx-ty)+math.pi/2
 contrast=np.sqrt(np.maximum(0,gaussian_filter(a*a,2)-gaussian_filter(a,2)**2))
 ridge=np.maximum(gaussian_filter(a,.6)-gaussian_filter(a,2.2),0);ry,rx=np.gradient(gaussian_filter(a,.8))
 mm=physical_side/max(a.shape);weight=1+2.8*(a<.025)
 profiles=json.loads((ROOT/'driver19/proxy-profiles.json').read_text())
 return dict(row=row,alpha=a,D=D,theta=theta,texture_theta=texture_theta,contrast=contrast,coherence=coh,ridge=ridge,rx=rx,ry=ry,mm=mm,size=[a.shape[1]*mm,a.shape[0]*mm],weight=weight,profiles=profiles,blur=gaussian_filter(a,2.2),grad=np.array(np.gradient(a)),high=a-gaussian_filter(a,1.3),paper_q=[float(lo),float(hi)])

def loss(A,D):
 a=-np.expm1(-D);diff=a-A['alpha'];grad=np.array(np.gradient(a))-A['grad'];high=a-gaussian_filter(a,1.3)-A['high']
 return float(.40*np.mean(A['weight']*diff*diff)+.20*np.mean((gaussian_filter(a,2.2)-A['blur'])**2)+1.20*np.mean(grad*grad)+.5*np.mean(high*high)+.02*diff.mean()**2)

def at(arr,p):
 x=int(np.clip(round(p[0]),0,arr.shape[1]-1));y=int(np.clip(round(p[1]),0,arr.shape[0]-1));return float(arr[y,x])

def path_trace(A,p,f,length,rng):
 field=A['texture_theta'] if f in ['hemp_long','hemp_short','rope','rub','axe'] else A['theta']
 theta=at(field,p);base=np.array([math.cos(theta),math.sin(theta)])
 if base[1]<0:base=-base
 if f in ('mi','moss'):base=np.array([1.,rng.normal(0,.07)])
 step=.70;n=max(2,round(length/A['mm']/step));paths=[]
 for sign,count in [(-1,n//2),(1,n-n//2)]:
  q=p.copy();v=base*sign;part=[]
  for k in range(count):
   t=at(field,q);z=np.array([math.cos(t),math.sin(t)])
   if np.dot(z,v)<0:z=-z
   if f in ('mi','moss'):z=base*sign
   elif f=='fold':
    t=round(math.atan2(z[1],z[0])/(math.pi/4))*math.pi/4;z=.65*z+.35*np.array([math.cos(t),math.sin(t)])
   v=.6*v+.4*z;v/=max(np.linalg.norm(v),1e-12)
   limit={'hemp_long':.60,'hemp_short':.75,'axe':.50,'rub':.75,'rope':1.10,'cloud':1.70,'wash':.45}.get(f,math.pi)
   anchor=base*sign;angle=math.atan2(anchor[0]*v[1]-anchor[1]*v[0],np.dot(anchor,v));angle=float(np.clip(angle,-limit,limit))
   v=np.array([math.cos(angle)*anchor[0]-math.sin(angle)*anchor[1],math.sin(angle)*anchor[0]+math.cos(angle)*anchor[1]])
   normal=np.array([-v[1],v[0]])
   pull=np.dot(np.array([at(A['rx'],q),at(A['ry'],q)]),normal)
   bend=(.10 if f=='rope' else .045 if f=='cloud' else .018)*math.sin(k*.8+rng.uniform(-.15,.15))
   q=q+v*step+normal*(np.clip(pull*2.0,-.20,.20)+bend)
   if q[0]<5 or q[1]<5 or q[0]>A['D'].shape[1]-6 or q[1]>A['D'].shape[0]-6:break
   if at(A['alpha'],q)<.008 and k>1:break
   part.append(q.copy())
  paths.append(part)
 pts=np.array(list(reversed(paths[0]))+[p]+paths[1])
 if len(pts)<2:return None
 return pts[::2] if len(pts)%2 else np.vstack([pts[::2],pts[-1]])

def proxy(A,pts,f,variation):
 delta=np.diff(pts,axis=0);cum=np.r_[0,np.cumsum(np.linalg.norm(delta,axis=1))];length=cum[-1]*A['mm']
 if length<.07:return None
 rows=sorted(A['profiles'][f],key=lambda x:x['length']);ls=[r['length'] for r in rows]
 sigma=np.interp(length,ls,[r['sigma_mm'] for r in rows])/A['mm']*variation['width'];sigma=max(.43,sigma)
 total=np.interp(length,ls,[r['integrated_density_mm2']/r['length'] for r in rows])*length/(A['mm']**2)*variation['amount']
 pick=rows[int(length>(sum(ls)/2))];u=np.array(pick['axial_u']);weights=np.array(pick['axial_weights']);keep=weights>.001;u=u[keep];weights=weights[keep];weights/=weights.sum()
 pos=[]
 for v in u:
  distance=v*cum[-1]
  if v<0:q=pts[0]+delta[0]/max(cum[1],1e-10)*distance
  elif v>1:q=pts[-1]+delta[-1]/max(cum[-1]-cum[-2],1e-10)*(distance-cum[-1])
  else:q=np.array([np.interp(distance,cum,pts[:,j]) for j in range(2)])
  pos.append(q)
 pos=np.array(pos);pad=3.5*sigma+1;x0=max(0,int(np.floor(pos[:,0].min()-pad)));x1=min(A['D'].shape[1],int(np.ceil(pos[:,0].max()+pad))+1)
 y0=max(0,int(np.floor(pos[:,1].min()-pad)));y1=min(A['D'].shape[0],int(np.ceil(pos[:,1].max()+pad))+1)
 yy,xx=np.mgrid[y0:y1,x0:x1];G=np.zeros(xx.shape)
 for q,w in zip(pos,weights):G+=w*np.exp(-((xx-q[0])**2+(yy-q[1])**2)/(2*sigma*sigma))
 G*=total/max(G.sum(),1e-20)
 return (slice(y0,y1),slice(x0,x1)),G,length

def propose(A,D,rng,regional,phase,branches,neutral=False):
 residual=np.maximum(A['D']-D,0);weight=residual**1.35*(.35+7*A['ridge']);weight[:6]=0;weight[-6:]=0;weight[:,:6]=0;weight[:,-6:]=0
 prob=weight.ravel();prob/=max(prob.sum(),1e-30)
 if prob.sum()<.99:return [],0
 choices=rng.choice(len(prob),size=branches,p=prob);out=[]
 for idx in choices:
  y,x=np.unravel_index(idx,D.shape);p=np.array([x+rng.uniform(-.6,.6),y+rng.uniform(-.6,.6)])
  co=A['coherence'][y,x];ri=A['ridge'][y,x];tone=A['alpha'][y,x];region=(min(3,int(y/D.shape[0]*4)),min(3,int(x/D.shape[1]*4)))
  contrast=A['contrast'][y,x];pr=prior(A['row']['tags'],co,ri,tone,phase,neutral,contrast);counts=regional[region]
  pr=(10*pr+counts)/(10+counts.sum());fi=int(rng.choice(len(FAMILIES),p=pr));f=FAMILIES[fi]
  tool,depth,tilt,water,c,speed,lrange=CONTROLS[f];length=rng.uniform(*lrange)
  pts=path_trace(A,p,f,length,rng)
  if pts is None:continue
  v=rng.uniform(.88,1.12);variation=dict(width=v**.45,amount=v**1.2)
  ans=proxy(A,pts,f,variation)
  if ans is None:continue
  sl,G,length=ans;R=A['D'][sl]-D[sl];W=A['weight'][sl]
  scale=float(np.clip((R*G*W).sum()/max((G*G*W).sum(),1e-20),.10,2.5))
  gain=float(((R*R-(R-scale*G)**2)*W).sum())
  if gain<=0:continue
  spec=dict(family=f,points=(pts*A['mm']).tolist(),length_mm=length,tool=tool,depth=depth*v,tilt=tilt,water=float(np.clip(water*v,0,1)),concentration=c*scale,duration=max(.035,length/speed),roll=float(rng.uniform(0,180)),reason=explain(f,A['row']['tags'],co,ri),region=list(region),proxy_scale=scale,local_gain=gain)
  out.append((gain,sl,G*scale,spec))
 out.sort(key=lambda a:a[0],reverse=True);expanded=[]
 for gain,sl,G,spec in out[:5]:
  nxt=D.copy();nxt[sl]+=G;expanded.append((loss(A,nxt),nxt,spec))
 return sorted(expanded,key=lambda x:x[0])[:2],branches

def search(A,seed=7019,budget=90,branches=96,depth=4,initial=None,neutral=False):
 rng=np.random.default_rng(seed);D=np.zeros_like(A['D']) if initial is None else initial.copy();regional=np.zeros((4,4,len(FAMILIES)));strokes=[];trace=[];attempts=0;start=time.time();initial_loss=loss(A,D)
 while len(strokes)<budget:
  phase=len(strokes)/max(budget,1);nodes,n=propose(A,D,rng,regional,phase,branches,neutral);attempts+=n
  if not nodes:break
  beams=[(l,d,[s]) for l,d,s in nodes]
  for k in range(1,min(depth,budget-len(strokes))):
   children=[]
   for l,d,seq in beams:
    ns,n=propose(A,d,rng,regional,(len(strokes)+k)/budget,max(12,branches//3),neutral);attempts+=n
    children.extend((ll,dd,seq+[ss]) for ll,dd,ss in ns)
   if not children:break
   beams=sorted(children,key=lambda x:x[0])[:2]
  l,d,seq=min(beams,key=lambda x:x[0])
  if l>=loss(A,D):break
  D=d
  for s in seq:
   region=tuple(s['region']);regional[region]*=.97;regional[region+(FAMILIES.index(s['family']),)]+=1
   s['index']=len(strokes);strokes.append(s)
  trace.append(dict(strokes=len(strokes),loss=l,attempts=attempts))
 return dict(version=VERSION,id=A['row']['id'],seed=seed,budget=budget,branches=branches,depth=depth,neutral_text=neutral,attempts=attempts,seconds=time.time()-start,initial_loss=initial_loss,loss=loss(A,D),strokes=strokes,trace=trace,counts=dict(Counter(s['family'] for s in strokes)),regional_counts=regional.tolist(),size_mm=A['size'],normalization_quantiles=A['paper_q']),D

def compile_recipe(plan,row):
 stages=['outline','cun','rub','wash','moss']
 if '烟雨' in row['tags']:stages=['wash','outline','cun','rub','moss']
 def stage(s):return s['family'] if s['family'] in ['outline','rub','wash','moss'] else 'cun'
 ops=[];order=[]
 for st in stages:
  todo=[s for s in plan['strokes'] if stage(s)==st]
  prev=np.array(todo[0]['points'][0]) if todo else None
  while todo:
   s=min(todo,key=lambda s:np.linalg.norm(np.array(s['points'][0])-prev));todo.remove(s);pts=np.array(s['points']);v=np.diff(pts,axis=0);theta=np.rad2deg(np.unwrap(np.arctan2(v[:,1],v[:,0])))+90;theta=np.r_[theta,theta[-1]]
   dep=[s['depth']*.48,s['depth'],s['depth']*.18]
   op=stroke(pts,depth=dep,tilt=s['tilt'],azimuth=theta,roll=s['roll'],duration=s['duration'],name=NAMES[s['family']])
   op.update(tool=s['tool'],loading=load(s['water'],s['concentration']),decision=s);ops.append(op);order.append(s['index']);prev=pts[-1]
  if ops and todo==[] and any(stage(s)==st for s in plan['strokes']):ops[-1]['checkpoint']=dict(outline='勾勒',cun='皴理',rub='侧擦',wash='淡染',moss='点苔')[st]
 return dict(id=row['id'],title=row['artist']+' · '+row['title'],family='山体临摹',size_mm=plan['size_mm'],operations=ops,settle_s=.08,search_order=order)

def save_plan(out,A,plan,D):
 out.mkdir(parents=True,exist_ok=True);(out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));(out/'recipe.json').write_text(json.dumps(compile_recipe(plan,A['row']),ensure_ascii=False,indent=2))
 np.savez_compressed(out/'planning.npz',target=A['alpha'],proxy_density=D)
 def gray(a,name):Image.fromarray(np.uint8(np.clip(1-a,0,1)*255)).save(out/name)
 gray(A['alpha'],'target.png');gray(-np.expm1(-D),'proxy.png')

def metrics(A,physical):
 D=np.load(physical/'fields.npz')['density'];D=np.array(Image.fromarray(D.astype('float32')).resize((A['D'].shape[1],A['D'].shape[0]),Image.Resampling.BOX));a=-np.expm1(-D)
 return dict(loss=loss(A,D),mae=float(np.mean(abs(a-A['alpha']))),ink_ratio=float(a.sum()/max(A['alpha'].sum(),1e-10)),spatial_correlation=float(np.corrcoef(gaussian_filter(a,1.3).ravel(),gaussian_filter(A['alpha'],1.3).ravel())[0,1])),D

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id',default='C00');ap.add_argument('--round',type=int,default=0);ap.add_argument('--budget',type=int,default=90);ap.add_argument('--seed',type=int,default=7019);ap.add_argument('--depth',type=int,default=4);ap.add_argument('--branches',type=int,default=96);a=ap.parse_args()
 rows=json.loads((ROOT/'references19/manifest.json').read_text());row=next(r for r in rows if r['id']==a.id);A=prepare(row);plan,D=search(A,a.seed,a.budget,a.branches,a.depth);out=ROOT/f'driver19/r{a.round}/{a.id}';save_plan(out,A,plan,D);print(json.dumps({k:v for k,v in plan.items() if k not in ['strokes','trace','regional_counts']},ensure_ascii=False))
if __name__=='__main__':main()
