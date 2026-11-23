from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,numpy as np
from .landscape_catalog import stroke,load
from .landscape_engine import ROOT
from .mountain_policy import CONTROLS
from .mountain_session import run
def probe(f,length):
 tool,depth,tilt,water,c,speed,_=CONTROLS[f]
 op=stroke([(2.5,3),(2.5+length*.5,3),(2.5+length,3)],depth=[depth*.48,depth,depth*.18],tilt=tilt,azimuth=90,duration=max(.035,length/speed),name=f)
 op.update(tool=tool,loading=load(water,c))
 recipe=dict(id=f'{f}-{length}',title=f,family=f,size_mm=[9,6],operations=[op],settle_s=.08)
 path=f'driver19/probes/{f}-{length}';r=run(dict(recipe=recipe,output=path,round=-1))
 D=np.load(ROOT/path/'fields.npz')['density'];y,x=np.indices(D.shape);total=D.sum();cy=float((D*y).sum()/max(total,1e-20));cx=float((D*x).sum()/max(total,1e-20))
 # Profile in physical mm, independent of final planning grid.
 sigma=float(np.sqrt((D*(y-cy)**2).sum()/max(total,1e-20))*.05)
 u=np.linspace(-.3,1.3,41);profile=np.interp(2.5+u*length,np.arange(D.shape[1])*.05,D.sum(axis=0),left=0,right=0)
 return dict(family=f,length=length,sigma_mm=sigma,integrated_density_mm2=float(total*.0025),centroid_mm=[cx*.05,cy*.05],axial_u=u.tolist(),axial_weights=(profile/max(profile.sum(),1e-20)).tolist(),concentration=c,seconds=r['seconds'])
def main():
 from .meniscus import native_film
 from .paper_transport import native_paper
 from .technique_paper import entry_native
 native_film();native_paper();entry_native(.04,.11)
 out=ROOT/'driver19';out.mkdir(exist_ok=True);rows=[]
 with ProcessPoolExecutor(max_workers=6) as pool:
  jobs=[pool.submit(probe,f,l) for f in CONTROLS for l in [1.5,3.5]]
  for q in as_completed(jobs):
   rows.append(q.result());(out/'probe-measurements.json').write_text(json.dumps(rows,indent=2))
 profiles={f:[r for r in rows if r['family']==f] for f in CONTROLS}
 (out/'proxy-profiles.json').write_text(json.dumps(profiles,indent=2));print('PROBES COMPLETE',flush=True)
if __name__=='__main__':main()
