"""Reservoir-fed imbibition scaling and saved-field refinement checks."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .paper_transport import PaperTransport,PaperSettings
from .meniscus import material

def reservoir_probe():
    f=material();p=PaperSettings(heterogeneity=0,anisotropy=0,max_dt_s=.002)
    b=PaperTransport((3,180),f,p);t=0.;added=0.;rows=[]
    for goal in [.08,.16,.32,.64,1.28,2.56]:
        while t<goal-1e-12:
            incoming=b.capacity[:,:,0]-b.water[:,:,0]
            added+=float(incoming.sum());b.water[:,:,0]=b.capacity[:,:,0]
            step=min(.002,goal-t);b.advance(step);t+=step
        # Integral water-equivalent wetting length; avoids arbitrary front cutoff.
        length=b.water.sum()/(p.porosity*sum(p.layer_depths_m)*3*f.pixel_m)
        rows.append({'t':t,'equivalent_wet_length_mm':float(length*1e3)})
    ts=np.array([r['t'] for r in rows]);length=np.array([r['equivalent_wet_length_mm'] for r in rows])
    coeff=np.polyfit(ts,length**2,1);pred=np.polyval(coeff,ts)
    r2=1-float(((pred-length**2)**2).sum()/((length**2-(length**2).mean())**2).sum())
    b.check();error=float(b.water.sum()/added-1)
    assert abs(error)<1e-10
    return {'trace':rows,'length_squared_fit_r2':r2,'length_squared_slope_mm2_s':float(coeff[0]),'water_relative_error':error}

def main():
    out=Path('paper-validation');out.mkdir(exist_ok=True)
    d={'reservoir':reservoir_probe()}
    for a,b,label in [('paper-round1','paper-fine-step','paper_time'),('paper-uniform','paper-uniform-fine','paper_space'),('paper-drop-final','paper-drop-fine','coupled_drop_time_correlated'),('paper-drop-fibers','paper-drop-fibers-fine','coupled_drop_time_strong_fibers'),('paper-drop-v013','paper-drop-v013-fine','coupled_drop_time')]:
        x=np.load(Path(a)/'fields.npz')['density'];y=np.load(Path(b)/'fields.npz')['density']
        if label=='paper_space':y=y.reshape(x.shape[0],2,x.shape[1],2).mean((1,3))
        d[label]={'density_relative_l1':float(abs(x-y).sum()/x.sum()),'coarse_case':a,'fine_case':b}
    d['scope']='Paper-only timestep check uses the first correlated substrate; spatial check uses uniform paper; final coupled timestep check uses the random-fiber droplet. No full brush/nonuniform-space convergence claim.'
    (out/'validation.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))

if __name__=='__main__':main()
