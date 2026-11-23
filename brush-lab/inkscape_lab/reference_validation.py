"""Conservation guards for blocked contact and recorded reference probes."""
from pathlib import Path
import json
import numpy as np
from .brush import InkCanvas
from .cun import ContactSettings
from .reference_brush import ReferenceBrush,SurfaceContactSettings
from .paper_transport import PaperSettings

def contact_probe():
    b=ReferenceBrush(InkCanvas(80,80),ContactSettings(width=12,bunches=12),
        paper=PaperSettings(heterogeneity=0,anisotropy=0),
        surface_contact=SurfaceContactSettings(macro_rms_m=0,micro_rms_m=0))
    rows=[];dv=1e-13;dm=dv*28
    for case in ['open','blocked','mixed']:
        b.film_h[:]=0;b.film_mass[:]=0;b.neck_v[:]=0;b.neck_m[:]=0;b.contact_height[:]=0
        b.gap[0]=0
        if case=='blocked':b.gap[0]=2*b.coat()[0]
        if case=='mixed':b.contact_height[:,:40]=-90e-6
        # Emulate the upstream transfer: debit from a live neck, then deposit.
        b.neck_v[0]=2*dv;b.neck_m[0]=2*dm
        b.neck_v[0]-=dv;b.neck_m[0]-=dm
        b._deposit(0,dv,dm,np.array([44.,40.]),1.7,np.array([34.,40.]))
        v=b.neck_v.sum()+b.film_h.sum()*b.fluid.pixel_m**2
        m=b.neck_m.sum()+b.film_mass.sum()
        deposited=b.film_h.sum()*b.fluid.pixel_m**2/dv
        assert abs(v/(2*dv)-1)<1e-12 and abs(m/(2*dm)-1)<1e-12
        if case=='open':assert deposited>1-1e-12
        if case=='blocked':assert deposited==0
        if case=='mixed':assert 0<deposited<1
        rows.append({'case':case,'deposited_fraction':float(deposited),'water_relative_error':float(v/(2*dv)-1),'pigment_relative_error':float(m/(2*dm)-1)})
    # Independent constant-section limit of the quadrature used for taper.
    u,w=np.polynomial.legendre.leggauss(32);u=(u+1)/2;w=w/2
    L=.0038;E=2.4e9;r=20e-6;I=np.pi*r**4/4
    k=1/(L**3*np.sum(w*(1-u)**2/(E*I)))
    expected=3*E*I/L**3
    assert abs(k/expected-1)<1e-12
    return {'contact_cases':rows,'constant_section_stiffness_relative_error':float(k/expected-1)}

def main():
    root=Path(__file__).resolve().parents[1];out=root/'reference-review';out.mkdir(exist_ok=True)
    d={'closure':contact_probe(),'runs':{},'r0_r1_source_relative_l1':{}}
    for r in range(6):
        for kind in ['wet','dry']:
            p=root/'reference-rounds'/f'r{r}'/kind
            if not (p/'run.json').exists():continue
            run=json.loads((p/'run.json').read_text());a=np.load(p/'fields.npz')
            for key in ['water','fine','coarse','fixed','film','film_mass']:
                assert np.isfinite(a[key]).all() and np.min(a[key])>=-1e-24
            assert np.max(a['water']/a['capacity'])<=1+1e-8
            source=np.load(p/'source.npz')
            sv=source['water'].sum();sm=source['pigment'].sum()
            pv=a['water'].sum()+a['film'].sum()*run['fluid']['pixel_m']**2
            pm=a['fine'].sum()+a['coarse'].sum()+a['fixed'].sum()+a['film_mass'].sum()
            balance={'source_water_nl':float(sv*1e12),'source_pigment_ng':float(sm*1e12),
                'paper_water_relative_error':float(pv/sv-1),'paper_pigment_relative_error':float(pm/sm-1),
                'whole_brush_water_relative_error':run['diagnostics']['water_relative_error'],
                'whole_brush_pigment_relative_error':run['diagnostics']['pigment_relative_error']}
            assert max(abs(balance[k]) for k in balance if 'error' in k)<1e-9
            d['runs'][f'r{r}/{kind}']=balance
    for kind in ['wet','dry']:
        a=np.load(root/'reference-rounds/r0'/kind/'source.npz');b=np.load(root/'reference-rounds/r1'/kind/'source.npz')
        assert np.array_equal(a['dt'],b['dt']) and np.array_equal(a['ids'],b['ids'])
        d['r0_r1_source_relative_l1'][kind]={k:float(abs(a[k]-b[k]).sum()/abs(a[k]).sum()) for k in ['water','pigment']}
    d['scope']='Finite/nonnegative states, paper capacity, flux closure and inventory conservation. Not physical calibration and not a full timestep/spatial convergence study.'
    (out/'validation.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))

if __name__=='__main__':main()
