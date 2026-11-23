"""Depth-resolved water transport and particle retention in porous paper.

This is an uncalibrated finite-volume model, not a MoXi implementation. The
substrate experiment deliberately leaves the v0.12 brush mechanics unchanged.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import ctypes, math, os, shutil, subprocess, hashlib, tempfile
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class PaperSettings:
    layer_depths_m: tuple = (6e-6, 12e-6, 22e-6, 40e-6)
    porosity: float = .60
    inplane_permeability_m2: float = 2e-13
    depth_permeability_m2: float = 7e-16
    retention_lambda: float = 2.
    heterogeneity: float = .55
    anisotropy: float = .65
    seed: int = 70713
    texture_model: str = 'correlated'
    fine_fraction: float = .75
    fine_filtration_length_m: float = .0012
    coarse_filtration_length_m: float = .00018
    fine_adsorption_s: float = .20
    coarse_adsorption_s: float = 1.50
    max_dt_s: float = .004
    surface_tick_s: float = .006

    def __post_init__(self):
        if self.texture_model not in ('correlated','fibers'):raise ValueError('unknown paper texture model')
        if len(self.layer_depths_m)<2 or any(not math.isfinite(v) or v<=0 for v in self.layer_depths_m):
            raise ValueError('paper needs at least two positive layer depths')
        if not 0<self.porosity<1 or not 0<=self.anisotropy<1 or not 0<=self.fine_fraction<=1:
            raise ValueError('invalid porosity, anisotropy or particle fraction')
        for name in ['inplane_permeability_m2','depth_permeability_m2','retention_lambda',
                     'fine_filtration_length_m','coarse_filtration_length_m','max_dt_s','surface_tick_s']:
            v=getattr(self,name)
            if not math.isfinite(v) or v<=0:raise ValueError(name+' must be positive')
        for name in ['heterogeneity','fine_adsorption_s','coarse_adsorption_s']:
            v=getattr(self,name)
            if not math.isfinite(v) or v<0:raise ValueError(name+' must be nonnegative')


def native_paper():
    source=Path(__file__).with_suffix('.cpp')
    lib=source.with_name('_paper_transport_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]+'.so')
    if not lib.exists():
        compiler=shutil.which(os.environ.get('CXX','g++'))
        if compiler is None:raise RuntimeError('Paper transport needs a C++17 compiler (g++ or CXX).')
        descriptor,temp_name=tempfile.mkstemp(prefix=lib.stem+'-',suffix='.so',dir=source.parent)
        os.close(descriptor);temporary=Path(temp_name)
        subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(temporary)],check=True)
        os.replace(temporary,lib)
    fn=ctypes.CDLL(str(lib)).paper_step
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[arr]*11+[ctypes.c_int]*7+[ctypes.c_double]*13+[arr]
    fn.restype=ctypes.c_int
    return fn


class PaperTransport:
    def __init__(self,shape,fluid,settings=None):
        self.settings=settings or PaperSettings();p=self.settings;self.fluid=fluid
        self.shape=tuple(shape);H,W=shape;self.depths=np.asarray(p.layer_depths_m,float)
        self.layers=len(self.depths);self.native=native_paper()
        rng=np.random.default_rng(p.seed)
        def field(sigma):
            a=gaussian_filter(rng.normal(size=shape),sigma)
            return a/max(a.std(),1e-20)
        # Fixed substrate fields, generated in paper coordinates only.
        if p.texture_model=='fibers':
            # Random curved cellulose-like paths in physical paper coordinates.
            # These are substrate geometry, not lines added to rendered ink.
            density=np.zeros(shape);ct=np.zeros(shape);st=np.zeros(shape)
            scale=50e-6/fluid.pixel_m
            count=int(H*W/scale**2*.035)
            for _ in range(count):
                center=rng.uniform([0,0],[W,H]);angle=rng.uniform(-math.pi,math.pi)
                length=np.clip(rng.lognormal(math.log(18),.45),6,50)*scale
                width=rng.uniform(.25,.55)*scale;bend=rng.normal(0,.10)
                u=np.linspace(-.5,.5,max(4,int(length*2)))
                phase=angle+bend*u*3;ds=length/(len(u)-1)
                points=np.c_[np.cos(phase),np.sin(phase)]*ds
                points=np.cumsum(points,axis=0);points+=center-points[len(points)//2]
                for (xx,yy),theta in zip(points,phase):
                    x0=max(0,int(xx-2*width-1));x1=min(W,int(xx+2*width+2))
                    y0=max(0,int(yy-2*width-1));y1=min(H,int(yy+2*width+2))
                    if x1<=x0 or y1<=y0:continue
                    gy,gx=np.mgrid[y0:y1,x0:x1]
                    w=np.exp(-((gx-xx)**2+(gy-yy)**2)/(2*width**2))*ds
                    density[y0:y1,x0:x1]+=w
                    ct[y0:y1,x0:x1]+=w*math.cos(2*theta);st[y0:y1,x0:x1]+=w*math.sin(2*theta)
            density=gaussian_filter(density,.35*scale)
            ct=gaussian_filter(ct,.9*scale);st=gaussian_filter(st,.9*scale)
            direction=.5*np.arctan2(st,ct)
            local=np.clip((density-density.mean())/max(density.std(),1e-20),-2.5,3.0)
            self.porosity=np.clip(p.porosity-.055*p.heterogeneity*local,.36,.78)
            self.kmap=p.inplane_permeability_m2*np.exp(-p.heterogeneity*local)
            self.kz=p.depth_permeability_m2*np.exp(-.5*p.heterogeneity*local)
        else:
            direction=.7*field(10.)
            local=.60*field((.6,3.5))+.40*field((3.2,.8))+.25*field(7.)
            local=np.clip(local,-3,3)
            self.porosity=np.clip(p.porosity+.045*p.heterogeneity*local,.42,.76)
            self.kmap=p.inplane_permeability_m2*np.exp(p.heterogeneity*local)
            self.kz=p.depth_permeability_m2*np.exp(.5*p.heterogeneity*local)
        self.capacity=np.ascontiguousarray(self.depths[:,None,None]*self.porosity[None,:,:]*fluid.pixel_m**2)
        self.water=np.zeros_like(self.capacity);self.fine=np.zeros_like(self.water)
        self.coarse=np.zeros_like(self.water);self.fixed=np.zeros_like(self.water)
        self.film=np.zeros(shape);self.film_mass=np.zeros(shape)
        # Dimensionless fiber density plus an immutable top-depth array.
        self.fiber=np.ascontiguousarray(np.stack([(1-self.porosity)/(1-p.porosity),np.full(shape,self.depths[0])]))
        self.suction=2*fluid.surface_tension_n_m*math.cos(.10)/fluid.paper_pore_radius_m
        base=self.suction/(fluid.viscosity_pa_s*(3*p.retention_lambda+1))
        self.conduct=np.zeros((self.layers,4,H,W))
        # Monotone 9-point stencil. Isotropic weights sum to the standard
        # continuum diffusivity; local anisotropy modifies material conductance.
        for d,(dy,dx,angle,weight) in enumerate([(0,1,0,2/3),(1,0,math.pi/2,2/3),(1,1,math.pi/4,1/6),(1,-1,-math.pi/4,1/6)]):
            directional=self.kmap*(1+p.anisotropy*np.cos(2*(direction-angle)))
            adjacent=np.roll(directional,(-dy,-dx),(0,1))
            face=2*directional*adjacent/(directional+adjacent)
            self.conduct[:,d]=base*self.depths[:,None,None]*weight*face[None,:,:]
        self.vertical=np.ascontiguousarray(base*fluid.pixel_m**2*self.kz[None,:,:]/(.5*(self.depths[:-1]+self.depths[1:]))[:,None,None])
        self.steps=0;self.absorbed_m3=0.;self.moved_m3=0.;self.time=0.

    def advance(self,dt):
        active=(self.water.sum(axis=0)>1e-24)|(self.film>0)
        yy,xx=np.nonzero(active)
        if not len(yy):self.time+=dt;return 0.
        # Each call expands beyond current support; there are no periodic edges.
        y0=max(0,int(yy.min())-8);y1=min(self.shape[0],int(yy.max())+9)
        x0=max(0,int(xx.min())-8);x1=min(self.shape[1],int(xx.max())+9)
        p=self.settings;f=self.fluid;out=np.zeros(3)
        code=self.native(self.film,self.film_mass,self.water,self.fine,self.coarse,self.fixed,
            self.capacity,self.conduct,self.vertical,self.kz,self.fiber,self.layers,*self.shape,y0,y1,x0,x1,
            dt,f.pixel_m,f.viscosity_pa_s,self.suction,p.porosity,2e-6,3+1/p.retention_lambda,
            p.fine_fraction,p.fine_filtration_length_m,p.coarse_filtration_length_m,
            p.fine_adsorption_s,p.coarse_adsorption_s,p.max_dt_s,out)
        if code:raise RuntimeError('porous paper exceeded adaptive step budget')
        self.steps+=int(out[0]);self.absorbed_m3+=out[1];self.moved_m3+=out[2];self.time+=dt
        return float(out[1])

    @property
    def pigment(self):return self.fine+self.coarse+self.fixed

    def density(self):
        return (self.pigment.sum(axis=0)+self.film_mass)*self.fluid.extinction_m2_kg/self.fluid.pixel_m**2

    def totals(self):
        return {'water_m3':float(self.water.sum()+self.film.sum()*self.fluid.pixel_m**2),
            'pigment_kg':float(self.pigment.sum()+self.film_mass.sum()),
            'film_nl':float(self.film.sum()*self.fluid.pixel_m**2*1e12),
            'layer_water_nl':(self.water.sum(axis=(1,2))*1e12).tolist(),
            'mobile_pigment_ng':float((self.fine.sum()+self.coarse.sum())*1e12),
            'fixed_pigment_ng':float(self.fixed.sum()*1e12),'steps':self.steps}

    def check(self):
        for a in [self.water,self.fine,self.coarse,self.fixed,self.film,self.film_mass]:
            if not np.isfinite(a).all() or a.min() < -1e-24:raise FloatingPointError('invalid water or pigment inventory')
        if np.max(self.water/self.capacity)>1+1e-8:raise FloatingPointError('paper capacity exceeded')

    def save(self,path):
        np.savez_compressed(path,water=self.water,fine=self.fine,coarse=self.coarse,fixed=self.fixed,
            film=self.film,film_mass=self.film_mass,capacity=self.capacity,permeability=self.kmap,density=self.density())
