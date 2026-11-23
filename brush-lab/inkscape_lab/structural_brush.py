"""Conservative structural alternatives and v0.17 decoupled contact geometry.

These are explicitly competing reduced closures, not calibrated models.
No image filtering, random ink masks, or reference pixels enter the solver.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter
from .technique_controls import TechniqueBrush,TechniqueSettings
from .meniscus import MeniscusBrush

@dataclass(frozen=True)
class Structure:
    geometry: str='laid'          # laid segment / rising distal arc
    flux: str='area'              # accessible area / series resistance
    supply: str='finite'          # instantaneous fill / finite Poiseuille feed
    release: str='drain'          # cap stamp / remembered patch split / drainage only
    relief: str='independent'     # independent height / paper-fiber density
    relief_um: float=20.
    correlation_mm: float=.15
    supply_fraction: float=.22
    arc_rise_scale: float=1.
    contact_transition_um: float=4.
    memory_seconds: float=0.      # optional causal footprint history, never future pixels
    neck_law: str='tube'          # h^4 equivalent tube / h^3 thin-film scaling
    axis_blend: float=0.          # rotate laid contact toward handle projection, keeping length
    contact_rise: float=0.        # independent quadratic contact arc; preserves tuft elevation
    fiber_coupling: float=0.      # shared fraction of paper-height/density latent field

class StructuralBrush(TechniqueBrush):
    version='sequential-structure-study-0.17.0'
    def __init__(self,canvas,*,structure=None,**kwargs):
        self.structure=Structure(**(structure or {}))
        s=self.structure
        if s.geometry not in ('laid','arc') or s.flux not in ('area','series') or s.supply not in ('instant','finite') or s.release not in ('cap','memory','drain') or s.relief not in ('independent','shared') or s.neck_law not in ('tube','film'):
            raise ValueError('unknown structural closure')
        if min(s.relief_um,s.correlation_mm,s.supply_fraction,s.contact_transition_um)<=0 or s.memory_seconds<0:
            raise ValueError('invalid structural parameter')
        if not 0<=s.axis_blend<=1 or not 0<=s.contact_rise<=1 or not 0<=s.fiber_coupling<=1:
            raise ValueError('invalid decoupled contact geometry')
        settings=kwargs.pop('technique',{})
        settings=dict(settings,contact_model='legacy' if s.geometry=='laid' else 'resistance',
                      finite_neck_supply=True,supply_hydraulic_fraction=s.supply_fraction)
        surface=dict(kwargs.pop('surface_contact_overrides',{}),transition_m=s.contact_transition_um*1e-6)
        super().__init__(canvas,technique=TechniqueSettings(**settings),surface_contact_overrides=surface,**kwargs)
        self.record_source=False
        n=self.settings.bunches
        self.patches=[None for _ in range(n)]
        self.patch_history=[[] for _ in range(n)]
        self.release_source=np.zeros_like(self.canvas.density)
        self.contact_source=np.zeros_like(self.canvas.density)
        self.detached_source=np.zeros_like(self.canvas.density)
        self.release_to_paper=0.;self.release_to_hair=0.;self.release_no_patch=0
        self.last_area=np.zeros(n)
        seed=self.surface_contact.seed
        if s.relief=='shared':
            field=self.substrate.fiber[0].copy();field-=field.mean()
            field/=max(field.std(),1e-15)
        else:
            rng=np.random.default_rng(seed)
            raw=rng.normal(size=self.canvas.density.shape)
            field=gaussian_filter(raw,s.correlation_mm*1e-3/self.fluid.pixel_m)
            field/=max(field.std(),1e-15)
            fine=gaussian_filter(raw,.65*50e-6/self.fluid.pixel_m)
            field+=.4*fine/max(fine.std(),1e-15)
            field-=field.mean();field/=max(field.std(),1e-15)
        if s.relief=='independent' and s.fiber_coupling>0:
            fiber=self.substrate.fiber[0].copy();fiber-=fiber.mean();fiber/=max(fiber.std(),1e-15)
            field=(1-s.fiber_coupling)*field+s.fiber_coupling*fiber
            field-=field.mean();field/=max(field.std(),1e-15)
        self.contact_height=field*s.relief_um*1e-6
        self.height=self.contact_height
        self.slope_y,self.slope_x=np.gradient(self.height,self.fluid.pixel_m)

    def _transfer(self,dt,gate,side):
        from .structural_transfer import structural_transfer
        self.transfer_dt=dt
        return structural_transfer(self,dt,gate,side)

    def _neck_conductance(self,j,coat,length):
        mc=self.meniscus;f=self.fluid
        ratio=np.clip(coat/(mc.max_coat_m*.5),0,1.6)
        if self.structure.neck_law=='tube':
            return math.pi*(mc.neck_radius_m*ratio)**4/(8*f.viscosity_pa_s*length)
        # Alternative h^3 film scaling, matched at h_ref. This tests a
        # constitutive exponent; it is not a resolved annular-flow solver.
        return math.pi*mc.neck_radius_m**4*ratio**3/(8*f.viscosity_pa_s*length)

    def _put(self,dv,dm,patch,channel):
        bounds,weight=patch;y0,y1,x0,x1=bounds
        self.film_h[y0:y1,x0:x1]+=dv*weight/self.fluid.pixel_m**2
        self.film_mass[y0:y1,x0:x1]+=dm*weight
        channel[y0:y1,x0:x1]+=dm*weight
        self._extend(bounds)

    def _put_memory(self,j,dv,dm,channel):
        if self.patches[j] is None:return False
        history=self.patch_history[j]
        if self.structure.memory_seconds>0 and history:
            # Occupancy-weighted causal footprint history. All support was
            # actually wetted by this connected hair, in fixed paper coordinates.
            weights=np.array([t[1] for t in history]);weights/=weights.sum()
            for (_,_,patch),w in zip(history,weights):self._put(dv*w,dm*w,patch,channel)
        else:self._put(dv,dm,self.patches[j],channel)
        return True

    def _release(self,j):
        n=self.settings.bunches;mc=self.meniscus;s=self.structure
        v=self.neck_v[j];m=self.neck_m[j]
        fraction=math.cos(mc.paper_angle_rad)/(math.cos(mc.paper_angle_rad)+math.cos(mc.contact_angle_rad))
        if s.release=='cap':
            radius=np.cbrt(4*v*fraction/(math.pi*mc.paper_angle_rad))/self.fluid.pixel_m
            before=self.film_mass.copy()
            MeniscusBrush._deposit(self,j,v*fraction,m*fraction,self.anchor[j],max(.4,radius),cap=True)
            self.release_source+=self.film_mass-before
        elif s.release=='memory':
            if not self._put_memory(j,v*fraction,m*fraction,self.release_source):
                fraction=0.;self.release_no_patch+=1
        else:
            # Drainage limiting closure: paper has received resolved flow while
            # connected. Untransferred residual stays on hair; no invented drop.
            fraction=0.
        self.volume[n+j]+=v*(1-fraction);self.mass[n+j]+=m*(1-fraction)
        self.release_to_paper+=v*fraction;self.release_to_hair+=v*(1-fraction)
        self.neck_v[j]=self.neck_m[j]=0.;self.neck_alive[j]=False;self.ruptures+=1
        self.patches[j]=None;self.patch_history[j]=[]

    def _deposit(self,j,dv,dm,center,radius_px,heel=None,cap=False):
        if dv<=0:return
        if cap:raise RuntimeError('structural rupture must use _release')
        f=self.fluid;mc=self.meniscus;p=self.surface_contact;s=self.structure
        coat=self.coat()[j]
        if self.gap[j]>=coat and s.release!='cap':
            # The paper end of a still-live bridge remains on its existing
            # wetted footprint. It does not move with the airborne hair.
            if not self._put_memory(j,dv,dm,self.detached_source):
                self.neck_v[j]+=dv;self.neck_m[j]+=dm
            return
        x,y=center;heel=center if heel is None else heel
        if s.geometry=='laid':
            segment=heel-center+p.bend_contact_fraction*self.local_bending_xy[j]
            if s.axis_blend>0:
                laid=float(np.linalg.norm(heel-center))
                if laid>1e-12:
                    direction=(1-s.axis_blend)*(heel-center)/laid+s.axis_blend*np.array([math.cos(self.angle),math.sin(self.angle)])
                    segment=laid*direction/max(np.linalg.norm(direction),1e-12)+p.bend_contact_fraction*self.local_bending_xy[j]
        else:
            laid=np.linalg.norm(center-heel)
            segment=laid*np.array([math.cos(self.angle),math.sin(self.angle)])+p.bend_contact_fraction*self.local_bending_xy[j]
        length=float(np.linalg.norm(segment));den=length*length
        bounds=self._bounds(x,y,max(radius_px,.45)+length+1)
        y0,y1,x0,x1=bounds
        if x1<=x0 or y1<=y0:raise ValueError('contact outside canvas')
        yy,xx=np.mgrid[y0:y1,x0:x1];weight=np.zeros_like(xx,dtype=float);nominal=np.zeros_like(weight)
        center_h=float(map_coordinates(self.contact_height,[[y],[x]],order=1,mode='nearest')[0])
        rise=length*f.pixel_m*max(0,math.cos(self.tilt))*(s.arc_rise_scale if s.geometry=='arc' else s.contact_rise)
        localh=self.contact_height[y0:y1,x0:x1]
        reach=coat+self.film_h[y0:y1,x0:x1]
        for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
            u=np.clip(((xx+ox-x)*segment[0]+(yy+oy-y)*segment[1])/max(den,1e-12),0,1)
            distance=np.hypot(xx+ox-x-u*segment[0],yy+oy-y-u*segment[1])
            gap=np.maximum(0,center_h+self.gap[j]+rise*u*u-localh)
            contact=np.clip((reach-gap)/p.transition_m,0,1)
            width=radius_px if s.geometry=='laid' and s.contact_rise==0 else np.sqrt(np.maximum(0,(self.radius[j]+reach)**2-(self.radius[j]+gap)**2))/f.pixel_m
            weight+=np.clip((width+.35-distance)/.7,0,1)*contact*.25
            nominal+=np.clip((radius_px+.35-distance)/.7,0,1)*.25
        area=float(weight.sum())*f.pixel_m**2
        if s.flux=='area':fraction=float(np.clip(weight.sum()/max(nominal.sum(),1e-30),0,1))
        else:
            gn=self._neck_conductance(j,coat,mc.neck_length_m+self.gap[j]+2e-6)
            gc=mc.contact_permeability_m2*area/(f.viscosity_pa_s*mc.neck_length_m)
            fraction=gc/(gc+gn) if area>0 else 0.
        self.neck_v[j]+=dv*(1-fraction);self.neck_m[j]+=dm*(1-fraction)
        self.blocked_water+=dv*(1-fraction);self.blocked_mass+=dm*(1-fraction);self.contact_deposits+=1
        if area<=0 or fraction<=0:return
        weight/=weight.sum();patch=(bounds,weight)
        self.patches[j]=patch;self.last_area[j]=area
        if s.memory_seconds>0:
            h=self.patch_history[j];h.append((self.time,self.transfer_dt,patch))
            while h and h[0][0]<self.time-s.memory_seconds:h.pop(0)
        self._put(dv*fraction,dm*fraction,patch,self.contact_source)

    def diagnostics(self):
        d=super().diagnostics()
        if hasattr(self,'release_source'):
            d.update(structure=asdict(self.structure),release_paper_nl=self.release_to_paper*1e12,
                     release_hair_nl=self.release_to_hair*1e12,release_without_patch=self.release_no_patch,
                     release_pigment_ng=float(self.release_source.sum()*1e12),
                     contact_pigment_ng=float(self.contact_source.sum()*1e12),
                     detached_drain_pigment_ng=float(self.detached_source.sum()*1e12))
        return d
