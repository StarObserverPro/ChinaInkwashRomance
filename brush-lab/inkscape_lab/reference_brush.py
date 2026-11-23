"""Reference-guided v0.14 experiments, preserving v0.13 as a control.

The added closure resolves liquid contact along the laid hair segment against
fixed paper relief. Rejected flux stays in that hair's liquid neck. Reference
images never enter this solver. This is not a calibrated fiber-scale model.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from .absorbent_brush import AbsorbentBrush


@dataclass(frozen=True)
class SurfaceContactSettings:
    enabled: bool = True
    macro_rms_m: float = 20e-6
    micro_rms_m: float = 8e-6
    transition_m: float = 4e-6
    bend_contact_fraction: float = 0.
    shaft_radius_ratio: float = 1.
    seed: int = 14009


class ReferenceBrush(AbsorbentBrush):
    version='reference-guided-paper-contact-0.14.0'

    def __init__(self,*args,surface_contact=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.surface_contact=surface_contact or SurfaceContactSettings()
        p=self.surface_contact
        if min(p.macro_rms_m,p.micro_rms_m)<0 or p.transition_m<=0 or not 0<=p.bend_contact_fraction<=1 or p.shaft_radius_ratio<1:
            raise ValueError('invalid contact relief')
        rng=np.random.default_rng(p.seed)
        field=rng.normal(size=self.film_h.shape)
        scale=50e-6/self.fluid.pixel_m
        coarse=gaussian_filter(field,3.*scale);coarse/=max(coarse.std(),1e-15)
        fine=gaussian_filter(field,.65*scale);fine/=max(fine.std(),1e-15)
        self.contact_height=coarse*p.macro_rms_m+fine*p.micro_rms_m
        self.blocked_water=0.;self.blocked_mass=0.;self.contact_deposits=0
        if p.shaft_radius_ratio!=1:
            # Tip-load compliance of a linearly tapered Euler-Bernoulli beam:
            # 1/k = integral_0^L (L-s)^2 / (E I(s)) ds. This retains only an
            # endpoint degree of freedom, not a full discrete elastic rod.
            nodes,weights=np.polynomial.legendre.leggauss(32)
            u=(nodes+1)*.5;w=weights*.5
            radius=self.radius[:,None]*(p.shaft_radius_ratio+(1-p.shaft_radius_ratio)*u)
            inertia=np.pi*radius**4/4
            compliance=self.length**3*np.sum(w*(1-u)**2/(self.meniscus.young_pa*inertia),axis=1)
            self.stiffness=1/compliance
        if p.enabled:
            self.height=self.contact_height
            self.slope_y,self.slope_x=np.gradient(self.height,self.fluid.pixel_m)

    def _network(self,desired,dt):
        super()._network(desired,dt)
        self.local_bending_xy=desired-self.centers

    def _deposit(self,j,dv,dm,center,radius_px,heel=None,cap=False):
        if cap or not self.surface_contact.enabled:
            return super()._deposit(j,dv,dm,center,radius_px,heel,cap)
        if dv<=0:return
        f=self.fluid;p=self.surface_contact;x,y=center
        heel=center if heel is None else heel
        # A laid contact follows its own elastic deflection toward the free
        # position, rather than assigning every hair the same global heel.
        # This fractional contact-length closure is uncalibrated; the bending
        # vector itself comes from the mechanical solve, not random stroke noise.
        if p.bend_contact_fraction and hasattr(self,'local_bending_xy'):
            heel=heel+p.bend_contact_fraction*self.local_bending_xy[j]
        pad=max(radius_px,.45)+np.linalg.norm(heel-center)
        y0,y1,x0,x1=self._bounds(x,y,pad+1)
        if x1<=x0 or y1<=y0:raise ValueError('contact left canvas')
        yy,xx=np.mgrid[y0:y1,x0:x1];weight=np.zeros_like(xx,dtype=float)
        segment=heel-center;den=float(segment@segment)
        # Same anti-aliased geometric footprint as v0.13, before material contact.
        for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
            u=np.clip(((xx+ox-x)*segment[0]+(yy+oy-y)*segment[1])/max(den,1e-12),0,1)
            distance=np.hypot(xx+ox-x-u*segment[0],yy+oy-y-u*segment[1])
            weight+=np.clip((radius_px+.35-distance)/.7,0,1)/4
        nominal=float(weight.sum())
        # The solid tip gap was calculated at the tip, not over its full laid
        # contact. Compare the same hair plane with local paper height here.
        center_h=float(map_coordinates(self.contact_height,[[y],[x]],order=1,mode='nearest')[0])
        local_gap=np.maximum(0.,center_h+self.gap[j]-self.contact_height[y0:y1,x0:x1])
        liquid_reach=self.coat()[j]+self.film_h[y0:y1,x0:x1]
        contact=np.clip((liquid_reach-local_gap)/p.transition_m,0,1)
        connected=weight*contact
        fraction=float(np.clip(connected.sum()/max(nominal,1e-30),0,1))
        # Upstream transfer has debited dv/dm; restore the part that cannot
        # access paper to the SAME live neck, not to the painted image.
        self.neck_v[j]+=dv*(1-fraction);self.neck_m[j]+=dm*(1-fraction)
        self.blocked_water+=dv*(1-fraction);self.blocked_mass+=dm*(1-fraction)
        self.contact_deposits+=1
        if fraction<=0:return
        connected/=connected.sum()
        self.film_h[y0:y1,x0:x1]+=dv*fraction*connected/f.pixel_m**2
        self.film_mass[y0:y1,x0:x1]+=dm*fraction*connected
        self._extend((y0,y1,x0,x1))

    def diagnostics(self):
        d=super().diagnostics()
        if hasattr(self,'surface_contact'):
            d['blocked_flux_integral_nl']=self.blocked_water*1e12
            d['blocked_flux_note']='Cumulative rejected attempts; same liquid may be counted repeatedly. Not a separate inventory.'
            d['surface_contact_deposits']=self.contact_deposits
        return d
