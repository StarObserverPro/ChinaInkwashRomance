"""SI controls and persistent material state for the v0.15 studies.

The public motion uses millimetres, seconds and degrees. Compression is a
displacement command, not a force in newtons. Contact/ink are never painted
from a technique label. This remains an uncalibrated reduced brush model.
"""
from dataclasses import dataclass,asdict,replace
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from .reference_brush import ReferenceBrush,SurfaceContactSettings
from .paper_transport import PaperSettings
from .meniscus import MeniscusSettings,material
from .cun import ContactSettings
from .response import _follow_linear

@dataclass(frozen=True)
class InkMaterial:
    viscosity_pa_s: float=.003
    surface_tension_n_m: float=.055
    fine_fraction: float=.65
    extinction_m2_kg: float=11000.

@dataclass(frozen=True)
class InkPaperInteraction:
    fine_filtration_length_m: float=.0012
    coarse_filtration_length_m: float=.00016
    fine_adsorption_s: float=.20
    coarse_adsorption_s: float=1.50
    entry_capture_fine: float=0.
    entry_capture_coarse: float=0.

@dataclass(frozen=True)
class BrushLoading:
    body_saturation: float=.40
    tip_saturation: float=.40
    body_concentration_kg_m3: float=24.
    tip_concentration_kg_m3: float=24.
    side_contrast: float=0.
    side_axis_deg: float=0.

@dataclass(frozen=True)
class TechniqueSettings:
    # R0 preserves the v0.14 paper/contact closures under the new controller.
    contact_model: str='legacy'
    evaporation_m_s: float=0.
    relative_humidity: float=.50
    max_step_s: float=.0015
    spatial_step_mm: float=.035
    fiber_surface_rms_m: float=9e-6
    pixel_size_um: float=50.
    finite_neck_supply: bool=False
    supply_hydraulic_fraction: float=.22

def material_settings(ink,interaction,paper_overrides=None):
    """Move ink composition and ink/paper interactions out of the paper UI.

    The old solver adapter still expects these coefficients in PaperSettings.
    """
    ink=ink or InkMaterial();interaction=interaction or InkPaperInteraction()
    if not all(np.isfinite(v) for v in [*asdict(ink).values(),*asdict(interaction).values()]):raise ValueError('nonfinite material parameter')
    if ink.viscosity_pa_s<=0 or ink.surface_tension_n_m<=0 or not 0<=ink.fine_fraction<=1:
        raise ValueError('invalid ink material')
    if ink.extinction_m2_kg<=0:raise ValueError('invalid optical coefficient')
    fluid=replace(material(),viscosity_pa_s=ink.viscosity_pa_s,
        surface_tension_n_m=ink.surface_tension_n_m,extinction_m2_kg=ink.extinction_m2_kg)
    bulk={k:v for k,v in asdict(interaction).items() if not k.startswith('entry_capture')}
    if not (0<=interaction.entry_capture_fine<=1 and 0<=interaction.entry_capture_coarse<=1):raise ValueError('invalid entry capture')
    paper=replace(PaperSettings(),texture_model='fibers',heterogeneity=.25,anisotropy=.35,
        inplane_permeability_m2=1e-13,**bulk,fine_fraction=ink.fine_fraction)
    if paper_overrides:paper=replace(paper,**paper_overrides)
    return fluid,paper

def compile_gesture(spec,settings=None):
    """Explicit keyframes; position segments are linear, pose is PCHIP.

    Repeated positions implement a dwell. Speed changes are changes in the
    physical clock, never changes in point density. Negative compression lifts.
    """
    settings=settings or TechniqueSettings();points=spec['keyframes']
    keys=['t_s','x_mm','y_mm','compression_mm','tilt_deg','azimuth_deg','roll_deg']
    data=np.array([[p.get(k,0.) for k in keys] for p in points],float)
    if len(data)<2 or not np.isfinite(data).all():raise ValueError('finite keyframes required')
    t=data[:,0]
    if t[0]!=0 or np.any(np.diff(t)<=0):raise ValueError('times must start at zero and increase')
    if data[0,3]>=0 or data[-1,3]>=0:raise ValueError('gesture starts and ends airborne')
    if np.max(data[:,3])>.8 or np.min(data[:,3])<-.8:raise ValueError('compression outside study range')
    if np.any(abs(data[:,4])>85):raise ValueError('tilt measured from normal; study supports +/-85 degrees')
    for j in [5,6]:data[:,j]=np.rad2deg(np.unwrap(np.deg2rad(data[:,j])))
    interp=PchipInterpolator(t,data[:,3:],axis=0)
    samples=[]
    for i in range(len(t)-1):
        span=t[i+1]-t[i];dist=np.linalg.norm(data[i+1,1:3]-data[i,1:3])
        n=max(1,math.ceil(span/settings.max_step_s),math.ceil(dist/settings.spatial_step_mm))
        for j in range(n):
            a=t[i]+span*j/n;b=t[i]+span*(j+1)/n;mid=(a+b)*.5
            r0=(a-t[i])/span;r1=(b-t[i])/span
            start=data[i,1:3]*(1-r0)+data[i+1,1:3]*r0
            end=data[i,1:3]*(1-r1)+data[i+1,1:3]*r1
            samples.append((b-a,b,start,end,interp(mid),i))
    return {'spec':spec,'initial':data[0], 'samples':samples, 'duration_s':float(t[-1])}

class TechniqueBrush(ReferenceBrush):
    version='stateful-technique-study-0.15.0'

    def __init__(self,canvas,*,ink=None,interaction=None,loading=None,technique=None,
                 paper_overrides=None,width_mm=1.45,brush_overrides=None,
                 meniscus_overrides=None,surface_contact_overrides=None,record_source=True):
        self.technique=technique or TechniqueSettings()
        s=self.technique
        if not all(np.isfinite(v) for k,v in asdict(s).items() if k!='contact_model'):raise ValueError('nonfinite technique parameter')
        if s.contact_model not in ('legacy','curved','resistance','fiber_resistance'):raise ValueError('unknown contact model')
        if s.fiber_surface_rms_m<=0:raise ValueError('invalid paper relief')
        if not 10<=s.pixel_size_um<=200:raise ValueError('pixel size outside study range')
        if not 0<s.supply_hydraulic_fraction<=1:raise ValueError('invalid coating hydraulic radius')
        if s.evaporation_m_s<0 or not 0<=s.relative_humidity<=1:raise ValueError('invalid drying setting')
        if s.max_step_s<=0 or s.spatial_step_mm<=0:raise ValueError('invalid integration controls')
        self.evaporated_water=0.;self.paper_added_water=0.;self.control_trace=[];self.tip_trace=[]
        self.ink_material=ink or InkMaterial();self.ink_paper=interaction or InkPaperInteraction()
        fluid,paper=material_settings(self.ink_material,self.ink_paper,paper_overrides)
        fluid=replace(fluid,pixel_m=s.pixel_size_um*1e-6)
        brush=ContactSettings(width=width_mm*1e-3/fluid.pixel_m,bunches=96)
        if brush_overrides:brush=replace(brush,**brush_overrides)
        super().__init__(canvas,brush,fluid=fluid,paper=paper,
            meniscus=replace(replace(MeniscusSettings(),radius_m=20e-6,max_coat_m=24e-6),**(meniscus_overrides or {})),
            surface_contact=replace(SurfaceContactSettings(bend_contact_fraction=.55,shaft_radius_ratio=2.4),**(surface_contact_overrides or {})),
            record_source=record_source)
        if self.ink_paper.entry_capture_fine or self.ink_paper.entry_capture_coarse:
            from .technique_paper import entry_native
            self.substrate.native=entry_native(self.ink_paper.entry_capture_fine,self.ink_paper.entry_capture_coarse)
        if s.contact_model=='fiber_resistance':
            # One fixed substrate microstructure governs contact and porous
            # resistance. Dense fiber regions protrude and have lower pore
            # volume/permeability; no new random field is sampled here.
            density=self.substrate.fiber[0]
            self.contact_height=(density-density.mean())/max(density.std(),1e-20)*s.fiber_surface_rms_m
            self.height=self.contact_height
            self.slope_y,self.slope_x=np.gradient(self.height,self.fluid.pixel_m)
        self.roll=0.;self.compression_m=0.;self.next_tip_frame=0.
        self.load_profile(loading or BrushLoading(),initial=True)

    def _transfer(self,dt,gate,side):
        if self.technique.finite_neck_supply:
            from .technique_transfer import finite_neck_transfer
            return finite_neck_transfer(self,dt,gate,side)
        return super()._transfer(dt,gate,side)

    def _deposit(self,j,dv,dm,center,radius_px,heel=None,cap=False):
        if self.technique.contact_model=='legacy' or cap:
            return super()._deposit(j,dv,dm,center,radius_px,heel,cap)
        if dv<=0:return
        f=self.fluid;mc=self.meniscus;p=self.surface_contact;x,y=center
        # The handle projection defines the axial direction. v0.14's laid
        # segment was perpendicular to its footprint's long axis.
        laid=np.linalg.norm(center-heel) if heel is not None else 0.
        axial=np.array([math.cos(self.angle),math.sin(self.angle)])
        segment=laid*axial+p.bend_contact_fraction*self.local_bending_xy[j]
        length=float(np.linalg.norm(segment));den=length*length
        pad=max(radius_px,.45)+length
        y0,y1,x0,x1=self._bounds(x,y,pad+1)
        if x1<=x0 or y1<=y0:raise ValueError('contact left canvas')
        yy,xx=np.mgrid[y0:y1,x0:x1];weight=np.zeros_like(xx,dtype=float)
        nominal=np.zeros_like(weight)
        center_h=float(map_coordinates(self.contact_height,[[y],[x]],order=1,mode='nearest')[0])
        coat=self.coat()[j];r=self.radius[j]
        # Quadratic distal arc, tangent at the tip. The proximal endpoint
        # rises toward the tilted shaft. This is a reduced contact geometry,
        # not an elastic-rod solution. No random quantity depends on time.
        rise=length*f.pixel_m*max(0.,math.cos(self.tilt))
        for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
            u=np.clip(((xx+ox-x)*segment[0]+(yy+oy-y)*segment[1])/max(den,1e-12),0,1)
            distance=np.hypot(xx+ox-x-u*segment[0],yy+oy-y-u*segment[1])
            gap=np.maximum(0.,center_h+self.gap[j]+rise*u*u-self.contact_height[y0:y1,x0:x1])
            reach=coat+self.film_h[y0:y1,x0:x1]
            width=np.sqrt(np.maximum(0.,(r+reach)**2-(r+gap)**2))/f.pixel_m
            contact=np.clip((reach-gap)/p.transition_m,0,1)
            weight+=np.clip((width+.35-distance)/.7,0,1)*contact*.25
            nominal+=np.clip((radius_px+.35-distance)/.7,0,1)*.25
        area=float(weight.sum())*f.pixel_m**2
        if self.technique.contact_model=='curved':
            fraction=float(np.clip(weight.sum()/max(nominal.sum(),1e-30),0,1))
        else:
            # Neck and paper contact are resistances in series. The neck
            # already limits upstream flux; an area ratio would count that
            # restriction twice. Zero accessible area still transfers zero.
            rh=mc.neck_radius_m*np.clip(coat/(mc.max_coat_m*.5),0,1.6)
            gn=math.pi*rh**4/(8*f.viscosity_pa_s*(mc.neck_length_m+self.gap[j]+2e-6))
            gc=mc.contact_permeability_m2*area/(f.viscosity_pa_s*mc.neck_length_m)
            fraction=gc/(gc+gn) if area>0 else 0.
        self.neck_v[j]+=dv*(1-fraction);self.neck_m[j]+=dm*(1-fraction)
        self.blocked_water+=dv*(1-fraction);self.blocked_mass+=dm*(1-fraction);self.contact_deposits+=1
        if area<=0 or fraction<=0:return
        weight/=weight.sum()
        self.film_h[y0:y1,x0:x1]+=dv*fraction*weight/f.pixel_m**2
        self.film_mass[y0:y1,x0:x1]+=dm*fraction*weight
        self._extend((y0,y1,x0,x1))

    def load_profile(self,loading,*,initial=False):
        """A preparatory liquid distribution; subsequent use only adds liquid.

        Initial preparation is permitted only before the brush first moves.
        An explicit later loading conservatively fills available capacity with
        the specified local solution; it never replaces existing pigment.
        """
        p=loading;n=self.settings.bunches
        if not all(np.isfinite(v) for v in asdict(p).values()):raise ValueError('nonfinite brush preparation')
        if not (0<=p.body_saturation<=1 and 0<=p.tip_saturation<=1 and abs(p.side_contrast)<=1):
            raise ValueError('invalid loading saturation/contrast')
        if min(p.body_concentration_kg_m3,p.tip_concentration_kg_m3)<0:raise ValueError('negative pigment')
        side=self.offsets@np.array([math.cos(math.radians(p.side_axis_deg)),math.sin(math.radians(p.side_axis_deg))])
        side=np.clip(side,-1,1)
        concentrations=np.r_[np.full(n,p.body_concentration_kg_m3),p.tip_concentration_kg_m3*(1+p.side_contrast*side)]
        target=self.capacity*np.r_[np.full(n,p.body_saturation),np.full(n,p.tip_saturation)]
        if initial:
            if self.time!=0 or self.hand is not None:raise ValueError('initial loading requires an unused brush')
            self.volume[:]=target;self.mass[:]=target*concentrations
            self.water_initial=float(self.volume.sum());self.mass_initial=float(self.mass.sum())
        else:
            amount=np.maximum(0,target-self.volume);mass=amount*concentrations
            self.volume+=amount;self.mass+=mass
            self.water_added_si+=float(amount.sum());self.mass_added_si+=float(mass.sum())
        self.loading=asdict(p);self._sync()

    def prewet(self,saturation,mask=None):
        if not 0<=saturation<=1:raise ValueError('invalid paper saturation')
        mask=np.ones(self.canvas.density.shape) if mask is None else np.asarray(mask,float)
        if mask.shape!=self.canvas.density.shape or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)):
            raise ValueError('invalid wetting mask')
        target=self.substrate.capacity*saturation*mask
        amount=np.maximum(0,target-self.substrate.water)
        self.substrate.water+=amount;self.paper_added_water+=float(amount.sum());self._sync_paper()

    def flush_paper(self):
        dt=self.pending_paper_s
        result=super().flush_paper()
        if dt>0 and self.technique.evaporation_m_s>0:
            p=self.substrate;area=self.fluid.pixel_m**2
            demand=self.technique.evaporation_m_s*(1-self.technique.relative_humidity)*dt*area
            film_take=np.minimum(p.film*area,demand);p.film-=film_take/area
            remaining=demand-film_take
            # A reduced top-boundary evaporation law; deeper layers replenish
            # this layer through the existing conservative porous transport.
            sat=np.clip(p.water[0]/p.capacity[0],0,1)
            take=np.minimum(p.water[0],remaining*sat);p.water[0]-=take
            dry=(p.water[0]<1e-24)&(p.film<1e-15)
            p.fixed[0,dry]+=p.fine[0,dry]+p.coarse[0,dry]+p.film_mass[dry]
            p.fine[0,dry]=0;p.coarse[0,dry]=0;p.film_mass[dry]=0
            self.evaporated_water+=float(film_take.sum()+take.sum());self._sync_paper()
        return result

    def draw_gesture(self,spec,callback=None):
        driver=compile_gesture(spec,self.technique);mc=self.meniscus;f=self.fluid;cfg=self.settings
        initial=driver['initial'];start=initial[1:3]*1e-3/f.pixel_m
        if self.hand is None:
            self.hand=start.copy();self.tip=start.copy();self.angle=math.radians(initial[5]);self.tilt=math.radians(initial[4]);self.roll=math.radians(initial[6])
            self.force=0.;self.spread=0.;self.compression_m=initial[3]*1e-3
            self.centers=start+self.offsets*cfg.width*.15
            self.target_z=mc.compression_m*self.rest-self.compression_m
            self.gap=self.target_z.copy();self.normal_deflection=np.zeros(cfg.bunches)
            ready=self.centers.copy()
            for _ in range(200):self._network(ready,.00025)
            self.anchor=self.centers.copy()
        elif np.linalg.norm(self.hand-start)>1e-6:raise ValueError('next gesture must start at prior hand position')
        offset=self.time
        for dt,t,h0,h1,pose,event in driver['samples']:
            h0=h0*1e-3/f.pixel_m;h1=h1*1e-3/f.pixel_m
            compression,tilt,azimuth,roll=pose
            self.compression_m=compression*1e-3;self.force=max(0.,self.compression_m/mc.compression_m)
            _,self.tip=_follow_linear(self.tip,h0,h1,dt,cfg.tip_tau)
            self.tilt=math.radians(tilt);self.angle=math.radians(azimuth);self.roll=math.radians(roll)
            _,self.spread=_follow_linear(self.spread,self.force,self.force,dt,cfg.spread_tau)
            side=abs(math.sin(self.tilt));radii=np.array([cfg.width*(.18+.66*side),cfg.width*.25])*math.sqrt(.07+self.spread)
            cr,sr=math.cos(self.roll),math.sin(self.roll)
            roots=self.offsets@np.array([[cr,sr],[-sr,cr]])
            co,si=math.cos(self.angle),math.sin(self.angle)
            desired=self.tip+(roots*radii)@np.array([[co,si],[-si,co]])
            ph=map_coordinates(self.height,[self.centers[:,1],self.centers[:,0]],order=1,mode='nearest')
            self.target_z=mc.compression_m*self.rest-self.compression_m*self.compliance-ph
            if self.technique.contact_model!='legacy':
                # A pointed tuft, tilted about the authored handle axis.
                # Roll moves the same hair identities through this geometry.
                elevation=mc.compression_m*self.rest+roots[:,0]*cfg.width*f.pixel_m*.18*math.sin(self.tilt)
                elevation-=elevation.min()-20e-6
                self.target_z=elevation-self.compression_m*self.compliance-ph
            self.normal_deflection=np.maximum(0,-self.target_z)
            self._network(desired,dt);self._feed(dt)
            transfer=self._pore_step(dt)+self._transfer(dt,compression>0,side)
            absorbed=self._surface(dt)
            if compression<=0:self.air_water+=transfer;self.after_detach_water+=absorbed
            self._sync();self.hand=h1;self.time=offset+t
            if self.time>=self.next_tip_frame:
                self.control_trace.append({'t_s':self.time,'x_mm':float(h1[0]*f.pixel_m*1e3),'y_mm':float(h1[1]*f.pixel_m*1e3),
                    'compression_mm':float(compression),'tilt_deg':float(tilt),'azimuth_deg':float(azimuth),'roll_deg':float(roll),
                    'speed_mm_s':float(np.linalg.norm(h1-h0)*f.pixel_m*1e3/dt),
                    'contact_count':int(self.contact.sum()),'liquid_contacts':int(self.neck_alive.sum()),
                    'brush_nl':float(self.volume.sum()*1e12),'pigment_ng':float(self.mass.sum()*1e12)})
                self.tip_trace.append(np.c_[self.centers,self.gap,self.volume[cfg.bunches:],self.mass[cfg.bunches:],self.contact])
                self.next_tip_frame=self.time+.025
            if callback:callback(self.time,self)
        self.flush_paper();self.previous_film[:]=self.film_h;self.previous_mass[:]=self.film_mass
        return driver

    def diagnostics(self):
        # The old diagnostic divides by initial inventory. Keep pure-water
        # and initially empty preparations valid without touching old code.
        wi,mi=self.water_initial,self.mass_initial
        self.water_initial=max(wi,1e-30);self.mass_initial=max(mi,1e-30)
        try:d=super().diagnostics()
        finally:self.water_initial,self.mass_initial=wi,mi
        if hasattr(self,'evaporated_water'):
            error=d['water_relative_error']*max(wi,1e-30)+max(wi,1e-30)-wi
            total=wi+self.paper_added_water+self.water_added_si
            d['water_relative_error']=(error-self.paper_added_water+self.evaporated_water)/max(total,1e-30)
            pigment_error=d['pigment_relative_error']*max(mi,1e-30)+max(mi,1e-30)-mi
            d['pigment_relative_error']=pigment_error/max(mi+self.mass_added_si,1e-30)
            d['initial_nl']=wi*1e12;d['initial_pigment_ng']=mi*1e12
            d['evaporated_nl']=self.evaporated_water*1e12;d['prewet_added_nl']=self.paper_added_water*1e12
            d['limits']='Uncalibrated reduced brush and top-boundary paper drying; no brush evaporation or full rod model.'
        return d
