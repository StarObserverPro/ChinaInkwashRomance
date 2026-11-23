"""Conservative SI-unit capillary transport study, v0.10.

This is a lumped porous-flow model, NOT a resolved free-surface/Young-Laplace
bridge solver. Static geometry can be heterogeneous; no travelling random
field modulates ink supply. The numerical closures and assumptions are
documented in docs/capillary_v010.md.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import math
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from .reference_iteration import JoinedContactBrush
from .cun import contact_update
from .response import _follow_linear


@dataclass(frozen=True)
class FluidSettings:
    pixel_m: float = 50e-6
    viscosity_pa_s: float = .003
    surface_tension_n_m: float = .055
    contact_angle_rad: float = .45
    body_capacity_m3: float = 4e-9
    tip_storage_height_m: float = 30e-6
    brush_length_m: float = .006
    brush_porosity: float = .65
    brush_permeability_m2: float = 2e-11
    body_pore_radius_m: float = 22e-6
    tip_pore_radius_m: float = 18e-6
    retention_lambda: float = 2.
    lateral_fraction: float = .04
    refill_multiplier: float = 1.
    paper_permeability_m2: float = 2e-16
    paper_lateral_permeability_m2: float = 8e-15
    paper_pore_radius_m: float = 4e-6
    paper_porosity: float = .60
    paper_thickness_m: float = 80e-6
    front_regularization_m: float = 8e-6
    gap_mean_m: float = 10e-6
    gap_std_m: float = 4e-6
    compression_depth_m: float = 7e-6
    wetting_regularization_m: float = .8e-6
    concentration_kg_m3: float = 20.
    extinction_m2_kg: float = 5000.
    pigment_fixation_rate_s: float = 18.
    paper_tick_s: float = .01
    max_fluid_dt_s: float = .00075
    paper_seed: int = 707
    anatomy_seed: int = 60610

    def validate(self):
        nonnegative={'surface_tension_n_m','lateral_fraction','refill_multiplier',
                     'gap_std_m','compression_depth_m','concentration_kg_m3',
                     'paper_lateral_permeability_m2','pigment_fixation_rate_s'}
        for key,value in asdict(self).items():
            if not np.isfinite(value) or (value<0 if key in nonnegative else value<=0):
                raise ValueError('invalid fluid setting: '+key)
        if not 0<self.contact_angle_rad<math.pi/2:
            raise ValueError('this study assumes a hydrophilic contact angle')
        if not 0<self.brush_porosity<=1 or not 0<self.paper_porosity<=1:
            raise ValueError('porosity must be in (0,1]')


def conservative_exchange(volume,mass,capacity,donor,receiver,proposal):
    """Simultaneous directed network flux with conservative upwind tracer.

    Positive proposals are limited by total donor inventory and receiver
    capacity. Limiters are safeguards, not a replacement for refinement tests.
    Arrays volume and mass are updated in place; the returned volumes are the
    actual edge flux integrated over the time step.
    """
    size=len(volume)
    outgoing=np.bincount(donor,weights=proposal,minlength=size)
    incoming=np.bincount(receiver,weights=proposal,minlength=size)
    source=np.minimum(1.,np.divide(volume,outgoing,out=np.ones(size),where=outgoing>0))
    dest=np.minimum(1.,np.divide(np.maximum(0.,capacity-volume),incoming,
                                out=np.ones(size),where=incoming>0))
    amount=proposal*np.minimum(source[donor],dest[receiver])
    concentration=np.divide(mass,volume,out=np.zeros(size),where=volume>1e-30)
    pigment=amount*concentration[donor]
    np.add.at(volume,donor,-amount);np.add.at(volume,receiver,amount)
    np.add.at(mass,donor,-pigment);np.add.at(mass,receiver,pigment)
    return amount


def washburn_increment(depth,drive_pa,permeability,dt,phi,viscosity,z0,max_depth):
    """Integrated 1D wetting-front advance with constant pressure this step."""
    new=np.sqrt((depth+z0)**2+2.*permeability*np.maximum(drive_pa,0.)*dt/(phi*viscosity))-z0
    return np.maximum(0.,np.minimum(new,max_depth)-depth)


class CapillaryBrush(JoinedContactBrush):
    version='capillary-contact-study-0.10.0'

    def __init__(self,canvas,settings=None,seed=606,fluid=None):
        super().__init__(canvas,settings,seed)
        self.fluid=fluid or FluidSettings();self.fluid.validate()
        f=self.fluid;n=self.settings.bunches
        rng=np.random.default_rng(f.anatomy_seed)
        # Fixed solid geometry only. No state is sampled randomly during use.
        g=min(7,max(1,n//3))
        counts=np.full(g,2)+rng.multinomial(n-2*g,np.ones(g)/g)
        centers=np.linspace(-.87,.87,g) if g>1 else np.zeros(1)
        centers+=rng.uniform(-.06,.06,g)
        ids=np.repeat(np.arange(g),counts)
        self.offsets=np.c_[centers[ids]+rng.normal(0.,.055,n),rng.normal(0.,.065,n)]
        self.rest=.04+.16*np.abs(self.offsets[:,0])**3
        self.patch_scale=rng.uniform(.82,1.18,n)
        pore_factor=np.exp(rng.normal(0.,.22,g))[ids]
        self.pore_radii=np.r_[f.body_pore_radius_m*pore_factor,f.tip_pore_radius_m*pore_factor]
        self.capillary_entry_pa=2.*f.surface_tension_n_m*math.cos(f.contact_angle_rad)/self.pore_radii
        # Storage footprint is fixed in brush coordinates. Compression affects
        # contact geometry separately; it does not silently create liquid.
        areas=[]
        for j in range(n):
            *_,area,_=JoinedContactBrush._bunch_footprint(self,j,[0.,0.],.45,1.,.8,0.)
            areas.append(area*f.pixel_m**2)
        self.storage_area_m2=np.array(areas)
        body_capacity=np.full(n,f.body_capacity_m3/n)
        self.capacity=np.r_[body_capacity,self.storage_area_m2*f.tip_storage_height_m]
        self.volume=self.capacity*self.settings.initial_water
        self.mass=self.volume*f.concentration_kg_m3
        self.water_initial=float(self.volume.sum());self.mass_initial=float(self.mass.sum())
        self.water_added_si=self.mass_added_si=0.
        self.paper_water=np.zeros(canvas.density.shape,dtype=np.float64)
        self.paper_mass=np.zeros_like(self.paper_water)
        self.paper_mobile=np.zeros_like(self.paper_water)
        self.paper_fixed=np.zeros_like(self.paper_water)
        self.paper_pending_s=0.
        self.paper_bounds=None
        prng=np.random.default_rng(f.paper_seed)
        micro=gaussian_filter(prng.normal(size=canvas.density.shape),.65)
        meso=gaussian_filter(prng.normal(size=canvas.density.shape),2.1)
        relief=.75*micro/max(micro.std(),1e-15)+.25*meso/max(meso.std(),1e-15)
        self.gap_field=np.maximum(.2e-6,f.gap_mean_m+f.gap_std_m*relief)
        self.relief=np.clip(1.-self.gap_field/(3*f.gap_mean_m),0.,1.)
        # Darcy conductance k A/(mu L), with pore size affecting both k and Pc.
        area=body_capacity/(f.brush_porosity*f.brush_length_m)
        self.root_conductance=f.brush_permeability_m2*pore_factor**2*area/(f.viscosity_pa_s*f.brush_length_m)
        order=np.argsort(self.offsets[:,0])
        self.edge_a=np.r_[np.arange(n),n+order[:-1]]
        self.edge_b=np.r_[n+np.arange(n),n+order[1:]]
        self.conductance=np.r_[self.root_conductance,np.full(n-1,f.lateral_fraction*self.root_conductance.mean())]
        self.conductance*=f.refill_multiplier
        self.fluid_trace=[];self._sync()

    def _paper_contact(self,yy,xx,compression,wet):
        # The parent supplies geometry only. Fluid contact is evaluated below.
        return np.ones_like(xx,dtype=float)

    def _sync(self):
        n=self.settings.bunches
        self.water=self.volume[n:]/self.capacity[n:]
        self.pigment=self.mass[:n]+self.mass[n:]

    def pressure(self):
        saturation=np.clip(self.volume/self.capacity,1e-4,1.)
        # Brooks-Corey-type retention closure, with an illustrative exponent.
        # This curve has not been measured for the reference brush.
        return -self.capillary_entry_pa*saturation**(-1./self.fluid.retention_lambda)

    def replenish(self,dt):
        p=self.pressure();a,b=self.edge_a,self.edge_b
        signed=p[a]-p[b]
        donor=np.where(signed>=0,a,b);receiver=np.where(signed>=0,b,a)
        saturation=np.clip(self.volume[donor]/self.capacity[donor],0.,1.)
        mobility=saturation**(3.+2./self.fluid.retention_lambda)
        proposed=self.conductance*mobility*np.abs(signed)*dt
        actual=conservative_exchange(self.volume,self.mass,self.capacity,donor,receiver,proposed)
        n=self.settings.bunches
        return float(np.sum(actual[:n]*np.where(donor[:n]<n,1.,-1.)))

    def footprint(self,centers,compression,side,angle):
        """Assemble all visible geometric patches for simultaneous transfer."""
        indices=[];bunches=[];weights=[];gaps=[]
        h,w=self.canvas.density.shape;f=self.fluid
        for j in np.flatnonzero(self.contact):
            x,y,kernel,_,_=JoinedContactBrush._bunch_footprint(self,j,centers[j],compression[j],self.water[j],side,angle)
            yy,xx=np.mgrid[y:y+kernel.shape[0],x:x+kernel.shape[1]]
            take=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(kernel>1e-6)
            pixel=yy[take]*w+xx[take]
            indices.append(pixel);bunches.append(np.full(len(pixel),j,dtype=int));weights.append(kernel[take])
            gaps.append(np.maximum(.2e-6,self.gap_field.ravel()[pixel]-f.compression_depth_m*compression[j]))
        if not indices:return None
        pixel=np.concatenate(indices)
        unique,inverse=np.unique(pixel,return_inverse=True)
        return unique,inverse,np.concatenate(bunches),np.concatenate(weights),np.concatenate(gaps)

    def absorb(self,dt,footprint):
        if footprint is None:return 0.,0.,0
        unique,inverse,j,coverage,gap=footprint
        f=self.fluid;n=self.settings.bunches;pixel_area=f.pixel_m**2
        depth=self.paper_water.ravel()[unique]/(f.paper_porosity*pixel_area)
        film=self.volume[n:]/self.storage_area_m2
        # Finite film reach is a subgrid geometric closure, not a free-surface
        # curvature solve. When a film cannot span a gap there is no transfer.
        bridge=np.clip((film[j]-gap)/f.wetting_regularization_m,0.,1.)
        pc_paper=2*f.surface_tension_n_m*math.cos(f.contact_angle_rad)/f.paper_pore_radius_m
        drive=pc_paper+self.pressure()[n:][j]
        advance=washburn_increment(depth[inverse],drive,f.paper_permeability_m2,
            dt*coverage*bridge,f.paper_porosity,f.viscosity_pa_s,
            f.front_regularization_m,f.paper_thickness_m)
        wanted=advance*f.paper_porosity*pixel_area
        demand=np.bincount(j,weights=wanted,minlength=n)
        donor_scale=np.minimum(1.,np.divide(self.volume[n:],demand,out=np.ones(n),where=demand>0))
        per_pixel=np.bincount(inverse,weights=wanted,minlength=len(unique))
        free=np.maximum(0.,f.paper_porosity*pixel_area*f.paper_thickness_m-self.paper_water.ravel()[unique])
        paper_scale=np.minimum(1.,np.divide(free,per_pixel,out=np.ones(len(unique)),where=per_pixel>0))
        water=wanted*np.minimum(donor_scale[j],paper_scale[inverse])
        c=np.divide(self.mass[n:],self.volume[n:],out=np.zeros(n),where=self.volume[n:]>1e-30)
        ink=water*c[j]
        self.volume[n:]-=np.bincount(j,weights=water,minlength=n)
        self.mass[n:]-=np.bincount(j,weights=ink,minlength=n)
        dw=np.bincount(inverse,weights=water,minlength=len(unique))
        dm=np.bincount(inverse,weights=ink,minlength=len(unique))
        self.paper_water.ravel()[unique]+=dw;self.paper_mass.ravel()[unique]+=dm
        self.paper_mobile.ravel()[unique]+=dm
        self.canvas.density.ravel()[unique]+=dm*f.extinction_m2_kg/pixel_area
        if np.any(dw>0):
            ys,xs=np.divmod(unique[dw>0],self.canvas.width)
            box=(int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1)
            if self.paper_bounds is None:self.paper_bounds=box
            else:
                a,b,c,d=self.paper_bounds;e,g,h,i=box
                self.paper_bounds=(min(a,e),max(b,g),min(c,h),max(d,i))
        active=len(np.unique(j[water>1e-25]))
        return float(water.sum()),float(ink.sum()),active

    def fluid_step(self,dt,footprint):
        nsteps=max(1,math.ceil(dt/self.fluid.max_fluid_dt_s));sub=dt/nsteps
        supply=water=mass=0.;active=0
        for _ in range(nsteps):
            supply+=self.replenish(sub*.5)
            self.paper_pending_s+=sub
            while self.paper_pending_s+1e-14>=self.fluid.paper_tick_s:
                self.paper_step(self.fluid.paper_tick_s)
                self.paper_pending_s=max(0.,self.paper_pending_s-self.fluid.paper_tick_s)
            dw,dm,active=self.absorb(sub,footprint);water+=dw;mass+=dm
            supply+=self.replenish(sub*.5)
        self._sync()
        return supply,water,mass,active

    def paper_step(self,dt):
        """2D conservative porous flow + upwind pigment and first-order fixation.

        This evolves actual water and pigment. It is not an image blur. The
        paper lateral permeability and fixation rate are uncalibrated inputs.
        """
        if self.paper_bounds is None:return
        f=self.fluid
        y0,y1,x0,x1=self.paper_bounds
        y0=max(0,y0-1);y1=min(self.canvas.height,y1+1)
        x0=max(0,x0-1);x1=min(self.canvas.width,x1+1)
        v=self.paper_water[y0:y1,x0:x1]
        mobile=self.paper_mobile[y0:y1,x0:x1]
        fixed=self.paper_fixed[y0:y1,x0:x1]
        cap=f.pixel_m**2*f.paper_porosity*f.paper_thickness_m
        s=np.clip(v/cap,0.,1.)
        pe=2*f.surface_tension_n_m*math.cos(f.contact_angle_rad)/f.paper_pore_radius_m
        pressure=-pe*np.maximum(s,1e-3)**(-1/f.retention_lambda)
        # Interface cross section is dx*thickness; path length is dx.
        conductance=f.paper_lateral_permeability_m2*f.paper_thickness_m/f.viscosity_pa_s
        dx=pressure[:,:-1]-pressure[:,1:]
        dy=pressure[:-1,:]-pressure[1:,:]
        exponent=3.+2/f.retention_lambda
        qx=conductance*dx*np.where(dx>=0,s[:,:-1],s[:,1:])**exponent*dt
        qy=conductance*dy*np.where(dy>=0,s[:-1,:],s[1:,:])**exponent*dt
        xp,xm=np.maximum(qx,0.),np.maximum(-qx,0.)
        yp,ym=np.maximum(qy,0.),np.maximum(-qy,0.)
        outgoing=np.zeros_like(v);incoming=np.zeros_like(v)
        outgoing[:,:-1]+=xp;outgoing[:,1:]+=xm
        outgoing[:-1,:]+=yp;outgoing[1:,:]+=ym
        incoming[:,1:]+=xp;incoming[:,:-1]+=xm
        incoming[1:,:]+=yp;incoming[:-1,:]+=ym
        source=np.minimum(1.,np.divide(v,outgoing,out=np.ones_like(v),where=outgoing>0))
        dest=np.minimum(1.,np.divide(np.maximum(0.,cap-v),incoming,out=np.ones_like(v),where=incoming>0))
        qx*=np.where(qx>=0,np.minimum(source[:,:-1],dest[:,1:]),np.minimum(source[:,1:],dest[:,:-1]))
        qy*=np.where(qy>=0,np.minimum(source[:-1,:],dest[1:,:]),np.minimum(source[1:,:],dest[:-1,:]))
        concentration=np.divide(mobile,v,out=np.zeros_like(v),where=v>1e-30)
        mx=qx*np.where(qx>=0,concentration[:,:-1],concentration[:,1:])
        my=qy*np.where(qy>=0,concentration[:-1,:],concentration[1:,:])
        v[:,:-1]-=qx;v[:,1:]+=qx;v[:-1,:]-=qy;v[1:,:]+=qy
        mobile[:,:-1]-=mx;mobile[:,1:]+=mx;mobile[:-1,:]-=my;mobile[1:,:]+=my
        attached=mobile*(-math.expm1(-f.pigment_fixation_rate_s*dt))
        mobile-=attached;fixed+=attached
        self.paper_mass[y0:y1,x0:x1]=mobile+fixed
        self.canvas.density[y0:y1,x0:x1]=(mobile+fixed)*f.extinction_m2_kg/f.pixel_m**2
        yy,xx=np.nonzero(v>1e-24)
        if len(yy):self.paper_bounds=(y0+int(yy.min()),y0+int(yy.max())+1,x0+int(xx.min()),x0+int(xx.max())+1)

    def reload(self,water_load,ink_load):
        # Add actual liquid carrying pigment; do not top up pigment alone.
        dv=np.maximum(0.,self.capacity*water_load-self.volume)
        dm=dv*self.fluid.concentration_kg_m3*ink_load
        self.volume+=dv;self.mass+=dm
        self.water_added_si+=float(dv.sum());self.mass_added_si+=float(dm.sum())
        self._sync()

    def state(self):
        return {**super().state(),'liquid_volume_m3':self.volume.tolist(),
                'pigment_mass_kg':self.mass.tolist(),
                'paper_pending_s':self.paper_pending_s,
                'paper_bounds':None if self.paper_bounds is None else list(self.paper_bounds),
                'liquid_added_m3':self.water_added_si,'pigment_added_kg':self.mass_added_si}

    def diagnostics(self):
        v=float(self.volume.sum());m=float(self.mass.sum())
        pw=float(self.paper_water.sum());pm=float(self.paper_mass.sum())
        return {'version':self.version,'liquid_initial_m3':self.water_initial,
            'liquid_added_m3':self.water_added_si,'liquid_in_brush_m3':v,'liquid_in_paper_m3':pw,
            'liquid_balance_error_m3':self.water_initial+self.water_added_si-v-pw,
            'pigment_initial_kg':self.mass_initial,'pigment_added_kg':self.mass_added_si,
            'pigment_in_brush_kg':m,'pigment_in_paper_kg':pm,
            'pigment_balance_error_kg':self.mass_initial+self.mass_added_si-m-pm,
            'air_transfer_m3':sum(r['to_paper_m3'] for r in self.fluid_trace if not r['contact_gate']),
            'actual_touches':sum(a['kind']=='touch' for r in self.records for a in r['program']['actions']),
            'reloads':sum(a['kind']=='reload' for r in self.records for a in r['program']['actions']),
            'limits':'Lumped capillary reservoirs, vertical and lateral porous paper flow, passive pigment with first-order fixation. No resolved liquid bridge, elastocapillary clumping, evaporation or calibration.'}

    def draw_driver(self,driver):
        first=driver['samples'][0];start=np.array([first['hand_x'],first['hand_y']],dtype=float)
        cfg=self.settings;targets={k:first[k] for k in ('pressure_target','tilt_target','angle_target')}
        if self.hand is None:
            self.hand=self.tip=start.copy();self.angle=first['angle_target'];self.tilt=first['tilt_target']
            self.centers=np.repeat(start[None,:],cfg.bunches,axis=0)
        elif not np.allclose(start,self.hand,rtol=0,atol=1e-9) or targets!=self.targets:
            raise ValueError('continue at previous hand and targets; use explicit airmove')
        if np.any(self.contact):raise ValueError('previous program must finish airborne')
        initial=self.state();rows=[];offset=self.time
        for sample in driver['samples']:
            dt=sample['dt']
            if dt<=0:continue
            hand0=np.array([sample['hand_x_start'],sample['hand_y_start']])
            hand1=np.array([sample['hand_x'],sample['hand_y']])
            gate=sample['contact_gate_mid']>0
            tip_mid,self.tip=_follow_linear(self.tip,hand0,hand1,dt,cfg.tip_tau)
            fmid,self.force=_follow_linear(self.force,sample['pressure_target'],sample['pressure_target'],dt,cfg.pressure_tau)
            tmid,self.tilt=_follow_linear(self.tilt,sample['tilt_target'],sample['tilt_target'],dt,cfg.tilt_tau)
            target_angle=self.angle+math.atan2(math.sin(sample['angle_target']-self.angle),math.cos(sample['angle_target']-self.angle))
            amid,self.angle=_follow_linear(self.angle,target_angle,target_angle,dt,cfg.angle_tau)
            smid,self.spread=_follow_linear(self.spread,max(0.,fmid),max(0.,fmid),dt,cfg.spread_tau)
            side=abs(math.sin(tmid));spread=math.sqrt(max(0.,smid))
            radii=np.array([cfg.width*(.18+.66*side),cfg.width*.25])*spread
            co,si=math.cos(amid),math.sin(amid)
            desired=tip_mid+(self.offsets*radii)@np.array([[co,si],[-si,co]])
            tau=.004+.018*(1.-self.water)*self.compliance
            centers_mid=desired+(self.centers-desired)*np.exp(-dt/(2.*tau[:,None]))
            self.centers=desired+(self.centers-desired)*np.exp(-dt/tau[:,None])
            relief=map_coordinates(self.relief,[centers_mid[:,1],centers_mid[:,0]],order=1,mode='constant',cval=.5)
            gap=fmid*self.compliance+.11*self.water-self.rest-cfg.paper_relief*(1.-relief)
            self.contact=contact_update(self.contact,gap,cfg.hysteresis) if gate else np.zeros_like(self.contact)
            compression=np.maximum(0.,gap+cfg.hysteresis)
            footprint=self.footprint(centers_mid,compression,side,amid) if gate else None
            supply,water,mass,active=self.fluid_step(dt,footprint)
            event=driver['events'][sample['event_index']]
            if sample['event_end'] and event['kind']=='reload':self.reload(event['water_load'],event['ink_load'])
            if sample['event_end'] and event['kind']=='release':self.contact[:]=False
            self.hand=hand1;self.time=offset+sample['t'];self.targets={k:sample[k] for k in targets}
            n=cfg.bunches
            row={'t':self.time,'dt':dt,'hand':hand1.tolist(),'tip':self.tip.tolist(),
                 'phase':event['kind'],'event':sample['event_index'],'contact_gate':bool(gate),
                 'root_to_tip_m3':supply,'to_paper_m3':water,'to_paper_kg':mass,
                 'body_volume_m3':float(self.volume[:n].sum()),'tip_volume_m3':float(self.volume[n:].sum()),
                 'tip_saturation':self.water.tolist(),'fluid_active_bunches':active,
                 'pressure_target':sample['pressure_target'],'force':self.force,'angle':self.angle,
                 'speed_m_s':float(np.linalg.norm(hand1-hand0)/dt*self.fluid.pixel_m)}
            self.fluid_trace.append(row);rows.append(row)
        record={'version':self.version,'program':copy.deepcopy(driver['program']),
                'curves':copy.deepcopy(driver.get('curves')),'driver_hash':driver['driver_hash'],
                'settings':asdict(cfg),'fluid':asdict(self.fluid),'seed':self.seed,
                'initial_state':initial,'final_state':self.state(),'trace':rows,'events':driver['events']}
        self.records.append(record)
        return record


__all__=['FluidSettings','CapillaryBrush','conservative_exchange','washburn_increment']
