"""v0.11: persistent bristle ribbons with a reduced cohesive contact network.

Capillary reservoirs and conservative paper transport are inherited unchanged.
This adds overdamped elastic tips, finite-range cohesive bonds, and swept narrow
contact ribbons. The cohesive springs are a reduced model, not a resolved
liquid-bridge/free-surface solver. No image masks or post-render grain removal.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import copy,math
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.linalg import solve_banded
from .capillary import CapillaryBrush,FluidSettings
from .cun import contact_update
from .response import _follow_linear


@dataclass(frozen=True)
class BristleSettings:
    bending_stiffness_n_m: float = .035
    cohesive_stiffness_n_m: float = .28
    bridge_length_m: float = .00016
    bridge_volume_fraction: float = .35
    form_range: float = .75
    break_range: float = 1.6
    drag_tau_s: float = .035
    rest_width_px: float = 1.2
    width_variation: float = .35
    laid_length_px: float = 2.8
    paper_contact_scale_px: float = 4.
    paper_height_fraction: float = .32
    contact_band: float = .035
    rest_height_variation: float = .28
    dry_splay: float = .20
    contact_clearance_m: float = 1.8e-6
    paper_texture_strength: float = .13
    paper_height_m: float = 18e-6
    normal_stiffness_n_m: float = .30
    normal_deflection_m: float = .0001
    bridge_contact_enabled: bool = True
    cohesive_enabled: bool = True
    mechanical_dt_s: float = .0002
    seed: int = 611


def material_settings():
    return replace(FluidSettings(),gap_mean_m=2.4e-6,gap_std_m=.25e-6,
                   brush_permeability_m2=3e-12,paper_lateral_permeability_m2=3e-15,
                   concentration_kg_m3=40.,extinction_m2_kg=11000.)


class BristleBrush(CapillaryBrush):
    version='bristle-contact-study-0.11.0'

    def __init__(self,canvas,settings=None,seed=606,fluid=None,bristles=None):
        super().__init__(canvas,settings,seed,fluid or material_settings())
        self.bristles=bristles or BristleSettings();bc=self.bristles;n=self.settings.bunches
        for key,value in asdict(bc).items():
            if not np.isfinite(value) or value<0:raise ValueError('invalid bristle setting: '+key)
        if bc.bending_stiffness_n_m<=0 or bc.drag_tau_s<=0 or bc.bridge_length_m<=0:
            raise ValueError('positive stiffness, drag time and bridge length required')
        rng=np.random.default_rng(bc.seed)
        # Each identity is a narrow tuft, with fixed length/stiffness/radius.
        spacing=rng.lognormal(0.,.36,n-1)
        u=np.r_[0.,np.cumsum(spacing)];u=1.9*(u/u[-1]-.5)
        self.offsets=np.c_[u,rng.normal(0.,.05,n)]
        self.hair_radius=bc.rest_width_px*np.exp(rng.normal(0.,bc.width_variation,n))
        self.hair_length=rng.uniform(.78,1.25,n)
        self.hair_stiffness=bc.bending_stiffness_n_m/self.hair_length**3
        self.rest=.025+.28*np.abs(u)**2.2+rng.uniform(0.,bc.rest_height_variation,n)
        self.compliance=rng.uniform(.92,1.08,n)
        self.bonds=np.zeros(n-1,dtype=bool)
        self.bond_changes=0
        self.ribbon_direction=np.tile([1.,0.],(n,1))
        self.sweep_start=None;self.sweep_end=None
        self._compression=np.zeros(n)
        # Paper relief is averaged over the contact scale before it changes
        # normal contact. Fine tooth only changes permeability, never cuts holes.
        field=gaussian_filter(canvas.paper.astype(float),bc.paper_contact_scale_px)
        self.macro_height=(field-field.mean())/max(field.std(),1e-12)
        self.slope_y,self.slope_x=np.gradient(self.macro_height*bc.paper_height_m,self.fluid.pixel_m)
        self.paper_tooth=np.clip(1.+bc.paper_texture_strength*(canvas.paper-.5)/.2,.6,1.4)
        self.edge_a=np.r_[np.arange(n),n+np.arange(n-1)]
        self.edge_b=np.r_[n+np.arange(n),n+np.arange(1,n)]
        self._base_conductance=self.conductance.copy()
        self.mechanical_trace=[]
        self._anchor_target=None

    def advance_tips(self,desired,dt,compression,gate):
        """Substep the stiff cohesive network independently of stroke sampling."""
        before=self.centers.copy()
        anchor0=desired if self._anchor_target is None else self._anchor_target
        count=max(1,math.ceil(dt/self.bristles.mechanical_dt_s))
        for k in range(count):
            anchor=anchor0+(desired-anchor0)*((k+.5)/count)
            self._advance_tips_sub(anchor,dt/count,compression,gate)
        self._anchor_target=desired.copy()
        return before,self.centers.copy()

    def _advance_tips_sub(self,desired,dt,compression,gate):
        """Implicit overdamped elastic network; bond topology carries memory.

        E=.5 sum k_i |x_i-a_i|^2 plus a finite-range cohesive potential.
        Bond reach scales as sqrt(available liquid / bridge contact length).
        Surface tension scales the cohesive stiffness; no bridge volume is
        added: reach uses a fraction of the existing tip inventory.
        """
        n=self.settings.bunches;bc=self.bristles;f=self.fluid
        before=self.centers.copy()
        delta=before[1:]-before[:-1]
        distance=np.linalg.norm(delta,axis=1)
        fallback=desired[1:]-desired[:-1]
        direction=np.divide(delta,distance[:,None],out=np.zeros_like(delta),where=distance[:,None]>1e-10)
        small=distance<1e-10
        direction[small]=fallback[small]/np.maximum(np.linalg.norm(fallback[small],axis=1)[:,None],1e-10)
        wet=np.clip(self.water,0.,1.)
        # Solid-core radius differs from the wider, wet paper contact ribbon.
        radius=.12*self.hair_radius
        surface_gap=np.maximum(0.,distance-radius[:-1]-radius[1:])
        bridge_volume=bc.bridge_volume_fraction*np.minimum(self.volume[n:-1],self.volume[n+1:])
        reach=np.sqrt(np.maximum(0.,bridge_volume)/(math.pi*bc.bridge_length_m))/f.pixel_m
        old=self.bonds.copy()
        if bc.cohesive_enabled:
            self.bonds=np.where(surface_gap<bc.form_range*reach,True,
                np.where(surface_gap>bc.break_range*reach,False,self.bonds))
            self.bonds &= (reach>1e-6)
        else:self.bonds[:]=False
        self.bond_changes+=int(np.count_nonzero(old!=self.bonds))
        rest=radius[:-1]+radius[1:]
        # Attraction weakens with separation. This permits wet tufts to gather
        # into unequal clumps instead of stabilizing an evenly spaced comb.
        delta=.6*np.maximum(reach,1e-9)
        force_length=reach/(1.+surface_gap/delta)**2
        kb=bc.cohesive_stiffness_n_m*(f.surface_tension_n_m/.055)*self.bonds
        kb*=force_length/np.maximum(distance-rest,.04)
        tau=bc.drag_tau_s*self.hair_length*(.4+1.4*compression)*(1.+.7*(1.-wet))
        if not gate:tau=np.full(n,.018)
        damping=self.hair_stiffness*tau
        diagonal=damping/dt+self.hair_stiffness
        diagonal[:-1]+=kb;diagonal[1:]+=kb
        band=np.zeros((3,n));band[1]=diagonal;band[0,1:]=-kb;band[2,:-1]=-kb
        rhs=(damping/dt)[:,None]*before+self.hair_stiffness[:,None]*desired
        if gate:
            slope=np.c_[map_coordinates(self.slope_x,[before[:,1],before[:,0]],order=1,mode='nearest'),
                        map_coordinates(self.slope_y,[before[:,1],before[:,0]],order=1,mode='nearest')]
            normal_force=bc.normal_stiffness_n_m*bc.normal_deflection_m*compression
            rhs-=normal_force[:,None]*slope/f.pixel_m
        pull=kb[:,None]*rest[:,None]*direction
        rhs[:-1]-=pull;rhs[1:]+=pull
        after=solve_banded((1,1),band,rhs,check_finite=False)
        self.centers=after
        # A liquid connection also carries transverse capillary exchange.
        self.conductance[:n]=self._base_conductance[:n]
        self.conductance[n:]=self._base_conductance[n:]*self.bonds
        movement=after-before
        length=np.linalg.norm(movement,axis=1)
        tangent=np.divide(movement,length[:,None],out=self.ribbon_direction.copy(),where=length[:,None]>1e-8)
        alpha=-math.expm1(-dt/.016)
        self.ribbon_direction=(1-alpha)*self.ribbon_direction+alpha*tangent
        self.ribbon_direction/=np.maximum(np.linalg.norm(self.ribbon_direction,axis=1)[:,None],1e-12)
        return before,after

    def footprint(self,centers,compression,side,angle):
        """Time-averaged narrow ribbons following persistent physical tufts.

        Two temporal samples and four pixel-area samples integrate the actual
        contact support. Supersampling is coverage integration, not image blur.
        Paper does not randomly enable/disable individual pixels.
        """
        indices=[];bunches=[];weights=[];gaps=[]
        h,w=self.canvas.density.shape;bc=self.bristles;f=self.fluid;n=self.settings.bunches
        film=self.volume[n:]/self.storage_area_m2
        wet_reach=np.sqrt(np.clip((film-bc.contact_clearance_m)/12e-6,0.,1.))
        for j in np.flatnonzero(self.contact):
            comp=compression[j]
            halfwidth=self.hair_radius[j]*(.24+.85*wet_reach[j])*(.6+.85*math.sqrt(comp))
            halflength=.5+bc.laid_length_px*side*math.sqrt(comp)*self.hair_length[j]
            a,b=self.sweep_start[j],self.sweep_end[j]
            direction=self.ribbon_direction[j]
            normal=np.array([-direction[1],direction[0]])
            pad=math.ceil(halflength+halfwidth+1.)
            lo=np.floor(np.minimum(a,b)).astype(int)-pad
            hi=np.ceil(np.maximum(a,b)).astype(int)+pad+1
            x0,y0=max(0,lo[0]),max(0,lo[1]);x1,y1=min(w,hi[0]),min(h,hi[1])
            if x1<=x0 or y1<=y0:continue
            yy,xx=np.mgrid[y0:y1,x0:x1]
            kernel=np.zeros_like(xx,dtype=float)
            for t in (.2113248654,.7886751346):
                center=a*(1-t)+b*t
                for ox,oy in ((-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)):
                    dx,dy=xx+ox-center[0],yy+oy-center[1]
                    along=(dx*direction[0]+dy*direction[1])/halflength
                    across=(dx*normal[0]+dy*normal[1])/max(halfwidth,.07)
                    r2=along**2+across**2
                    kernel+=np.clip((1.-r2)*2.,0.,1.)/8.
            take=kernel>1e-7
            if not np.any(take):continue
            pixel=yy[take]*w+xx[take]
            indices.append(pixel);bunches.append(np.full(len(pixel),j,dtype=int))
            weights.append(kernel[take]*self.paper_tooth.ravel()[pixel])
            # A tuft-scale clearance; no per-pixel liquid-film threshold.
            clearance=bc.contact_clearance_m*(.65+.45*self.hair_length[j])/(1.+1.5*comp)
            gaps.append(np.full(len(pixel),clearance))
        # A connected wet tuft group contacts paper through the interstitial
        # meniscus as well as through solid hair tips. Sharing this contact
        # debits the two endpoint reservoirs; it does not manufacture pigment.
        if bc.bridge_contact_enabled:
            for j in np.flatnonzero(self.bonds & self.contact[:-1] & self.contact[1:]):
                a,b=centers[j],centers[j+1];across=b-a
                along=self.ribbon_direction[j]+self.ribbon_direction[j+1]
                along/=max(np.linalg.norm(along),1e-12)
                det=across[0]*along[1]-across[1]*along[0]
                if abs(det)<.12:continue
                comp=min(compression[j],compression[j+1])
                length=.5+bc.laid_length_px*side*math.sqrt(comp)*min(self.hair_length[j:j+2])
                lo=np.floor(np.minimum(a,b)-length-1).astype(int)
                hi=np.ceil(np.maximum(a,b)+length+1).astype(int)
                x0,y0=max(0,lo[0]),max(0,lo[1]);x1,y1=min(w,hi[0]),min(h,hi[1])
                if x1<=x0 or y1<=y0:continue
                yy,xx=np.mgrid[y0:y1,x0:x1];kernel=np.zeros_like(xx,dtype=float)
                for ox,oy in ((-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)):
                    dx,dy=xx+ox-a[0],yy+oy-a[1]
                    u=(dx*along[1]-dy*along[0])/det
                    v=(across[0]*dy-across[1]*dx)/det
                    neck=length*(.55+.45*np.clip(np.abs(2*u-1),0.,1.))
                    kernel+=((u>=0)&(u<=1))*np.clip((1.-(v/neck)**2)*2.,0.,1.)/4.
                take=kernel>1e-7
                if not np.any(take):continue
                pixel=yy[take]*w+xx[take]
                for endpoint in (j,j+1):
                    indices.append(pixel);bunches.append(np.full(len(pixel),endpoint,dtype=int))
                    weights.append(.5*kernel[take]*self.paper_tooth.ravel()[pixel])
                    gaps.append(np.full(len(pixel),bc.contact_clearance_m/(1.+comp)))
        if not indices:return None
        pixel=np.concatenate(indices);unique,inverse=np.unique(pixel,return_inverse=True)
        return unique,inverse,np.concatenate(bunches),np.concatenate(weights),np.concatenate(gaps)

    def state(self):
        return {**super().state(),'cohesive_bonds':self.bonds.astype(int).tolist(),
            'bond_changes':self.bond_changes,'ribbon_direction':self.ribbon_direction.tolist(),
            'anchor_target':None if self._anchor_target is None else self._anchor_target.tolist()}

    def diagnostics(self):
        return {**super().diagnostics(),'bond_changes':self.bond_changes,
            'limits':'Reduced overdamped elastic/cohesive tuft network and conservative capillary reservoirs. No resolved 3D hairs or free liquid surface; parameters uncalibrated.'}

    def draw_driver(self,driver,frame_callback=None):
        first=driver['samples'][0];start=np.array([first['hand_x'],first['hand_y']],dtype=float)
        cfg=self.settings;bc=self.bristles
        targets={k:first[k] for k in ('pressure_target','tilt_target','angle_target')}
        if self.hand is None:
            self.hand=self.tip=start.copy();self.angle=first['angle_target'];self.tilt=first['tilt_target']
            # Start from a finite brush head, not 36 coincident solid cores.
            # The previous degenerate pose let round-off/step size choose the
            # first clumps. Prepare a fixed relaxed, ink-free airborne state.
            radii0=np.array([cfg.width*(.18+.66*abs(math.sin(self.tilt))),cfg.width*.22])*.20
            co0,si0=math.cos(self.angle),math.sin(self.angle)
            self.centers=start+(self.offsets*radii0)@np.array([[co0,si0],[-si0,co0]])
            self._anchor_target=self.centers.copy()
            for _ in range(1200):
                self._advance_tips_sub(self._anchor_target,.0001,np.zeros(cfg.bunches),False)
        elif not np.allclose(start,self.hand,rtol=0,atol=1e-9) or targets!=self.targets:
            raise ValueError('continue at previous hand and targets; use explicit airmove')
        if np.any(self.contact):raise ValueError('previous program must finish airborne')
        initial=self.state();rows=[];offset=self.time
        for sample in driver['samples']:
            dt=sample['dt']
            if dt<=0:continue
            hand0=np.array([sample['hand_x_start'],sample['hand_y_start']]);hand1=np.array([sample['hand_x'],sample['hand_y']])
            gate=sample['contact_gate_mid']>0
            tip_mid,self.tip=_follow_linear(self.tip,hand0,hand1,dt,cfg.tip_tau)
            fmid,self.force=_follow_linear(self.force,sample['pressure_target'],sample['pressure_target'],dt,cfg.pressure_tau)
            tmid,self.tilt=_follow_linear(self.tilt,sample['tilt_target'],sample['tilt_target'],dt,cfg.tilt_tau)
            target_angle=self.angle+math.atan2(math.sin(sample['angle_target']-self.angle),math.cos(sample['angle_target']-self.angle))
            amid,self.angle=_follow_linear(self.angle,target_angle,target_angle,dt,cfg.angle_tau)
            smid,self.spread=_follow_linear(self.spread,max(0.,fmid),max(0.,fmid),dt,cfg.spread_tau)
            side=abs(math.sin(tmid));spread=math.sqrt(max(0.,smid))
            radii=np.array([cfg.width*(.18+.66*side),cfg.width*.22])*spread
            co,si=math.cos(amid),math.sin(amid)
            end_radii=np.array([cfg.width*(.18+.66*abs(math.sin(self.tilt))),cfg.width*.22])*math.sqrt(.04+max(0.,self.spread))
            ce,se=math.cos(self.angle),math.sin(self.angle)
            desired=self.tip+(self.offsets*end_radii)@np.array([[ce,se],[-se,ce]])
            height=map_coordinates(self.macro_height,[self.centers[:,1],self.centers[:,0]],order=1,mode='nearest')
            dry_lift=bc.dry_splay*np.maximum(0.,.6-self.water)*self.hair_length
            load=fmid*(self.compliance-bc.paper_height_fraction*height)-self.rest-dry_lift
            compression=np.maximum(0.,load)
            self.sweep_start,self.sweep_end=self.advance_tips(desired,dt,compression,gate)
            self.contact=contact_update(self.contact,load,bc.contact_band) if gate else np.zeros_like(self.contact)
            footprint=self.footprint((self.sweep_start+self.sweep_end)*.5,compression,side,amid) if gate else None
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
                'speed_m_s':float(np.linalg.norm(hand1-hand0)/dt*self.fluid.pixel_m),
                'bonds':self.bonds.astype(int).tolist(),'contacts':self.contact.astype(int).tolist(),
                'centers':self.centers.tolist()}
            self.fluid_trace.append(row);rows.append(row)
            if frame_callback is not None:frame_callback(self.time,self.canvas)
        self.records.append({'version':self.version,'program':copy.deepcopy(driver['program']),
            'curves':copy.deepcopy(driver.get('curves')),'driver_hash':driver['driver_hash'],
            'settings':asdict(cfg),'fluid':asdict(self.fluid),'bristles':asdict(bc),'seed':self.seed,
            'initial_state':initial,'final_state':self.state(),'trace':rows,'events':driver['events']})
        return self.records[-1]
