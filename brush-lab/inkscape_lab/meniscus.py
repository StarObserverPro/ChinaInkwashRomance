"""v0.12: energy-based inter-hair bridges + moving free surface + porous paper.

New research backend, not an implementation of the complete libWetHair solver.
Equal-cylinder circular menisci follow Fei et al. 2017 §3.1; forces are numerical
derivatives of interfacial energy at fixed volume. Tip mechanics are a reduced
cantilever model k=3EI/L^3, not a full 3-D rod discretization. Surface liquid is
an independently evolved conservative thin film. See docs/meniscus_v012.md.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import ctypes,math,subprocess,json,hashlib,shutil,os
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix,eye,diags
from scipy.sparse.linalg import spsolve
from .capillary import CapillaryBrush,FluidSettings,conservative_exchange
from .response import _follow_linear


@dataclass(frozen=True)
class MeniscusSettings:
    radius_m: float=38e-6
    young_pa: float=2.4e9
    hair_length_m: float=.0038
    wet_length_m: float=.00024
    max_coat_m: float=48e-6
    mechanical_tau_s: float=.012
    friction_tau_s: float=.035
    compression_m: float=650e-6
    lift_height_m: float=320e-6
    contact_angle_rad: float=.45
    precursor_m: float=.5e-6
    paper_angle_rad: float=.10
    paper_roughness_m: float=.18e-6
    paper_relief_m: float=16e-6
    neck_radius_m: float=1.5e-6
    neck_length_m: float=50e-6
    bridge_hold_s: float=.009
    contact_line_log_ratio: float=8.
    contact_angle_advancing_rad: float=.28
    contact_angle_receding_rad: float=.045
    friction_coefficient: float=.50
    friction_regularization_m: float=.7e-6
    neck_capacity_m3: float=.11e-12
    contact_permeability_m2: float=1.4e-16
    pores_enabled: bool=True
    pore_depth_m: float=30e-6
    pore_permeability_m2: float=6e-15
    pore_fill_saturation: float=.62
    pore_drain_saturation: float=.24
    body_feed_scale: float=1.
    lateral_flow: bool=True
    cohesion: bool=True
    film_flow: bool=True
    retained_bridge: bool=True
    max_dt_s: float=.0015
    anatomy_seed: int=1212
    irregular_roots: bool=True
    radius_log_sd: float=.12
    length_log_sd: float=.22
    distributed_sliding_contact: bool=True
    contact_volume_taper: bool=True


def material():
    return replace(FluidSettings(),body_capacity_m3=.8e-9,brush_permeability_m2=1.2e-12,
        paper_permeability_m2=7e-16,paper_lateral_permeability_m2=2e-15,
        concentration_kg_m3=28.,extinction_m2_kg=11000.,paper_tick_s=.016)


class BridgeTable:
    """Circular Young-Laplace sections; dimensionless radii/area/energy.

    Tabulates the connected branch at fixed liquid area. The empirically based
    rupture limit in Fei eq.9 is additionally applied by the caller. Outside the
    connected branch no force or interstitial source is evaluated.
    """
    def __init__(self,theta=.45):
        self.theta=theta
        self.ds=np.linspace(2.002,11.,210)
        self.areas=np.geomspace(.003,65.,230)
        self.energy=np.empty((210,230));self.radius=self.energy.copy();self.waist=self.energy.copy()
        self.alpha=self.energy.copy();self.valid=np.zeros_like(self.energy,dtype=bool)
        a=np.linspace(.00001,math.pi-theta-.02,6000)
        t=a+theta
        for i,d in enumerate(self.ds):
            R=(d-2*np.cos(a))/(2*np.cos(t))
            A=2*R**2*(t-math.pi/2+.5*np.sin(2*t))+2*R*(np.sin(2*a+theta)-math.sin(theta))-(2*a-np.sin(2*a))
            H=np.sin(a)-R*(1-np.sin(t))
            keep=(A>0)&(H>0)&np.isfinite(A)&(abs(np.cos(t))>1e-4)
            # E/gamma/r = liquid-air arc + cos(theta)*dry solid arc;
            # the constant 4*pi*cos(theta) is omitted.
            E=4*R*(math.pi/2-t)-4*a*math.cos(theta)
            for dest,value in [(self.energy,E),(self.radius,R),(self.waist,H),(self.alpha,a)]:
                dest[i]=np.interp(self.areas,A[keep],value[keep])
            self.valid[i]=(self.areas>=A[keep].min())&(self.areas<=A[keep].max())
        self.force=np.maximum(0.,np.gradient(self.energy,self.ds,axis=0))

    def lookup(self,d,A):
        d=np.maximum(d,self.ds[0])
        coord=np.array([(np.clip(d,self.ds[0],self.ds[-1])-self.ds[0])/(self.ds[1]-self.ds[0]),
            (np.log(np.maximum(A,self.areas[0]))-math.log(self.areas[0]))/math.log(self.areas[1]/self.areas[0])])
        vals=[map_coordinates(v,coord,order=1,mode='nearest') for v in (self.radius,self.waist,self.alpha,self.force,self.valid.astype(float))]
        valid=(vals[-1]>.98)&(d<=self.ds[-1])&(A<=self.areas[-1])&(A>=self.areas[0])
        a=vals[2];theta=self.theta
        def section(alpha):
            t=alpha+theta;R=(d-2*np.cos(alpha))/(2*np.cos(t))
            return 2*R**2*(t-math.pi/2+.5*np.sin(2*t))+2*R*(np.sin(2*alpha+theta)-math.sin(theta))-(2*alpha-np.sin(2*alpha))
        # Radius has a pole at a flat meniscus. Interpolating R separately can
        # violate volume by tens of percent. Correct alpha at fixed A, then
        # recover R and the waist from one geometrically consistent section.
        for _ in range(2):
            slope=(section(a+1e-5)-section(a-1e-5))/2e-5
            delta=np.divide(section(a)-A,slope,out=np.zeros_like(a),where=abs(slope)>1e-10)
            a=np.clip(a-np.clip(delta,-.05,.05),.00001,math.pi-theta-.02)
        t=a+theta;R=(d-2*np.cos(a))/(2*np.cos(t))
        H=np.sin(a)-(d-2*np.cos(a))*np.cos(t)/(2*(1+np.sin(t)))
        return R,H,a,vals[3],valid


def native_film():
    source=Path(__file__).with_name('meniscus_film.cpp')
    lib=source.with_name('_meniscus_film.so')
    if not lib.exists() or lib.stat().st_mtime<source.stat().st_mtime:
        compiler=shutil.which(os.environ.get('CXX','g++'))
        if compiler is None:return None
        subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    fn=ctypes.CDLL(str(lib)).film_step
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[arr]*7+[ctypes.c_int]*6+[ctypes.c_double]*9+[arr]
    fn.restype=ctypes.c_int
    return fn


class MeniscusBrush(CapillaryBrush):
    version='meniscus-contact-study-0.12.0'
    def __init__(self,canvas,settings=None,seed=606,fluid=None,meniscus=None):
        super().__init__(canvas,settings,seed,fluid or material())
        self.meniscus=meniscus or MeniscusSettings();mc=self.meniscus;f=self.fluid;n=self.settings.bunches
        if min(mc.radius_m,mc.young_pa,mc.hair_length_m,mc.wet_length_m,mc.precursor_m,mc.max_dt_s)<=0:
            raise ValueError('geometry, modulus, precursor and time step must be positive')
        if not 0<mc.contact_angle_receding_rad<=mc.contact_angle_advancing_rad<math.pi/2:
            raise ValueError('invalid advancing/receding angle interval')
        self.native=native_film();self.table=BridgeTable(mc.contact_angle_rad)
        rng=np.random.default_rng(mc.anatomy_seed)
        # A two-dimensional tapered bundle. No predefined grouping or stripe IDs.
        rr=np.sqrt((np.arange(n)+.5)/n);az=np.arange(n)*math.pi*(3-math.sqrt(5))
        self.offsets=np.c_[rr*np.cos(az),rr*np.sin(az)]+rng.normal(0,.035,(n,2))
        if mc.irregular_roots:
            pts=[]
            while len(pts)<n:
                q=rng.uniform(-1,1,2)
                if q@q<=1 and (not pts or np.min(np.linalg.norm(np.asarray(pts)-q,axis=1))>.09):pts.append(q)
            self.offsets=np.asarray(pts);rr=np.linalg.norm(self.offsets,axis=1)
        self.length=mc.hair_length_m*np.exp(rng.normal(0,mc.length_log_sd,n))
        self.radius=mc.radius_m*np.exp(rng.normal(0,mc.radius_log_sd,n))
        self.stiffness=3*mc.young_pa*(math.pi*self.radius**4/4)/self.length**3
        self.rest=.03+.48*rr**1.8+rng.uniform(-.03,.045,n)
        self.compliance=rng.uniform(.90,1.1,n)
        tipcap=math.pi*((self.radius+mc.max_coat_m)**2-self.radius**2)*mc.wet_length_m
        self.capacity=np.r_[np.full(n,f.body_capacity_m3/n),tipcap]
        self.volume=self.capacity*self.settings.initial_water;self.mass=self.volume*f.concentration_kg_m3
        self.water_initial=float(self.volume.sum());self.mass_initial=float(self.mass.sum())
        self.water_added_si=self.mass_added_si=0.
        self.neck_v=np.zeros(n);self.neck_m=np.zeros(n);self.neck_alive=np.zeros(n,dtype=bool)
        self.anchor=np.zeros((n,2));self.gap=np.zeros(n)
        self.edge_alive=np.zeros((n,n),bool);self.edge_geometry=None
        self.edge_changes=0;self.ruptures=0;self.neck_formed=0;self.film_substeps=0
        self.film_h=np.zeros_like(self.paper_water);self.film_mass=np.zeros_like(self.paper_water)
        coarse=gaussian_filter(canvas.paper.astype(float),(3.,2.))
        coarse=(coarse-coarse.mean())/max(coarse.std(),1e-12)
        self.height=mc.paper_relief_m*coarse
        self.slope_y,self.slope_x=np.gradient(self.height,f.pixel_m)
        fine=gaussian_filter(canvas.paper.astype(float),.7)
        fine=(fine-fine.mean())/max(fine.std(),1e-12)
        self.film_z=np.ascontiguousarray(mc.paper_roughness_m*fine)
        self.theta=np.ascontiguousarray(mc.paper_angle_rad*np.exp(.35*coarse))
        self.perm=np.ascontiguousarray(f.paper_permeability_m2*np.exp(.22*fine))
        self.dynamic_bounds=None;self.air_water=0.;self.after_detach_water=0.
        self.bridge_trace=[];self.snapshots=[];self._sync()
        self.pores={};self.pore_clock=1.;self.pore_transitions=0
        self.implicit_steps=0

    def _sync(self):
        n=self.settings.bunches
        self.water=np.clip(self.volume[n:]/self.capacity[n:],0,1)
        self.pigment=self.mass[n:].copy()

    def coat(self):
        n=self.settings.bunches
        return np.sqrt(self.radius**2+self.volume[n:]/(math.pi*self.meniscus.wet_length_m))-self.radius

    def _network(self,desired,dt):
        n=self.settings.bunches;f=self.fluid;mc=self.meniscus
        coords=np.c_[self.centers,self.gap/f.pixel_m]
        delta=coords[:,None,:]-coords[None,:,:]
        dist=np.maximum(np.linalg.norm(delta,axis=2)*f.pixel_m,1e-12)
        coat=self.coat();rmean=(self.radius[:,None]+self.radius[None,:])*.5
        surfacegap=np.maximum(0.,dist-2*rmean)
        old=self.edge_alive.copy()
        candidate=(surfacegap<(coat[:,None]+coat[None,:]))|old
        np.fill_diagonal(candidate,False)
        ia,ib=np.nonzero(np.triu(candidate,1));degree=np.maximum(1,candidate.sum(axis=1))
        V=(self.volume[n+ia]/degree[ia]+self.volume[n+ib]/degree[ib])*.72
        rm=rmean[ia,ib];area=V/mc.wet_length_m
        R,H,alpha,force,valid=self.table.lookup(dist[ia,ib]/rm,area/rm**2)
        valid&=dist[ia,ib]<(1+.5*mc.contact_angle_rad)*np.sqrt(area)+2*rm
        self.edge_alive[:]=False;self.edge_alive[ia,ib]=valid;self.edge_alive[ib,ia]=valid
        self.edge_changes+=np.count_nonzero(old!=self.edge_alive)//2
        ia,ib,rm,R,H,alpha,force=(v[valid] for v in (ia,ib,rm,R,H,alpha,force))
        self.edge_geometry=(ia,ib,rm,R*rm,H*rm,alpha)
        damping=self.stiffness*(mc.mechanical_tau_s+mc.friction_tau_s*self.contact)
        diagonal=damping/dt+self.stiffness
        target=np.c_[desired,self.target_z/f.pixel_m]
        mat=np.diag(diagonal);rhs=(damping/dt)[:,None]*coords+self.stiffness[:,None]*target
        # Equal-cylinder energy is applied with the pair's mean radius. The
        # linearization is an SPD attraction, retaining finite-volume breakup.
        if len(ia) and mc.cohesion:
            F=f.surface_tension_n_m*mc.wet_length_m*force
            k=np.minimum(8.,F/np.maximum(dist[ia,ib]-2.02*rm,1e-6))
            np.add.at(mat,(ia,ia),k);np.add.at(mat,(ib,ib),k)
            np.add.at(mat,(ia,ib),-k);np.add.at(mat,(ib,ia),-k)
            direction=delta[ia,ib]/(dist[ia,ib,None]/f.pixel_m)
            pull=k[:,None]*(2.02*rm/f.pixel_m)[:,None]*direction
            np.add.at(rhs,ia,pull);np.add.at(rhs,ib,-pull)
        normal_force=self.stiffness*self.normal_deflection
        if mc.cohesion:
            for key,pore in self.pores.items():
                if not pore['wet']:continue
                ids=np.array(key);xy=self.centers[ids]*f.pixel_m
                grad=.5*np.c_[xy[[1,2,0],1]-xy[[2,0,1],1],xy[[2,0,1],0]-xy[[1,2,0],0]]
                signed=np.cross(xy[1]-xy[0],xy[2]-xy[0])
                grad*=np.sign(signed)
                force_p=-abs(pore['pressure'])*mc.pore_depth_m*grad
                for axis in range(2):np.add.at(rhs[:,axis],ids,force_p[:,axis]/f.pixel_m)
        slope=np.c_[map_coordinates(self.slope_x,[self.centers[:,1],self.centers[:,0]],order=1,mode='nearest'),
                    map_coordinates(self.slope_y,[self.centers[:,1],self.centers[:,0]],order=1,mode='nearest')]
        rhs[:,:2]-=normal_force[:,None]*slope/f.pixel_m
        new=solve(mat,rhs,assume_a='pos',check_finite=False)
        # Regularized Coulomb dissipation mu*N*|x-x_old|, minimized with IRLS.
        # Normal load and fixed paper slopes deform the bundle collectively.
        for _ in range(3):
            ds=np.linalg.norm(new[:,:2]-self.centers,axis=1)*f.pixel_m
            kfr=mc.friction_coefficient*normal_force/np.sqrt(ds*ds+mc.friction_regularization_m**2)
            new[:,:2]=solve(mat+np.diag(kfr),rhs[:,:2]+kfr[:,None]*self.centers,assume_a='pos',check_finite=False)
        new[:,2]=np.maximum(0,new[:,2])
        # Nonpenetration projection of solid cores using 3-D tip separation.
        for _ in range(3):
            dd=new[:,None]-new[None,:];ds=np.linalg.norm(dd,axis=2)
            aa,bb=np.nonzero(np.triu(ds<(self.radius[:,None]+self.radius[None,:])/f.pixel_m,1))
            if len(aa)==0:break
            sep=dd[aa,bb]/np.maximum(ds[aa,bb,None],1e-12)
            corr=.35*((self.radius[aa]+self.radius[bb])/f.pixel_m-ds[aa,bb])[:,None]*sep
            np.add.at(new,aa,corr);np.add.at(new,bb,-corr)
        self.centers=new[:,:2];self.gap=np.maximum(0,new[:,2]*f.pixel_m);self.contact=self.gap<1e-8
        # Capillary pressure from the actual connected menisci, else from the
        # cylindrical coating. Negative pressure is suction in a concave bridge.
        p=f.surface_tension_n_m/(self.radius+coat)
        count=np.ones(n);ps=p.copy()
        if len(ia):
            bridge_p=-f.surface_tension_n_m/np.where(abs(R*rm)>1e-8,R*rm,1e-8)
            np.add.at(ps,ia,bridge_p);np.add.at(ps,ib,bridge_p)
            np.add.at(count,ia,1);np.add.at(count,ib,1)
        self.tip_pressure=ps/count
        if mc.lateral_flow and len(ia):
            # Poiseuille conductance of an equivalent connected channel;
            # bridge area fixes the hydraulic radius instead of a distance ramp.
            rh=np.minimum(8e-6,np.sqrt(np.maximum(H*rm,1e-12)*rm)*.18)
            G=math.pi*rh**4/(8*f.viscosity_pa_s*np.maximum(dist[ia,ib],1e-6))
            q=G*(self.tip_pressure[ia]-self.tip_pressure[ib])*dt
            conservative_exchange(self.volume,self.mass,self.capacity,np.where(q>=0,n+ia,n+ib),
                np.where(q>=0,n+ib,n+ia),np.abs(q))

    def _feed(self,dt):
        f=self.fluid;mc=self.meniscus;n=self.settings.bunches
        # Body reservoir pressure and finite capacity; no infinite ink source.
        sat=np.maximum(self.volume[:n]/self.capacity[:n],.002)
        bodyp=-2*f.surface_tension_n_m*math.cos(mc.contact_angle_rad)/35e-6/sat**.5
        # Tip drying increases retention in interstitial micro-pores.
        tips=np.maximum(self.volume[n:]/self.capacity[n:],.002)
        tipp=self.tip_pressure-2*f.surface_tension_n_m*math.cos(mc.contact_angle_rad)/18e-6/tips**.5
        conduct=f.brush_permeability_m2*math.pi*(self.radius+mc.max_coat_m)**2/(f.viscosity_pa_s*self.length)*mc.body_feed_scale
        conduct*=np.clip(sat,0,1)**3
        q=conduct*(bodyp-tipp)*dt
        a=np.arange(n);b=n+a
        conservative_exchange(self.volume,self.mass,self.capacity,np.where(q>=0,a,b),np.where(q>=0,b,a),abs(q))

    def _bounds(self,x,y,pad):
        return max(0,int(math.floor(y-pad))),min(self.canvas.height,int(math.ceil(y+pad))+1),max(0,int(math.floor(x-pad))),min(self.canvas.width,int(math.ceil(x+pad))+1)

    def _extend(self,b):
        if self.paper_bounds is None:self.paper_bounds=b
        else:
            a=self.paper_bounds;self.paper_bounds=(min(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),max(a[3],b[3]))

    def _pore_step(self,dt):
        """Moving 2-D pore network with independent liquid inventories.

        Three neighboring hairs bound a pore. Its storage is geometric void
        area times a contact-layer depth; capillary entry is 2 gamma cos(theta)/r.
        Saturation thresholds represent cooperative filling/drainage hysteresis,
        an explicit reduced closure, not a resolved three-cylinder free surface.
        Delaunay flips return old pore inventories to their hairs conservatively.
        """
        mc=self.meniscus;f=self.fluid;n=self.settings.bunches
        if not mc.pores_enabled:return 0.
        self.pore_clock+=dt
        if self.pore_clock>=.025:
            eligible=np.flatnonzero(self.gap<mc.max_coat_m)
            keys={tuple(sorted(eligible[s])) for s in Delaunay(self.centers[eligible]).simplices} if len(eligible)>=3 else set()
            for key in set(self.pores)-keys:
                pore=self.pores.pop(key)
                self.volume[n+np.array(key)]+=pore['v']/3;self.mass[n+np.array(key)]+=pore['m']/3
            for key in keys-self.pores.keys():self.pores[key]={'v':0.,'m':0.,'wet':False,'pressure':0.}
            self.pore_clock=0.
        keys=sorted(self.pores)
        if not keys:return 0.
        ids=np.array(keys);xyz=self.centers[ids]*f.pixel_m
        cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);area=.5*abs(cross)
        # Subtract the circular sectors inside the triangular solid corners.
        void=np.maximum(.04*area,area-.5*math.pi*np.mean(self.radius[ids]**2,axis=1))
        perimeter=np.linalg.norm(xyz[:,1]-xyz[:,0],axis=1)+np.linalg.norm(xyz[:,2]-xyz[:,1],axis=1)+np.linalg.norm(xyz[:,0]-xyz[:,2],axis=1)
        lengths=np.stack([np.linalg.norm(xyz[:,1]-xyz[:,0],axis=1),np.linalg.norm(xyz[:,2]-xyz[:,1],axis=1),np.linalg.norm(xyz[:,0]-xyz[:,2],axis=1)])
        circum=np.prod(lengths,axis=0)/np.maximum(4*area,1e-20)
        rp=np.maximum(3e-6,circum-np.mean(self.radius[ids],axis=1))
        cap=void*mc.pore_depth_m
        v=np.array([self.pores[k]['v'] for k in keys]);m=np.array([self.pores[k]['m'] for k in keys])
        # Compression expels excess into the adjoining wet hairs, preserving
        # both inventories; their subsequent paper bridges deliver the liquid.
        excess=np.maximum(0,v-cap);cm=m/np.maximum(v,1e-30);em=excess*cm
        for j in range(3):np.add.at(self.volume,n+ids[:,j],excess/3);np.add.at(self.mass,n+ids[:,j],em/3)
        v-=excess;m-=em
        sat=v/np.maximum(cap,1e-30)
        pc=2*f.surface_tension_n_m*math.cos(mc.contact_angle_rad)/rp
        pressure=-pc/np.sqrt(np.maximum(sat,.08))
        vv=np.r_[self.volume,v];mm=np.r_[self.mass,m];cc=np.r_[self.capacity,cap]
        source=(n+ids).ravel();dest=np.repeat(2*n+np.arange(len(keys)),3)
        conduct=np.repeat(mc.pore_permeability_m2*void/(f.viscosity_pa_s*mc.pore_depth_m*3),3)
        conduct*=np.clip(self.volume[source]/self.capacity[source],0,1)**3
        q=conduct*(self.tip_pressure[ids].ravel()-np.repeat(pressure,3))*dt
        conservative_exchange(vv,mm,cc,np.where(q>=0,source,dest),np.where(q>=0,dest,source),abs(q))
        self.volume[:]=vv[:2*n];self.mass[:]=mm[:2*n];v=vv[2*n:];m=mm[2*n:]
        total=0.
        for k,key in enumerate(keys):
            pore=self.pores[key];was=pore['wet'];S=v[k]/max(cap[k],1e-30)
            wet=S>=mc.pore_fill_saturation or (was and S>mc.pore_drain_saturation)
            self.pore_transitions+=int(was!=wet)
            pore.update(v=float(v[k]),m=float(m[k]),wet=bool(wet),pressure=float(pressure[k]))
            if not wet or max(self.gap[list(key)])>mc.max_coat_m*math.sqrt(max(0,S)):continue
            drive=max(0,2*f.surface_tension_n_m*math.cos(mc.paper_angle_rad)/f.paper_pore_radius_m+pressure[k])
            dv=min(v[k],mc.contact_permeability_m2*void[k]*drive*dt/(f.viscosity_pa_s*mc.neck_length_m))
            if dv<=0:continue
            dm=m[k]*dv/max(v[k],1e-30)
            tri=self.centers[list(key)];mid=tri.mean(axis=0);pad=np.linalg.norm(tri-mid,axis=1).max()+1
            y0,y1,x0,x1=self._bounds(*mid,pad);yy,xx=np.mgrid[y0:y1,x0:x1]
            weight=np.zeros_like(xx,dtype=float)
            edges=np.roll(tri,-1,axis=0)-tri
            for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
                px=xx+ox;py=yy+oy
                signs=[edges[j,0]*(py-tri[j,1])-edges[j,1]*(px-tri[j,0]) for j in range(3)]
                inside=((signs[0]>=0)&(signs[1]>=0)&(signs[2]>=0))|((signs[0]<=0)&(signs[1]<=0)&(signs[2]<=0))
                for j,hair in enumerate(key):inside&=(px-tri[j,0])**2+(py-tri[j,1])**2>(self.radius[hair]/f.pixel_m)**2
                weight+=inside*.25
            if weight.sum()<=0:continue
            weight/=weight.sum();self.film_h[y0:y1,x0:x1]+=dv*weight/f.pixel_m**2
            self.film_mass[y0:y1,x0:x1]+=dm*weight;self._extend((y0,y1,x0,x1))
            pore['v']-=dv;pore['m']-=dm;total+=dv
        return total

    def _deposit(self,j,dv,dm,center,radius_px,heel=None,cap=False):
        if dv<=0:return
        f=self.fluid;x,y=center
        heel=center if heel is None else heel
        pad=max(radius_px,.45)+np.linalg.norm(heel-center)
        y0,y1,x0,x1=self._bounds(x,y,pad+1)
        if x1<=x0 or y1<=y0:raise ValueError('stroke liquid left the canvas')
        yy,xx=np.mgrid[y0:y1,x0:x1];weight=np.zeros_like(xx,dtype=float)
        segment=heel-center;den=float(segment@segment)
        for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
            u=np.clip(((xx+ox-x)*segment[0]+(yy+oy-y)*segment[1])/max(den,1e-12),0,1)
            distance=np.hypot(xx+ox-x-u*segment[0],yy+oy-y-u*segment[1])
            if cap:weight+=np.maximum(0,1-(distance/max(radius_px,.1))**2)/4
            else:weight+=np.clip((radius_px+.35-distance)/.7,0,1)/4
        if weight.sum()<1e-20:
            weight[np.unravel_index(np.argmin((xx-x)**2+(yy-y)**2),xx.shape)]=1.
        weight/=weight.sum()
        self.film_h[y0:y1,x0:x1]+=dv*weight/f.pixel_m**2
        self.film_mass[y0:y1,x0:x1]+=dm*weight
        self._extend((y0,y1,x0,x1))

    def _transfer(self,dt,gate,side):
        # Returned volume tracks continuous contact flow. Rupture drops are
        # deposited conservatively but are not included in this trace channel;
        # use diagnostics()/fields for the complete material balance.
        n=self.settings.bunches;mc=self.meniscus;f=self.fluid
        coat=self.coat();new_contact=self.gap<coat
        born=new_contact&~self.neck_alive
        self.anchor[born]=self.centers[born];self.neck_alive[born]=True;self.neck_formed+=born.sum()
        # Each paper bridge has its own finite inventory and moving contact line.
        # It is not erased by contact_gate=0; gap and volume determine rupture.
        target=mc.neck_capacity_m3*np.clip(coat/mc.max_coat_m,0,1)
        if mc.contact_volume_taper:
            # The liquid contact length decreases as a bent hair is lifted.
            # The 8% distal-tip allowance is a reduced conical-tip closure;
            # shrinking contact does not destroy liquid already in a bridge.
            target*=.08+.92*np.clip(np.sqrt(np.maximum(0,self.force-self.rest)/.60),0,1)
        ids=np.flatnonzero(self.neck_alive)
        tofilm=0.
        for j in ids:
            # Supply precedes geometry. Computing mobility with an empty old
            # neck would pin its paper endpoint even while fresh ink arrives.
            wanted=max(0.,target[j]-self.neck_v[j])
            feed=min(self.volume[n+j],wanted)
            cm=self.mass[n+j]/max(self.volume[n+j],1e-30)
            self.volume[n+j]-=feed;self.mass[n+j]-=feed*cm;self.neck_v[j]+=feed;self.neck_m[j]+=feed*cm
            # While the coating intersects paper, it is an extended sliding
            # liquid contact, not a droplet tethered to one old paper point.
            # Ink left behind lives in the Eulerian film. Only after this
            # connection lifts clear does an independently retained neck remain.
            if mc.distributed_sliding_contact and new_contact[j]:self.anchor[j]=self.centers[j]
            lag=self.centers[j]-self.anchor[j]
            initial_lag=np.linalg.norm(lag)*f.pixel_m
            height=self.gap[j]+2e-6
            V=max(self.neck_v[j],1e-23)
            foot=np.cbrt(4*V/(math.pi*mc.paper_angle_rad))
            pin=f.surface_tension_n_m*2*foot*(math.cos(mc.contact_angle_receding_rad)-math.cos(mc.contact_angle_advancing_rad))
            drag=3*f.viscosity_pa_s*(2*math.pi*foot)*mc.contact_line_log_ratio/mc.paper_angle_rad
            # Implicit contact-line motion. The capillary relaxation near a
            # 2-micron gap is much faster than a hand timestep. An explicit
            # update can jump past the stable branch and falsely rupture once
            # each sampling interval, creating regular dotted trails.
            def residual(distance):
                ell=math.hypot(distance,height)
                force=f.surface_tension_n_m*math.sqrt(math.pi*V/ell)*distance/ell
                return distance+dt*max(0.,force-pin)/max(drag,1e-14)-initial_lag
            lo=0.;hi=initial_lag
            for fraction in np.linspace(0,1,25)[1:]:
                candidate=initial_lag*fraction*fraction
                if residual(candidate)>=0:hi=candidate;break
                lo=candidate
            for _ in range(16):
                middle=(lo+hi)*.5
                if residual(middle)>=0:hi=middle
                else:lo=middle
            final_lag=(lo+hi)*.5
            if initial_lag>1e-20:self.anchor[j]+=lag*(1-final_lag/initial_lag)
            lag=self.centers[j]-self.anchor[j]
            ell=math.hypot(final_lag,height)
            rupture=(1+.5*mc.contact_angle_rad)*np.cbrt(max(self.neck_v[j],1e-23))*2.2
            if ell>rupture or (not mc.retained_bridge and not new_contact[j]):
                # Geometric volume split: equal cap areas are weighted by the
                # two solid affinities. This is a closure, not resolved pinch-off.
                frac=math.cos(mc.paper_angle_rad)/(math.cos(mc.paper_angle_rad)+math.cos(mc.contact_angle_rad))
                radius=np.cbrt(4*self.neck_v[j]*frac/(math.pi*mc.paper_angle_rad))/f.pixel_m
                self._deposit(j,self.neck_v[j]*frac,self.neck_m[j]*frac,self.anchor[j],max(.4,radius),cap=True)
                self.volume[n+j]+=self.neck_v[j]*(1-frac);self.mass[n+j]+=self.neck_m[j]*(1-frac)
                self.neck_v[j]=self.neck_m[j]=0;self.neck_alive[j]=False;self.ruptures+=1
                continue
            radius=mc.neck_radius_m*np.clip(coat[j]/(mc.max_coat_m*.5),0,1.6)
            G=math.pi*radius**4/(8*f.viscosity_pa_s*(mc.neck_length_m+ell))
            suction=2*f.surface_tension_n_m*math.cos(mc.paper_angle_rad)/f.paper_pore_radius_m
            dv=min(self.neck_v[j],G*max(0.,suction+self.tip_pressure[j])*dt)
            dm=self.neck_m[j]*dv/max(self.neck_v[j],1e-30)
            self.neck_v[j]-=dv;self.neck_m[j]-=dm
            # Circular coating intersecting the plane and a laid segment of
            # solid hair. This is liquid input geometry, never final ink opacity.
            width=math.sqrt(max(0.,(self.radius[j]+coat[j])**2-(self.radius[j]+self.gap[j])**2))/f.pixel_m
            laid=side*math.sqrt(max(0.,self.force-self.rest[j]))*self.length[j]*.07/f.pixel_m
            direction=np.array([-math.sin(self.angle),math.cos(self.angle)])
            heel=self.anchor[j]-direction*laid
            self._deposit(j,dv,dm,self.anchor[j],max(.32,width),heel)
            tofilm+=dv
        # Actual interstitial liquid areas, bounded by circular menisci, also
        # supply the paper. No painted rectangle is added between neighbors.
        if self.edge_geometry is not None:
            ia,ib,radii,Rs,Hs,alphas=self.edge_geometry
            for i,j,rm,R,H,alpha in zip(ia,ib,radii,Rs,Hs,alphas):
                if max(self.gap[i],self.gap[j])>=H or H<=0:continue
                a,b=self.centers[i],self.centers[j];delta=b-a;dist=np.linalg.norm(delta)
                if dist<1e-10:continue
                mid=(a+b)*.5;direction=delta/dist;normal=np.array([-direction[1],direction[0]])
                xc=max(0.,dist*f.pixel_m*.5-rm*math.cos(alpha))
                yc=rm*math.sin(alpha)+R*math.sin(mc.contact_angle_rad+alpha)
                pad=(dist*.5+abs(H/f.pixel_m)+2)
                y0,y1,x0,x1=self._bounds(*mid,pad)
                if y1<=y0 or x1<=x0:continue
                yy,xx=np.mgrid[y0:y1,x0:x1];weight=np.zeros_like(xx,dtype=float)
                for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
                    ux=((xx+ox-mid[0])*direction[0]+(yy+oy-mid[1])*direction[1])*f.pixel_m
                    uy=((xx+ox-mid[0])*normal[0]+(yy+oy-mid[1])*normal[1])*f.pixel_m
                    localH=yc-np.sign(R)*np.sqrt(np.maximum(0.,R*R-ux*ux))
                    weight+=((abs(ux)<=xc)&(abs(uy)<=localH))*.25
                area=weight.sum()*f.pixel_m**2
                if area<=0:continue
                drive=2*f.surface_tension_n_m*math.cos(mc.paper_angle_rad)/f.paper_pore_radius_m-f.surface_tension_n_m/R
                dv=mc.contact_permeability_m2*area*max(0.,drive)*dt/(f.viscosity_pa_s*mc.neck_length_m)
                dv=min(dv,2*self.volume[n+i],2*self.volume[n+j])
                dm=0.
                for e in (i,j):
                    take=dv*.5;cm=self.mass[n+e]/max(self.volume[n+e],1e-30)
                    self.volume[n+e]-=take;self.mass[n+e]-=take*cm;dm+=take*cm
                weight/=weight.sum()
                self.film_h[y0:y1,x0:x1]+=dv*weight/f.pixel_m**2
                self.film_mass[y0:y1,x0:x1]+=dm*weight
                self._extend((y0,y1,x0,x1));tofilm+=dv
        return tofilm

    def _surface(self,dt):
        if self.paper_bounds is None:return 0.
        f=self.fluid;mc=self.meniscus
        b=self.paper_bounds;y0,y1,x0,x1=b
        b=(max(0,y0-3),min(self.canvas.height,y1+3),max(0,x0-3),min(self.canvas.width,x1+3))
        out=np.zeros(2)
        if mc.film_flow and (self.native is None or self.film_h.max()>3e-6):
            out[1]=self._implicit_film(dt)
        elif mc.film_flow:
            code=self.native(self.film_h,self.film_mass,self.paper_water,self.paper_mobile,
                self.film_z,self.theta,self.perm,self.canvas.height,self.canvas.width,*b,
                dt,f.pixel_m,f.viscosity_pa_s,f.surface_tension_n_m,mc.precursor_m,
                f.paper_porosity,f.paper_thickness_m,f.front_regularization_m,
                2*f.surface_tension_n_m*math.cos(mc.paper_angle_rad)/f.paper_pore_radius_m,out)
            if code:raise RuntimeError('film step exceeded subcycling budget')
            self.film_substeps+=int(out[0])
        else:
            sl=np.s_[b[0]:b[1],b[2]:b[3]];area=f.pixel_m**2
            depth=self.paper_water[sl]/(f.paper_porosity*area)
            dz=np.sqrt((depth+f.front_regularization_m)**2+2*self.perm[sl]*2*f.surface_tension_n_m*math.cos(mc.paper_angle_rad)/f.paper_pore_radius_m*dt/(f.paper_porosity*f.viscosity_pa_s))-f.front_regularization_m-depth
            take=np.minimum(self.film_h[sl],np.maximum(0,np.minimum(dz,f.paper_thickness_m-depth))*f.paper_porosity)
            mass=self.film_mass[sl]*take/np.maximum(self.film_h[sl],1e-30)
            self.film_h[sl]-=take;self.film_mass[sl]-=mass;self.paper_water[sl]+=take*area;self.paper_mobile[sl]+=mass
            out[1]=take.sum()*area
        self.paper_bounds=b
        self.paper_pending_s+=dt
        if self.paper_pending_s>=f.paper_tick_s:
            self.paper_step(self.paper_pending_s);self.paper_pending_s=0.
        self.paper_mass=self.paper_mobile+self.paper_fixed
        self.canvas.density=(self.paper_mass+self.film_mass)*f.extinction_m2_kg/f.pixel_m**2
        if not np.isfinite(self.canvas.density).all() or min(self.paper_mobile.min(),self.film_mass.min()) < -1e-20:
            raise FloatingPointError('invalid pigment state; rendering must be rejected')
        return float(out[1])

    def _implicit_film(self,dt):
        """Semi-implicit finite-volume pressure solve for thicker surface pools.

        Same thin-film PDE as the native explicit solver. Freeze mobility and
        implicitize capillary curvature plus the convex local wetting energy.
        Evaluate conservative face fluxes from the solved pressure; a donor
        limiter preserves positivity and advects pigment with those same fluxes.
        No clipping/rescaling of an image or of a liquid field is used.
        """
        f=self.fluid;mc=self.meniscus;area=f.pixel_m**2
        ys,xs=np.nonzero(self.film_h>1e-12)
        if not len(ys):return 0.
        b=(max(0,ys.min()-2),min(self.canvas.height,ys.max()+3),max(0,xs.min()-2),min(self.canvas.width,xs.max()+3))
        sl=np.s_[b[0]:b[1],b[2]:b[3]];shape=self.film_h[sl].shape;ny,nx=shape;N=nx*ny
        ids=np.arange(N).reshape(shape);a=np.r_[ids[:,:-1].ravel(),ids[:-1,:].ravel()];bb=np.r_[ids[:,1:].ravel(),ids[1:,:].ravel()]
        row=np.r_[a,bb,a,bb];col=np.r_[bb,a,a,bb]
        sign=np.r_[np.ones(len(a)*2),-np.ones(len(a)*2)]
        L=coo_matrix((sign/area,(row,col)),shape=(N,N)).tocsr();I=eye(N,format='csr')
        z=self.film_z[sl].ravel();theta=self.theta[sl].ravel();perm=self.perm[sl].ravel()
        water=self.paper_water[sl].ravel().copy();mass=self.paper_mobile[sl].ravel().copy()
        h=self.film_h[sl].ravel().copy();m=self.film_mass[sl].ravel().copy();absorbed=0.
        steps=max(1,math.ceil(dt/.0007));sub=dt/steps
        for _ in range(steps):
            ht=h+mc.precursor_m;rat=mc.precursor_m/ht
            potential=3*f.surface_tension_n_m*theta**2/mc.precursor_m*(rat**4-rat**7)
            deriv=3*f.surface_tension_n_m*theta**2/mc.precursor_m/ht*(-4*rat**4+7*rat**7)
            convex=np.maximum(0.,deriv)
            face=(.5*(ht[a]+ht[bb]))**3/(3*f.viscosity_pa_s)
            D=coo_matrix((np.r_[face,face,-face,-face]/area,(row,col)),shape=(N,N)).tocsr()
            A=I+sub*f.surface_tension_n_m*(D@L)-sub*(D@diags(convex))
            rhs=h+sub*(D@(potential-convex*h-f.surface_tension_n_m*(L@z)))
            trial=spsolve(A,rhs)
            pressure=-f.surface_tension_n_m*(L@(trial+z))+potential+convex*(trial-h)
            q=face*(pressure[a]-pressure[bb])*sub/area
            donor=np.where(q>=0,a,bb);receiver=np.where(q>=0,bb,a);vol=abs(q)
            outgoing=np.bincount(donor,weights=vol,minlength=N)
            scale=np.minimum(1,np.divide(np.maximum(h,0)*(1-1e-12),outgoing,out=np.ones(N),where=outgoing>0))
            vol*=scale[donor];pig=vol*m[donor]/np.maximum(h[donor],1e-30)
            np.add.at(h,donor,-vol);np.add.at(h,receiver,vol)
            np.add.at(m,donor,-pig);np.add.at(m,receiver,pig)
            depth=water/(f.paper_porosity*area)
            suction=2*f.surface_tension_n_m*math.cos(mc.paper_angle_rad)/f.paper_pore_radius_m
            dz=np.sqrt((depth+f.front_regularization_m)**2+2*perm*suction*sub/(f.paper_porosity*f.viscosity_pa_s))-f.front_regularization_m-depth
            take=np.minimum(np.maximum(h,0),np.maximum(0,np.minimum(dz,f.paper_thickness_m-depth))*f.paper_porosity)
            pig=m*take/np.maximum(h,1e-30)
            h-=take;m-=pig;water+=take*area;mass+=pig;absorbed+=take.sum()*area
            dry=h<1e-12
            water[dry]+=h[dry]*area;absorbed+=h[dry].sum()*area;mass[dry]+=m[dry]
            h[dry]=0.;m[dry]=0.
        self.film_h[sl]=h.reshape(shape);self.film_mass[sl]=m.reshape(shape)
        self.paper_water[sl]=water.reshape(shape);self.paper_mobile[sl]=mass.reshape(shape)
        self.implicit_steps+=steps
        return float(absorbed)

    def paper_step(self,dt):
        """Conservative porous-paper flux with a strictly positive donor margin."""
        if self.paper_bounds is None:return
        f=self.fluid;y0,y1,x0,x1=self.paper_bounds
        y0=max(0,y0-1);y1=min(self.canvas.height,y1+1);x0=max(0,x0-1);x1=min(self.canvas.width,x1+1)
        sl=np.s_[y0:y1,x0:x1];shape=self.paper_water[sl].shape;N=np.prod(shape)
        v=self.paper_water[sl].ravel().copy();m=self.paper_mobile[sl].ravel().copy();fixed=self.paper_fixed[sl].ravel().copy()
        ids=np.arange(N).reshape(shape);a=np.r_[ids[:,:-1].ravel(),ids[:-1,:].ravel()];bb=np.r_[ids[:,1:].ravel(),ids[1:,:].ravel()]
        cap=f.pixel_m**2*f.paper_porosity*f.paper_thickness_m;s=np.clip(v/cap,0,1)
        pressure=-2*f.surface_tension_n_m*math.cos(f.contact_angle_rad)/f.paper_pore_radius_m/np.maximum(s,.001)**(1/f.retention_lambda)
        dp=pressure[a]-pressure[bb];donor=np.where(dp>=0,a,bb);receiver=np.where(dp>=0,bb,a)
        q=f.paper_lateral_permeability_m2*f.paper_thickness_m/f.viscosity_pa_s*abs(dp)*s[donor]**(3+2/f.retention_lambda)*dt
        outgoing=np.bincount(donor,weights=q,minlength=N);incoming=np.bincount(receiver,weights=q,minlength=N)
        scale=np.minimum(1,np.maximum(v,0)*(1-1e-12)/np.maximum(outgoing,1e-100))
        dest=np.minimum(1,np.maximum(0,cap-v)/np.maximum(incoming,1e-100))
        q*=np.minimum(scale[donor],dest[receiver])
        pigment=q*np.divide(m,v,out=np.zeros(N),where=v>1e-30)[donor]
        mp=np.bincount(donor,weights=np.maximum(0,pigment),minlength=N)
        pigment*=np.minimum(1,np.maximum(m,0)*(1-1e-12)/np.maximum(mp,1e-100))[donor]
        np.add.at(v,donor,-q);np.add.at(v,receiver,q)
        np.add.at(m,donor,-pigment);np.add.at(m,receiver,pigment)
        fix=m*(-math.expm1(-f.pigment_fixation_rate_s*dt));m-=fix;fixed+=fix
        self.paper_water[sl]=v.reshape(shape);self.paper_mobile[sl]=m.reshape(shape);self.paper_fixed[sl]=fixed.reshape(shape)
        ys,xs=np.nonzero(v.reshape(shape)>1e-24)
        if len(ys):self.paper_bounds=(y0+ys.min(),y0+ys.max()+1,x0+xs.min(),x0+xs.max()+1)

    def draw_driver(self,driver,frame_callback=None):
        first=driver['samples'][0];start=np.array([first['hand_x'],first['hand_y']],float)
        cfg=self.settings;mc=self.meniscus;f=self.fluid;n=cfg.bunches
        if self.hand is None:
            self.hand=self.tip=start.copy();self.angle=first['angle_target'];self.tilt=first['tilt_target']
            side=abs(math.sin(self.tilt));radii=np.array([cfg.width*(.18+.66*side),cfg.width*.25])*math.sqrt(.07)
            co,si=math.cos(self.angle),math.sin(self.angle)
            desired=start+(self.offsets*radii)@np.array([[co,si],[-si,co]])
            self.centers=desired.copy();self.target_z=mc.compression_m*self.rest+mc.lift_height_m
            self.gap=self.target_z.copy();self.normal_deflection=np.zeros(n);self.contact[:]=False
            # Canonical airborne preparation makes the starting bundle
            # independent of hand-path temporal/spatial sampling.
            for _ in range(300):self._network(desired,.00025)
            self.anchor=self.centers.copy()
        elif not np.allclose(start,self.hand):raise ValueError('use explicit airmove')
        offset=self.time;rows=[]
        for sample in driver['samples']:
            if sample['dt']<=0:continue
            count=max(1,math.ceil(sample['dt']/mc.max_dt_s));dt=sample['dt']/count
            event=driver['events'][sample['event_index']]
            transfer=absorbed=0.
            for k in range(count):
                hand0=np.array([sample['hand_x_start'],sample['hand_y_start']]);hand1=np.array([sample['hand_x'],sample['hand_y']])
                h0=hand0+(hand1-hand0)*k/count;h1=hand0+(hand1-hand0)*(k+1)/count
                gate=sample['contact_gate_mid']>0
                _,self.tip=_follow_linear(self.tip,h0,h1,dt,cfg.tip_tau)
                _,self.force=_follow_linear(self.force,sample['pressure_target'],sample['pressure_target'],dt,cfg.pressure_tau)
                _,self.tilt=_follow_linear(self.tilt,sample['tilt_target'],sample['tilt_target'],dt,cfg.tilt_tau)
                ta=self.angle+math.atan2(math.sin(sample['angle_target']-self.angle),math.cos(sample['angle_target']-self.angle))
                _,self.angle=_follow_linear(self.angle,ta,ta,dt,cfg.angle_tau)
                _,self.spread=_follow_linear(self.spread,max(0.,self.force),max(0.,self.force),dt,cfg.spread_tau)
                side=abs(math.sin(self.tilt));radii=np.array([cfg.width*(.18+.66*side),cfg.width*.25])*math.sqrt(.07+max(0,self.spread))
                co,si=math.cos(self.angle),math.sin(self.angle)
                desired=self.tip+(self.offsets*radii)@np.array([[co,si],[-si,co]])
                paperheight=map_coordinates(self.height,[self.centers[:,1],self.centers[:,0]],order=1,mode='nearest')
                normal_position=mc.compression_m*(self.rest-self.force*self.compliance)-paperheight
                self.target_z=normal_position.copy()
                self.normal_deflection=np.maximum(0.,-normal_position)
                if not gate:self.target_z+=mc.lift_height_m
                if not gate:self.normal_deflection[:]=0.
                self.contact=self.gap<1e-8
                self._network(desired,dt);self._feed(dt)
                transfer+=self._pore_step(dt)+self._transfer(dt,gate,side);absorbed+=self._surface(dt);self._sync()
            if sample['event_end'] and event['kind']=='reload':self.reload(event['water_load'],event['ink_load'])
            self.hand=hand1;self.time=offset+sample['t'];self.targets={key:sample[key] for key in ('pressure_target','tilt_target','angle_target')}
            row={'t':self.time,'phase':event['kind'],'event':sample['event_index'],'hand':hand1.tolist(),
                 'to_film_m3':transfer,'absorbed_m3':absorbed,'neck_count':int(self.neck_alive.sum()),
                 'contact_count':int(self.contact.sum()),'wet_edges':int(self.edge_alive.sum()//2),
                 'filled_pores':sum(p['wet'] for p in self.pores.values()),
                 'retained_noncontact_bridges':int(np.count_nonzero(self.neck_alive&~self.contact)),
                 'contact_line_lag_um':float(np.linalg.norm(self.centers[self.neck_alive]-self.anchor[self.neck_alive],axis=1).mean()*f.pixel_m*1e6) if self.neck_alive.any() else 0.,
                 'tip_nl':float(self.volume[n:].sum()*1e12),'neck_nl':float(self.neck_v.sum()*1e12),
                 'film_nl':float(self.film_h.sum()*f.pixel_m**2*1e12),'force':self.force}
            self.bridge_trace.append(row);rows.append(row)
            if not gate:self.air_water+=transfer;self.after_detach_water+=absorbed
            if frame_callback:frame_callback(self.time,self)
        self.records.append({'program':driver['program'],'curves':driver.get('curves'),'trace':rows})
        return self.records[-1]

    def diagnostics(self):
        f=self.fluid
        pore_v=sum(p['v'] for p in self.pores.values());pore_m=sum(p['m'] for p in self.pores.values())
        water=float(self.volume.sum()+self.neck_v.sum()+pore_v+self.paper_water.sum()+self.film_h.sum()*f.pixel_m**2)
        pigment=float(self.mass.sum()+self.neck_m.sum()+pore_m+self.paper_mobile.sum()+self.paper_fixed.sum()+self.film_mass.sum())
        return {'version':self.version,'water_relative_error':(water-self.water_initial-self.water_added_si)/self.water_initial,
            'pigment_relative_error':(pigment-self.mass_initial-self.mass_added_si)/self.mass_initial,
            'initial_nl':self.water_initial*1e12,'added_nl':self.water_added_si*1e12,
            'initial_pigment_ng':self.mass_initial*1e12,'added_pigment_ng':self.mass_added_si*1e12,
            'brush_nl':float(self.volume.sum()*1e12),
            'neck_nl':float(self.neck_v.sum()*1e12),'pore_nl':pore_v*1e12,'pore_filling_transitions':self.pore_transitions,'paper_nl':float(self.paper_water.sum()*1e12),
            'film_nl':float(self.film_h.sum()*f.pixel_m**2*1e12),'bridge_topology_changes':int(self.edge_changes),
            'paper_bridge_formations':int(self.neck_formed),'paper_bridge_ruptures':int(self.ruptures),
            'film_substeps':self.film_substeps,'implicit_film_steps':self.implicit_steps,'air_supply_nl':self.air_water*1e12,
            'absorption_after_detachment_nl':self.after_detach_water*1e12,
            'limits':'Reduced cantilevers; mean-radius circular pair menisci; reduced paper bridges and regularized thin film. Uncalibrated. Not the complete libWetHair solver.'}
