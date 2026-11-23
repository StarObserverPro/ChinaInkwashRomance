"""v0.13: v0.12 brush contact coupled to stratified porous paper."""
import numpy as np
from .meniscus import MeniscusBrush
from .paper_transport import PaperTransport


class AbsorbentBrush(MeniscusBrush):
    version='stratified-paper-study-0.13.0'

    def __init__(self,*args,paper=None,record_source=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.substrate=PaperTransport(self.canvas.density.shape,self.fluid,paper)
        self.film_h=self.substrate.film;self.film_mass=self.substrate.film_mass
        self.pending_paper_s=0.;self.record_source=record_source;self.source_records=[]
        self.previous_film=self.film_h.copy();self.previous_mass=self.film_mass.copy()
        self.deposited_water=0.;self.deposited_mass=0.;self.substrate_trace=[]
        self.empty_perm=np.zeros_like(self.perm)

    def _sync_paper(self):
        p=self.substrate
        self.paper_water=p.water.sum(axis=0)
        self.paper_mobile=(p.fine+p.coarse).sum(axis=0)
        self.paper_fixed=p.fixed.sum(axis=0);self.paper_mass=self.paper_mobile+self.paper_fixed
        self.canvas.density=p.density()

    def _film_only(self,dt):
        """Keep the v0.12 free-surface solver; disable its full-depth sink."""
        if not np.any(self.film_h>0):return 0.
        saved=(self.paper_water,self.paper_mobile,self.perm,self.paper_bounds)
        self.paper_water=np.zeros_like(self.film_h);self.paper_mobile=np.zeros_like(self.film_h)
        self.perm=self.empty_perm
        yy,xx=np.nonzero(self.film_h>0)
        self.paper_bounds=(max(0,int(yy.min())-3),min(self.canvas.height,int(yy.max())+4),max(0,int(xx.min())-3),min(self.canvas.width,int(xx.max())+4))
        if self.native is None or self.film_h.max()>3e-6:
            absorbed=self._implicit_film(dt)
        else:
            f=self.fluid;mc=self.meniscus;out=np.zeros(2)
            code=self.native(self.film_h,self.film_mass,self.paper_water,self.paper_mobile,
                self.film_z,self.theta,self.perm,self.canvas.height,self.canvas.width,*self.paper_bounds,
                dt,f.pixel_m,f.viscosity_pa_s,f.surface_tension_n_m,mc.precursor_m,
                f.paper_porosity,f.paper_thickness_m,f.front_regularization_m,
                self.substrate.suction,out)
            if code:raise RuntimeError('surface film step failed')
            self.film_substeps+=int(out[0]);absorbed=float(out[1])
        # The film solver's vanishing-film cleanup moves tiny residuals into
        # temporary arrays. Place them in available layer volume, conservatively.
        v=self.paper_water;m=self.paper_mobile;p=self.substrate
        for l in range(p.layers):
            take=np.minimum(v,np.maximum(0,p.capacity[l]-p.water[l]))
            pig=m*np.divide(take,v,out=np.zeros_like(v),where=v>0)
            p.water[l]+=take;p.fine[l]+=pig*p.settings.fine_fraction
            p.coarse[l]+=pig*(1-p.settings.fine_fraction);v-=take;m-=pig
        self.film_h+=v/self.fluid.pixel_m**2;self.film_mass+=m
        self.paper_water,self.paper_mobile,self.perm,self.paper_bounds=saved
        return absorbed-float(v.sum())

    def flush_paper(self):
        if self.pending_paper_s<=0:return 0.
        dt=self.pending_paper_s;self.pending_paper_s=0.
        absorbed=self.substrate.advance(dt*.5)
        absorbed+=self._film_only(dt)
        absorbed+=self.substrate.advance(dt*.5)
        self._sync_paper()
        return absorbed

    def _surface(self,dt):
        # Tape records actual conservative source increments before substrate
        # evolution. It includes rupture drops omitted from the old flow trace.
        dv=(self.film_h-self.previous_film)*self.fluid.pixel_m**2
        dm=self.film_mass-self.previous_mass
        self.deposited_water+=float(dv.sum());self.deposited_mass+=float(dm.sum())
        if self.record_source:
            ids=np.flatnonzero((dv.ravel()!=0)|(dm.ravel()!=0))
            self.source_records.append((float(dt),ids.astype(np.int32),dv.ravel()[ids].copy(),dm.ravel()[ids].copy()))
        self.pending_paper_s+=dt;absorbed=0.
        if self.pending_paper_s>=self.substrate.settings.surface_tick_s-1e-12:
            absorbed=self.flush_paper()
        self.previous_film[:]=self.film_h;self.previous_mass[:]=self.film_mass
        return absorbed

    def draw_driver(self,*args,**kwargs):
        result=super().draw_driver(*args,**kwargs)
        self.flush_paper()
        self.previous_film[:]=self.film_h;self.previous_mass[:]=self.film_mass
        return result

    def relax(self,duration,callback=None,dt=.012):
        self.flush_paper();elapsed=0.
        while elapsed<duration-1e-12:
            sub=min(dt,duration-elapsed);self.pending_paper_s=sub;self.flush_paper()
            elapsed+=sub;self.time+=sub
            if callback:callback(self.time,self)
        self.previous_film[:]=self.film_h;self.previous_mass[:]=self.film_mass

    def save_source(self,path):
        rec=self.source_records
        np.savez_compressed(path,dt=np.array([r[0] for r in rec]),
            offsets=np.r_[0,np.cumsum([len(r[1]) for r in rec])],
            ids=np.concatenate([r[1] for r in rec]),water=np.concatenate([r[2] for r in rec]),
            pigment=np.concatenate([r[3] for r in rec]),shape=self.film_h.shape)

    def diagnostics(self):
        d=super().diagnostics()
        if hasattr(self,'substrate'):
            d['substrate']=self.substrate.totals()
            d['deposited_nl_including_rupture']=self.deposited_water*1e12
            d['limits']='Uncalibrated reduced brush with layered porous paper. Retention and filtration closures are not measurements of xuan paper.'
        return d
