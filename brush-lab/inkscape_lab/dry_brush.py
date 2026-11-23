"""v0.9 study: correlated material heterogeneity instead of per-frame noise.

An explicitly phenomenological model. Irregular tuft domains, unequal local
water/ink, and smooth seeded fields along accumulated contact travel control
contact and conductance. No trajectory jitter or post-render erasing occurs.
This is not a 3D capillary/cohesion solver or a calibrated paper model.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .reference_iteration import JoinedContactBrush


class TravelField:
    """Seeded C1 field with irregular knot spacing, queried by distance.

    The complete field is sampled once. Evaluation consumes no random numbers,
    so changing frame rate or integration step does not change the realization.
    The length scale is in paper pixels, not integration iterations.
    """
    def __init__(self, seed, channels, scale, length=32768.):
        if channels<1 or int(channels)!=channels or not np.isfinite(scale) or scale<=0 or not np.isfinite(length) or length<=0:
            raise ValueError('positive channels, scale and length required')
        rng=np.random.default_rng(seed)
        positions=[0.]
        while positions[-1]<length:
            positions.append(positions[-1]+scale*rng.uniform(.45,1.85))
        self.positions=np.asarray(positions)
        self.values=1.35*np.tanh(rng.normal(0.,1.,(len(positions),channels)))

    def at(self, distance):
        if not np.isfinite(distance) or not 0.<=distance<=self.positions[-1]:
            raise ValueError('contact travel outside the field range')
        i=min(len(self.positions)-2,int(np.searchsorted(self.positions,distance,side='right')-1))
        u=(distance-self.positions[i])/(self.positions[i+1]-self.positions[i])
        blend=u*u*(3.-2.*u)
        return (1.-blend)*self.values[i]+blend*self.values[i+1]


class CorrelatedDryBrush(JoinedContactBrush):
    version='correlated-dry-contact-study-0.9.0'

    def __init__(self, canvas, settings=None, seed=606, texture_seed=909, paper_seed=50509,
                 correlated=True, new_paper=True):
        super().__init__(canvas,settings,seed)
        self.texture_seed=int(texture_seed)
        self.paper_seed=int(paper_seed)
        self.correlated=bool(correlated)
        self.new_paper=bool(new_paper)
        rng=np.random.default_rng(self.texture_seed)
        n=self.settings.bunches
        self.domains=min(7,max(1,n//3))
        g=self.domains
        counts=np.full(g,2)+rng.multinomial(n-2*g,rng.dirichlet(np.full(g,2.)))
        self.domain_ids=np.repeat(np.arange(g),counts)
        centers=np.sort((np.linspace(-.84,.84,g) if g>1 else np.zeros(1))+rng.uniform(-.075,.075,g))
        widths=rng.uniform(.045,.12,g)
        offsets=[]
        for i,count in enumerate(counts):
            transverse=centers[i]+widths[i]*np.linspace(-1.,1.,count)+rng.normal(0.,.009,count)
            axial=rng.normal(rng.uniform(-.16,.16),.045,count)
            offsets.extend(zip(transverse,axial))
        self.base_offsets=np.asarray(offsets)
        self.domain_centers=centers
        self.offsets=self.base_offsets.copy()
        self.base_rest=.04+.17*np.abs(self.base_offsets[:,0])**3+rng.uniform(-.018,.018,n)
        self.rest=self.base_rest.copy()
        self.domain_bias=rng.uniform(-.13,.09,g)
        self.patch_scale=np.clip(rng.lognormal(-.13,.20,g)[self.domain_ids],.60,1.23)

        # Keep the exact initial TOTAL ink and water; change their local
        # allocation. This is part of brush anatomy, not extra ink injection.
        pigment_total=float(self.pigment.sum())
        self.pigment*=np.exp(rng.normal(0.,.45,g)[self.domain_ids])
        if pigment_total>0.:
            self.pigment*=pigment_total/float(self.pigment.sum())
        water_total=float(self.water.sum())
        logits=rng.normal(0.,.70,g)[self.domain_ids]+rng.normal(0.,.10,n)
        low,high=-20.,20.
        for _ in range(60):
            shift=(low+high)/2.
            w=1./(1.+np.exp(-(logits+shift)))
            if float(w.sum())<water_total:low=shift
            else:high=shift
        self.water=1./(1.+np.exp(-(logits+(low+high)/2.)))
        self.initial_water_total=float(self.water.sum())
        self.initial_pigment_total=float(self.pigment.sum())

        self.coarse=TravelField(self.texture_seed+1,g,46.)
        self.medium=TravelField(self.texture_seed+2,g,13.)
        self.fine=TravelField(self.texture_seed+3,n,7.)
        self.contact_travel=0.
        self.mid_travel=0.
        self.domain_noise=np.zeros(g)
        self.bunch_noise=np.zeros(n)
        self.mid_supply=np.ones(n)
        self.domain_log=[]
        if self.new_paper:
            prng=np.random.default_rng(self.paper_seed)
            field=np.zeros(canvas.density.shape,dtype=float)
            for scale,weight in [(.70,.090),(2.4,.110),(5.0,.085)]:
                layer=gaussian_filter(prng.normal(size=field.shape),scale)
                layer=(layer-layer.mean())/max(float(layer.std()),1e-12)
                field+=weight*layer
            self.tooth_height=np.clip(.48+field,0.,1.)
            # One depth convention in both the bundle gate and pixel contact:
            # larger depth is harder to reach. v0.8 used opposite signs here.
            self.relief=1.-self.tooth_height

    def _paper_contact(self, yy, xx, compression, wet):
        if not self.new_paper:
            return super()._paper_contact(yy,xx,compression,wet)
        depth=map_coordinates(self.tooth_height,[yy.ravel(),xx.ravel()],
                              order=1,mode='constant',cval=.5).reshape(xx.shape)
        reach=.18+.48*wet+.37*math.sqrt(compression)
        return np.clip((reach-depth+.035)/.07,0.,1.)

    def state(self):
        state=super().state()
        state['dry_material']={
            'texture_seed':self.texture_seed,'paper_seed':self.paper_seed,
            'correlated':self.correlated,'new_paper':self.new_paper,
            'contact_travel':self.contact_travel,'last_mid_travel':self.mid_travel,
            'last_mid_domain_noise':self.domain_noise.tolist(),
            'last_mid_bunch_noise':self.bunch_noise.tolist(),
            'last_mid_supply':self.mid_supply.tolist(),
            'last_mid_offsets':self.offsets.tolist(),'last_mid_rest':self.rest.tolist()}
        return state

    def _material_step(self, dt, hand0, hand1, gate, force_mid):
        distance=float(np.linalg.norm(hand1-hand0)) if gate else 0.
        self.mid_travel=self.contact_travel+.5*distance
        self.contact_travel+=distance
        if self.correlated:
            self.domain_noise=.75*self.coarse.at(self.mid_travel)+.60*self.medium.at(self.mid_travel)
            self.bunch_noise=self.fine.at(self.mid_travel)
        else:
            # Static heterogeneity ablation. No per-frame white noise is used.
            self.domain_noise=np.zeros(self.domains)
            self.bunch_noise=np.zeros(self.settings.bunches)
        ids=self.domain_ids
        group_wet=np.bincount(ids,weights=self.water,minlength=self.domains)/np.bincount(ids,minlength=self.domains)
        # Domains gently change width/relative placement as the tuft travels.
        # The average displacement is zero: the commanded path never jitters.
        local=self.base_offsets[:,0]-self.domain_centers[ids]
        expansion=np.exp(.32*self.domain_noise[ids])
        shift=.055*(1.-group_wet[ids])*min(1.,max(0.,force_mid))*self.domain_noise[ids]
        delta=local*(expansion-1.)+shift
        delta-=delta.mean()
        self.offsets=self.base_offsets.copy()
        self.offsets[:,0]+=delta
        self.rest=self.base_rest-.045*(1.-self.water)*self.domain_noise[ids]
        self.mid_supply[:]=0.
        self.domain_log.append({'travel':self.mid_travel,'gate':bool(gate),
            'noise':self.domain_noise.tolist(),'water':group_wet.tolist()})

    def _bunch_footprint(self,index,center,compression,wet,side,angle):
        group=self.domain_ids[index]
        accessibility=(.85*wet+.55*compression+.40*self.domain_noise[group]
                       +.14*self.bunch_noise[index]+self.domain_bias[group]-.14)
        supply=float(np.clip(accessibility/.20,0.,1.))
        self.mid_supply[index]=supply
        if supply<=0.:
            return math.floor(center[0]),math.floor(center[1]),np.zeros((1,1)),0.,0.
        x0,y0,kernel,area,coverage=super()._bunch_footprint(index,center,compression,wet,side,angle)
        # Reduce actual transfer before normalizing/debiting, so failed supply
        # retains pigment. This never edits an already deposited image.
        return x0,y0,kernel*supply,area*supply,coverage*supply

    def diagnostics(self):
        return {**super().diagnostics(),'domains':self.domains,
            'contact_travel':self.contact_travel,'texture_seed':self.texture_seed,
            'paper_seed':self.paper_seed,'model':'Uncalibrated correlated conductance and grouped tuft geometry; not capillary fluid or 3D bristle simulation.'}


__all__=['TravelField','CorrelatedDryBrush']
