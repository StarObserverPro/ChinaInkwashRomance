"""v0.16 structural transfer; retained contact-line and pair geometry.
Supply and rupture alternatives are selected by StructuralBrush.
"""
import math
import numpy as np

def structural_transfer(self,dt,gate,side):
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
        # Finite hydraulic supply: the annular coating is represented by
        # an equivalent throat. Short paper contacts cannot instantaneously
        # acquire a full liquid bridge. All liquid remains in tip/neck stores.
        lag0=np.linalg.norm(self.centers[j]-self.anchor[j])*f.pixel_m
        path=mc.neck_length_m+math.hypot(self.gap[j]+2e-6,lag0)
        hydraulic_radius=min(4e-6,self.technique.supply_hydraulic_fraction*coat[j])
        conduct=math.pi*hydraulic_radius**4/(8*f.viscosity_pa_s*path)
        neck_radius=max(2e-6,np.cbrt(3*max(self.neck_v[j],1e-23)/(4*math.pi)))
        neck_pressure=-2*f.surface_tension_n_m*math.cos(mc.contact_angle_rad)/neck_radius
        available=conduct*max(0.,self.tip_pressure[j]-neck_pressure)*dt
        feed=min(self.volume[n+j],wanted,available) if self.structure.supply=='finite' else min(self.volume[n+j],wanted)
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
            self._release(j)
            continue
        G=self._neck_conductance(j,coat[j],mc.neck_length_m+ell)
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
