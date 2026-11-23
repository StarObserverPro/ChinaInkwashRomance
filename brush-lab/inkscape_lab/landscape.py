"""Shape-conditioned landscape grammar. Coordinates in a 1000 x 1200 design space.

This is an original study, not a reconstruction or attribution of a historical painting.
Randomness perturbs a fixed large-scale composition; it does not decide its topology.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .brush import InkCanvas

@dataclass
class Recipe:
    round: int = 0
    seed: int = 731
    width: int = 1200
    height: int = 1440
    texture_gain: float = 1.0
    near_ink: float = .72
    far_ink: float = .14
    river_open: float = 1.0

class Painter:
    def __init__(self, recipe):
        self.recipe = recipe
        self.c = InkCanvas(recipe.width, recipe.height, seed=recipe.seed)
        self.rng = np.random.default_rng(recipe.seed)
        self.s = recipe.width / 1000
        self.sy = recipe.height / 1200
    def st(self, pts, width=2, ink=.5, water=.2, dry=.3, tag='', pressure=None, tilt=0):
        self.c.stroke([(float(x*self.s),float(y*self.sy)) for x,y in pts],
            width=width*self.s,ink=ink,water=water,dry=dry,pressure=pressure,tilt=tilt,tag=tag)
    def wash(self, pts, ink=.12, water=.75, tag='', **kwargs):
        self.c.wash([(float(x*self.s),float(y*self.sy)) for x,y in pts],ink=ink,water=water,tag=tag,**kwargs)

    def mountain(self, contour, base, ink=.4, texture='pima', count=120, tag='mountain'):
        r=self.rng; pts=np.asarray(contour,float); xmin,xmax=pts[:,0].min(),pts[:,0].max()
        ymin=pts[:,1].min(); height=base-ymin; cx=pts[np.argmin(pts[:,1]),0]
        poly=pts.tolist()+[[xmax,base],[xmin,base]]
        # A wash is an explicit broad application; its polygon is logged.
        if self.recipe.round>=2:
            self.c.wash([(float(x*self.s),float(y*self.sy)) for x,y in poly],ink=ink*.34,water=.88,tag=f'{tag}/wash',fade_bottom=height*.25*self.sy)
        else:
            self.wash(poly,ink=ink*.24,water=.80,tag=f'{tag}/wash')
        if self.recipe.round==0:
            # Baseline deliberately lacks surface-conditioned texture.
            for k in range(count):
                x=r.uniform(xmin,xmax); y=r.uniform(ymin,base)
                if y<np.interp(x,pts[:,0],pts[:,1])+4: continue
                self.st([(x,y),(x+r.normal(0,14),y+r.uniform(8,35))],width=r.uniform(1.5,5),ink=ink*.5,dry=.3,tag=f'{tag}/unstructured')
            self.st(pts.tolist(),width=2.8,ink=ink,dry=.15,tag=f'{tag}/outline')
            return
        # The contour is broken into overlaps. The air-side must not be a sealed vector path.
        for i in range(len(pts)-1):
            a,b=pts[i],pts[i+1]
            if i%5==3: continue
            mid=(a+b)/2 + r.normal(0,1.4,2)
            line=[a,mid,b]
            if self.recipe.round>=2:
                line=np.linspace(a,b,max(3,int(np.linalg.norm(b-a)/5)))
                jitter=np.cumsum(r.normal(0,.9,(len(line),2)),axis=0)
                jitter-=np.linspace(jitter[0],jitter[-1],len(line))
                line+=jitter
            self.st(line,width=r.uniform(1.1,2.5),ink=ink*.82,water=.22 if self.recipe.round>=2 else .14,dry=.35 if self.recipe.round>=2 else .55,tag=f'{tag}/gou')
        # Deliberate connected fold bands deepen only the shaded side of a face.
        if self.recipe.round>=3 and tag=='main-mountain':
            for j,(ax,ay) in enumerate([(280,268),(337,232),(419,360),(209,425)]):
                end=min(base-30,ay+height*.77)
                for k in range(9):
                    tline=np.linspace(0,1,22); yy=ay+(end-ay)*tline
                    xx=ax+k*2.6+14*np.sin(tline*3.7)+tline*(18+j*6)
                    self.st(list(zip(xx,yy)),width=7 if k<4 else 2.6,ink=ink*(.14 if k<4 else .25),water=.32,dry=.44,pressure=(.06+.85*np.sin(np.pi*tline)**.55).tolist(),tag=f'{tag}/fold')
                for k in range(52):
                    t0=r.uniform(.08,.78); xx=ax+14*np.sin(t0*3.7)+t0*(18+j*6)+r.normal(4,11); yy=ay+(end-ay)*t0
                    self.st([(xx,yy),(xx+1,yy+r.uniform(3,9))],width=r.uniform(1,3.4),ink=ink*r.uniform(.24,.57),water=.18,dry=.48,tag=f'{tag}/fold-grain')
            # A crown gathers into uneven vegetation masses rather than an edge bead-chain.
            for cc in (280,323,348,419):
                for k in range(23):
                    xx=float(np.clip(r.normal(cc,10),xmin,xmax)); yy=np.interp(xx,pts[:,0],pts[:,1])+r.uniform(1,17)
                    self.st([(xx,yy),(xx+r.uniform(-2,2),yy+r.uniform(2,5))],width=r.uniform(2,5),ink=ink*r.uniform(.5,.9),water=.22,dry=.23,tag=f'{tag}/crown')
        # Creases branch from higher ridges, never from random screen directions.
        ridges=[]
        for i in range(1,len(pts)-1):
            if pts[i,1]<pts[i-1,1] and pts[i,1]<pts[i+1,1]:
                x,y=pts[i]; end=min(base-5,y+height*r.uniform(.4,.78))
                lean=(x-cx)*.23+r.uniform(-15,15)
                line=[(x,y+4),(x+lean*.35,y+(end-y)*.32),(x+lean*.5-7,y+(end-y)*.65),(x+lean,end)]
                self.st(line,width=3.0,ink=ink*.7,water=.18,dry=.6,tag=f'{tag}/ridge')
                ridges.append((x,y,lean))
        valleys=[(pts[i,0],pts[i,1]) for i in range(1,len(pts)-1) if pts[i,1]>pts[i-1,1] and pts[i,1]>pts[i+1,1]]
        if not valleys: valleys=[(cx+(xmax-xmin)*.16,ymin+height*.2)]
        if self.recipe.round>=2:
            for vx,vy in valleys:
                for q in range(3):
                    length=(base-vy)*r.uniform(.28,.66)
                    dx=(vx-cx)*.20
                    yy=np.linspace(vy+8,min(base-8,vy+length),10)
                    xx=vx+np.linspace(0,dx,10)+np.sin(np.linspace(0,2.4,10))*7+q*4
                    self.st(list(zip(xx,yy)),width=r.uniform(9,23),ink=ink*.12,water=.68,dry=.30,pressure=np.linspace(.50,.03,10).tolist(),tag=f'{tag}/shadow')
        multiplier=1.7 if self.recipe.round>=2 and texture=='pima' else 1
        for k in range(int(count*self.recipe.texture_gain*multiplier)):
            x=r.uniform(xmin+2,xmax-2); top=np.interp(x,pts[:,0],pts[:,1]); y=r.uniform(top+3,base)
            t=(y-top)/max(1,base-top)
            # Preserve light faces, cluster marks on structurally meaningful slopes.
            band=.5+.5*np.sin((x-cx)*.055+1.2*np.sin(t*3))
            if self.recipe.round>=2:
                band=max(np.exp(-((x-(vx+(vx-cx)*.20*t+7*np.sin(t*2.4)))/max(9,(xmax-xmin)*.048))**2) for vx,vy in valleys)
                if r.random() > .10+.86*band: continue
            elif r.random() > .28+.64*band: continue
            length=r.uniform(12,65)*(1-t*.4)
            fade=1. if self.recipe.round<2 else min(1.,max(.04,(1-t)/.30))
            if self.recipe.round>=2 and texture=='axe': length*=.6
            length=min(length,base-y-2)
            if length<3: continue
            slope=(x-cx)/max(1,xmax-xmin)*.65 + .15
            if texture=='pima':
                nn=12 if self.recipe.round>=2 else 6
                yy=np.linspace(y,y+length,nn); drift=length*slope
                xx=x+np.linspace(0,drift,nn)+np.sin(np.linspace(0,r.uniform(2,4),nn))*r.uniform(2,7)
                line=list(zip(xx,yy))
                self.st(line,width=r.uniform(1.2,2.2) if self.recipe.round>=2 else r.uniform(.8,2.0),ink=ink*r.uniform(.32,.78)*fade,water=.18 if self.recipe.round>=2 else .12,dry=r.uniform(.22,.58) if self.recipe.round>=2 else r.uniform(.4,.8),tag=f'{tag}/pima')
                if band>.75 and k%3==0:
                    self.st([(xx[j]+3,yy[j]+1) for j in range(nn)],width=4.5,ink=ink*.16*fade,water=.45,dry=.48,tag=f'{tag}/ca')
            elif texture=='axe':
                dx=length*r.uniform(-.60,-.20)
                if self.recipe.round>=2:
                    dx=length*(slope-r.uniform(.2,.8))
                    self.st([(x,y),(x+dx*.2,y+length*.25),(x+dx*.7,y+length*.67),(x+dx,y+length)],width=r.uniform(2,7),ink=ink*r.uniform(.22,.65)*fade,water=.24,dry=.58,pressure=[.25,1.,.42,.01],tilt=.9,tag=f'{tag}/fupi')
                else:
                    self.st([(x,y),(x+dx*.5,y+length*.45),(x+dx,y+length)],width=r.uniform(4,12),ink=ink*r.uniform(.35,.85),water=.16,dry=.65,pressure=[.85,.65,.06],tilt=.75,tag=f'{tag}/fupi')
            else:
                self.st([(x,y),(x+r.uniform(-2,2),y+r.uniform(3,10))],width=r.uniform(1,3.4),ink=ink*.65*fade,water=.15,dry=.45,tag=f'{tag}/raindrop')
        # Dots gather near shoulders rather than coating the whole shape.
        for i in range(int(count*.13)):
            x=r.uniform(xmin,xmax); y=np.interp(x,pts[:,0],pts[:,1])+r.uniform(3,20)
            if self.recipe.round>=2:
                if i%3==0: dot_center=r.uniform(xmin+10,xmax-10)
                x=float(np.clip(dot_center+r.normal(0,8),xmin,xmax)); y=np.interp(x,pts[:,0],pts[:,1])+r.uniform(2,11)
            self.st([(x,y),(x+1.5,y+3)],width=r.uniform(2.0,4.6),ink=ink*.95,dry=.3,tag=f'{tag}/dian')

    def rock(self,x,y,w,h,ink=.65):
        self.mountain([(x-w*.5,y),(x-w*.44,y-h*.4),(x-w*.20,y-h*.82),(x+w*.05,y-h),
            (x+w*.24,y-h*.74),(x+w*.40,y-h*.71),(x+w*.55,y-h*.24)],y+7,ink,'axe',int(w*.7),'foreground-rock')
        self.st([(x-w*.40,y-h*.40),(x-w*.08,y-h*.32),(x+w*.18,y-h*.13)],width=2,ink=ink*.6,dry=.7,tag='foreground-rock/facet')
        if self.recipe.round>=3:
            # Three unequal faces: light crown, turning flank, dark root. The planes
            # remain explicit applications in the ledger, so exact replay is possible.
            flank=[(x-w*.43,y-h*.39),(x-w*.2,y-h*.67),(x-w*.03,y-h*.61),(x+w*.04,y-h*.36),(x-w*.10,y-h*.12),(x-w*.32,y+1),(x-w*.48,y+2)]
            self.wash(flank,ink*.37,.45,'foreground-rock/shaded-face')
            foot=[(x-w*.46,y-h*.11),(x-w*.16,y-h*.18),(x+w*.16,y-h*.04),(x+w*.53,y-h*.15),(x+w*.44,y+8),(x-w*.34,y+13)]
            self.wash(foot,ink*.28,.47,'foreground-rock/root-wash')
            self.st([(x-w*.44,y+4),(x-w*.14,y-1),(x+w*.1,y+9),(x+w*.44,y+4)],width=2.7,ink=ink*.8,dry=.55,tag='foreground-rock/root')
            for k in range(24):
                yy=y-self.rng.uniform(.07,.47)*h; xx=x-w*.35+(y-yy)*.16+self.rng.normal(0,w*.04)
                self.st([(xx,yy),(xx-w*.035,yy+h*.07),(xx-w*.028,yy+h*.12)],width=self.rng.uniform(1.4,4.2),ink=ink*.5,water=.2,dry=.55,pressure=[.4,.9,.05],tag='foreground-rock/flank-cun')

    def pine(self,x,y,scale=1,lean=-.3,ink=.75):
        r=self.rng; h=135*scale; w=5*scale
        trunk=np.array([(x,y),(x-lean*h*.18,y-h*.30),(x+lean*h*.35,y-h*.60),(x+lean*h*.7,y-h)])
        self.st(trunk,width=w,ink=ink,dry=.48,pressure=[.9,.82,.53,.12],tag='pine/trunk')
        self.st(trunk+np.array([3*scale,0]),width=w*.28,ink=ink*.7,dry=.75,tag='pine/bark')
        for k in range(8):
            t=.27+k*.092
            if self.recipe.round>=2: t+=r.uniform(-.027,.02)
            j=min(2,int(t*3)); f=t*3-j
            start=trunk[j]*(1-f)+trunk[j+1]*f
            side=-1 if k%2 else 1
            spread=(1-t)*h*.61*r.uniform(.8,1.25)
            end=start+np.array([side*spread,-h*r.uniform(.06,.17)])
            elbow=start+(end-start)*.55+np.array([0,6*scale])
            self.st([start,elbow,end],width=w*.42,ink=ink*.9,dry=.48,pressure=[.8,.4,.08],tag='pine/branch')
            for q in range(3):
                center=end+np.array([r.normal(0,10*scale),r.normal(0,4*scale)])
                clump=r.uniform(.65,1.5) if self.recipe.round>=3 else 1.
                n=15 if self.recipe.round>=2 else 17
                for i in range(n):
                    a=r.uniform(-np.pi*.97,-np.pi*.03); length=r.uniform(8,19)*scale*clump
                    tip=center+np.array([np.cos(a)*length,np.sin(a)*length*.7])
                    self.st([center,tip],width=r.uniform(.6,1.1)*scale*(1.3 if self.recipe.round>=3 else 1.),ink=ink*r.uniform(.52,.9),dry=.30,water=.12,pressure=[.68,.05],tag='pine/needle')

    def distant_grove(self, x,y,n=15,scale=1,ink=.32):
        r=self.rng
        for i in range(n):
            xx=x+i*7*scale+r.uniform(-3,3)*scale; yy=y+r.normal(0,3)*scale; h=r.uniform(14,27)*scale
            self.st([(xx,yy),(xx+r.normal(0,2),yy-h)],width=.9*scale,ink=ink,dry=.25,tag='grove/trunk')
            for k in range(5):
                yy0=yy-h+k*h*.15; ww=(k+1)*1.5*scale
                self.st([(xx-ww,yy0+3),(xx,yy0),(xx+ww,yy0+4)],width=2.5*scale,ink=ink*.8,dry=.2,tag='grove/foliage')

    def pavilion(self,x,y,s=1):
        # Paired horizontal strokes and a roof are scale cues, not text ornaments.
        self.st([(x-22*s,y),(x,y-13*s),(x+26*s,y+2*s)],width=2*s,ink=.6,dry=.3,tag='pavilion/roof')
        self.st([(x-27*s,y+3*s),(x-15*s,y+2*s),(x+20*s,y+5*s),(x+29*s,y+2*s)],width=1.7*s,ink=.58,dry=.28,tag='pavilion/eaves')
        for dx in [-16,16]: self.st([(x+dx*s,y+4*s),(x+dx*s,y+28*s)],width=1.4*s,ink=.55,dry=.2,tag='pavilion/pillar')
        self.st([(x-22*s,y+29*s),(x+25*s,y+30*s)],width=1.5*s,ink=.6,dry=.3,tag='pavilion/base')

    def paint(self):
        p=self.recipe; r=self.rng
        # Far -> middle -> near. White rivers/mist are unpainted paper, not white ink.
        self.mountain([(260,392),(291,345),(325,352),(360,269),(393,288),(428,251),(467,313),(506,319),
            (552,399),(602,385),(641,435)],493,p.far_ink,'pima',160,'far-left')
        self.mountain([(574,463),(611,421),(651,435),(680,381),(713,392),(750,352),(785,413),(834,421),(882,480)],537,p.far_ink*.8,'pima',120,'far-right')
        main=[(96,650),(127,567),(153,563),(175,482),(190,404),(215,417),(232,328),(256,309),
              (269,238),(294,264),(316,199),(338,231),(359,230),(387,314),(408,357),(428,340),
              (449,434),(475,468),(505,523),(528,599),(546,638)]
        if p.round>=2:
            main=[(96,650),(125,571),(148,559),(172,477),(188,427),(201,411),(222,408),(232,341),
                  (247,321),(266,313),(269,264),(284,254),(309,257),(322,229),(337,224),(357,233),
                  (368,262),(378,264),(393,327),(409,361),(428,349),(444,402),(450,446),
                  (477,470),(501,523),(522,597),(546,654)]
        self.mountain(main,716 if p.round>=2 else 696,.59 if p.round>=3 else (.51 if p.round>=2 else .40),'pima',760,'main-mountain')
        # Smaller face joins the main mass without competing with its summit.
        self.mountain([(83,748),(112,683),(143,678),(169,613),(199,600),(226,552),(252,579),(279,629),
                       (306,612),(334,661),(366,704),(397,726)],780,.46,'pima',420,'left-shoulder')
        # A mist channel separates banks.
        bank_y=737 if p.round<2 else 754
        self.mountain([(536,bank_y),(568,bank_y-35),(591,bank_y-79),(620,bank_y-86),
             (652,bank_y-60),(686,bank_y-115),(723,bank_y-90),(755,bank_y-35),(795,bank_y-17),(827,bank_y+11)],bank_y+39,.32,'axe',210,'middle-right')
        self.distant_grove(549,bank_y+9,26,.79,.35)
        self.pavilion(710,bank_y-19,.76)
        # Left bank is slanted, water leads the eye around it toward the pavilion.
        shore=[(83,855),(123,821),(163,814),(216,805),(248,827),(306,831),(345,854),(389,864),(407,879),(360,894),(302,894),(238,916),(184,945),(113,956)]
        self.wash(shore,.10 if p.round>=3 else .16,.72,'near-bank/wash',fade_bottom=80*self.sy if p.round>=3 else 0)
        for offset in (0,7,14):
            self.st([(x,y+offset) for x,y in shore[3:10]],width=1.8,ink=.50-offset*.009,dry=.60,tag='near-bank/shore')
        self.rock(150,852,139,106,.63); self.rock(249,858,145,80,.54)
        self.pine(176,840,1.40,-.40,.80); self.pine(226,847,1.03,.42,.69)
        # Foreground right rocks open toward water, not across it.
        self.rock(770,1059,223,166,p.near_ink)
        self.rock(898,1120,180,165,p.near_ink*.9)
        self.rock(664,1090,132,82,p.near_ink*.85)
        self.pine(793,1007 if p.round>=3 else 976,1.53,-.35,p.near_ink)
        self.pine(860,1056 if p.round>=3 else 1020,1.18,.35,p.near_ink*.93)
        if p.round>=3:
            for gx,gy in [(140,872),(274,873),(722,1094),(899,1141)]:
                for k in range(12):
                    xx=gx+r.normal(0,8); yy=gy+r.normal(0,3)
                    self.st([(xx,yy),(xx+r.uniform(-8,8),yy-r.uniform(8,22))],width=.85,ink=.55,dry=.35,pressure=[.85,.04],tag='foreground/grass')
        # Distant small boat; asymmetric water segments never become a full grid.
        self.st([(483,921),(500,926),(524,922)],width=1.5,ink=.48,dry=.36,tag='boat/hull')
        self.st([(502,919),(502,912),(505,910)],width=1.7,ink=.51,dry=.25,tag='boat/person')
        self.st([(507,916),(522,904)],width=.8,ink=.42,dry=.25,tag='boat/pole')
        for i in range(21 if p.round<2 else 12):
            y=r.uniform(802,1097); x=r.uniform(363,654); length=r.uniform(12,56)
            if 860<y<917 and x<435: continue
            self.st([(x,y),(x+length*.5,y+r.uniform(-1,1)),(x+length,y-1)],width=.7,ink=.14,dry=.70,tag='water/ripple')
        return self.c

def render(recipe,output):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    painter=Painter(recipe); canvas=painter.paint(); canvas.save(output)
    log={'recipe':asdict(recipe),'width':recipe.width,'height':recipe.height,
         'coordinate_system':'pixels, origin top-left','paper_color':list(canvas.paper_color),'renderer_version':'0.1.0','strokes':canvas.strokes}
    output.with_name(output.stem+'_strokes.json').write_text(json.dumps(log,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    return canvas
