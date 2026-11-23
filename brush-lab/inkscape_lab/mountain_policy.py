"""Tutorial-derived authored priors, not a learned historical attribution."""
import numpy as np
NAMES={'outline':'勾勒','hemp_long':'长披麻','hemp_short':'短披麻','rope':'解索','fold':'折带','cloud':'云头','axe':'斧劈','mi':'米点','rub':'擦','wash':'染','moss':'点苔'}
FAMILIES=list(NAMES)
# tool, compression mm, tilt degrees, saturation, concentration kg/m3, speed mm/s, length mm
CONTROLS={
 'outline':('fine',.105,15,.32,35,24,(1.3,4.3)),
 'hemp_long':('fine',.145,30,.30,26,25,(1.8,4.8)),
 'hemp_short':('fine',.135,32,.28,31,27,(.65,2.2)),
 'rope':('fine',.145,34,.30,27,23,(1.3,3.8)),
 'fold':('fine',.18,58,.28,30,25,(1.1,3.7)),
 'cloud':('fine',.18,45,.34,24,24,(1.2,3.2)),
 'axe':('side',.30,72,.34,24,29,(.9,3.4)),
 'mi':('side',.23,72,.55,17,24,(.35,1.3)),
 'rub':('side',.28,78,.18,32,30,(1.1,3.1)),
 'wash':('wash',.31,76,.62,4,26,(1.7,4.4)),
 'moss':('fine',.22,56,.36,38,25,(.16,.48)),
}
RULES=[
 ('圆缓',{'hemp_long':1.2,'hemp_short':.7,'cloud':.5},'圆缓坡体优先顺坡披麻，曲面可回转'),
 ('纤维',{'hemp_long':1.3,'hemp_short':1.0,'rope':.6},'成组细长纹理提高披麻与解索的候选概率'),
 ('平折',{'fold':1.8,'outline':.5,'rub':.3},'横向层面和折角优先折带、勾勒与侧擦'),
 ('硬朗',{'axe':1.5,'fold':.7,'outline':.5},'硬质岩面提高斧劈与折带概率'),
 ('峭立',{'axe':.9,'outline':.6},'峭壁强调有受力边的侧锋拖收'),
 ('盘曲',{'rope':1.4,'cloud':1.0,'hemp_short':.6},'回旋纹理增加解索、云头与短皴'),
 ('稠密',{'hemp_short':.7,'rub':.7,'moss':.2},'密集区域优先短皴搭接，不以均匀喷点填满'),
 ('烟雨',{'mi':1.7,'wash':1.3,'outline':-.8},'湿润云山提高米点与淡染，降低硬勾'),
 ('疏朗',{'outline':.3,'wash':-.9,'moss':-.7},'疏朗画面减少铺满与点苔'),
 ('雪景',{'outline':.7,'axe':.6,'wash':-1.3},'以相邻墨面衬托雪面，保留白处'),
]
def prior(tags,coherence,ridge,tone,phase,neutral=False,contrast=.1):
 z=np.zeros(len(FAMILIES))
 if not neutral:
  for tag,boost,why in RULES:
   if tag in tags:
    for f,b in boost.items():z[FAMILIES.index(f)]+=b
 z-=z.max();p=.015+.835*np.exp(z)/np.exp(z).sum()
 p[FAMILIES.index('wash')]*=(.02+np.exp(-(ridge/.045)**2))*max(.02,1-phase/.45)
 smooth=float(np.clip((.075-contrast)/.060,0,1)*np.clip(tone/.10,0,1))
 p[FAMILIES.index('wash')]*=1+45*smooth
 p[FAMILIES.index('rub')]*=1+2*smooth
 p[FAMILIES.index('hemp_long')]*=1-.75*smooth
 p[FAMILIES.index('outline')]*=1-.8*smooth
 p[FAMILIES.index('moss')]*=max(.04,1-coherence)*(.3+tone)
 if '烟雨' not in tags:p[FAMILIES.index('mi')]*=.2
 for f in ['hemp_long','rope','outline']:p[FAMILIES.index(f)]*=.4+coherence
 return p/p.sum()
def explain(f,tags,coherence,ridge):
 fired=[why for tag,boost,why in RULES if tag in tags and boost.get(f,0)>0]
 return '；'.join(fired[:2])+f'；局部方向一致性 {coherence:.2f}，线脊响应 {ridge:.3f}'
