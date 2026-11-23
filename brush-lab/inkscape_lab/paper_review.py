"""Evidence plates built directly from simulated density and water fields."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageSequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .brush import InkCanvas
from .paper_studies import label

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper-review'
PAPER=(245,240,225)
BG=(246,244,236)
STROKE='paper-stroke-v013'
DROP='paper-drop-v013'

def load(name,file='run.json'):
    return json.loads((ROOT/name/file).read_text())

def render(density):
    c=InkCanvas(density.shape[1],density.shape[0],seed=505,paper_color=PAPER)
    c.density=density;return c.to_image()

def main():
    OUT.mkdir(exist_ok=True)
    source=np.load(ROOT/'paper-stroke-r1/source.npz')
    new=np.load(ROOT/STROKE/'paper_fields.npz')
    old=np.load(ROOT/'paper-stroke-legacy/fields.npz')
    newmeta=load(STROKE,'paper_run.json');oldmeta=load('paper-stroke-legacy')
    source_v=float(source['water'].sum());source_m=float(source['pigment'].sum())
    nv=float(new['water'].sum()+new['film'].sum()*material_pixel()**2)
    nm=float(new['fine'].sum()+new['coarse'].sum()+new['fixed'].sum()+new['film_mass'].sum())
    balance={'source_water_nl':source_v*1e12,'source_pigment_ng':source_m*1e12,
        'paper_water_relative_error':nv/source_v-1,'paper_pigment_relative_error':nm/source_m-1}
    newmeta['paper_source_balance']=balance
    newmeta['replay_note']='Brush/source mechanics were recorded once in paper-stroke-r1/source.npz. In this replay the inherited brush reservoir is unused; assess the separate paper_source_balance.'
    (ROOT/STROKE/'paper_run.json').write_text(json.dumps(newmeta,indent=2)+'\n')
    summary={'balance':balance,'legacy_post_lift_density_l1':oldmeta['diagnostics']['post_lift_density_relative_l1'],
        'new_post_lift_density_l1':newmeta['post_lift_density_relative_l1'],
        'legacy_drop_final':load('paper-legacy')['trace'][-1],
        'new_drop_final':load(DROP)['trace'][-1],
        'validation':load('paper-validation','validation.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    comparison=Image.new('RGB',(1380,985),BG)
    label(comparison,(30,18),'v0.13 | Same brush input, different paper transport',30)
    panels=[('paper-stroke-legacy',old['density'],'v0.12 paper'),(STROKE,new['density'],'v0.13 stratified fiber paper')]
    for i,(name,density,title) in enumerate(panels):
        x=30+i*675;label(comparison,(x,75),title,25)
        at_lift=np.load(ROOT/name/'lift_state.npz')['density']
        label(comparison,(x,113),'Brush lifted | t = 2.41 s',17)
        comparison.paste(render(at_lift),(x,141))
        label(comparison,(x,353),'6 s later | no further ink supplied',17)
        final=render(density);comparison.paste(final,(x,381))
        label(comparison,(x,603),'Pause region (4x)',19)
        label(comparison,(x+300,603),'Lift region (4x)',19)
        comparison.paste(final.crop((302,78,362,137)).resize((240,236),Image.Resampling.NEAREST),(x,641))
        comparison.paste(final.crop((392,36,465,98)).resize((292,248),Image.Resampling.NEAREST),(x+300,641))
    label(comparison,(30,926),f"Both receive {source_v*1e12:.3f} nL and {source_m*1e12:.3f} ng. Common optical mapping; no image blur.",17)
    label(comparison,(30,953),'These are wet-state simulations, not calibrated predictions for a measured sheet of xuan paper.',17)
    comparison.save(OUT/'paper_comparison.png')

    iterations=Image.new('RGB',(660,1185),BG)
    label(iterations,(20,14),'Same input | visual iterations of the paper model',23)
    versions=[('paper-stroke-legacy','fields.npz','v0.12 | frozen ribbons'),
        ('paper-stroke-r1','paper_fields.npz','R1 | stratified transport'),
        ('paper-stroke-final','paper_fields.npz','R2 | excessive axial texture'),
        ('paper-stroke-fibers','paper_fields.npz','R3 | excessive fiber contrast (rejected)'),
        (STROKE,'paper_fields.npz','R4 / v0.13 | reduced material contrast')]
    for i,(folder,file,title) in enumerate(versions):
        y=55+i*222;label(iterations,(20,y),title,20)
        iterations.paste(render(np.load(ROOT/folder/file)['density']),(20,y+27))
    iterations.save(OUT/'paper_iterations.png')

    plate=Image.new('RGB',(1420,630),BG)
    label(plate,(25,16),'A 5 nL drop | redistribution continues after supply stops',28)
    selected=[0,2,4,6]
    for row,(folder,title) in enumerate([('paper-legacy','v0.12'),(DROP,'v0.13')]):
        label(plate,(25,68+row*265),title,22)
        data=load(folder)
        for j,index in enumerate(selected):
            x=150+j*310;y=72+row*265
            label(plate,(x,y),f"{data['times'][index]:g} s",18)
            im=Image.open(ROOT/folder/f'time_{index:02d}.png')
            plate.paste(im.crop((53,43,117,107)).resize((224,224),Image.Resampling.NEAREST),(x,y+30))
    plate.save(OUT/'drop_comparison.png')

    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for folder,style,prefix in [('paper-legacy','--','old'),(DROP,'-','new')]:
        d=load(folder);t=[r['t'] for r in d['trace']]
        axes[0].plot(t,[r['water_mass95_radius_mm'] for r in d['trace']],style,color='#387d9c',label=prefix+' water')
        axes[0].plot(t,[r['pigment_mass95_radius_mm'] for r in d['trace']],style,color='#383b39',label=prefix+' pigment')
    axes[0].set(xlabel='Time [s]',ylabel='95% mass radius [mm]',title='Water travels farther than retained pigment')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    test=summary['validation']['reservoir'];t=np.array([v['t'] for v in test['trace']]);length=np.array([v['equivalent_wet_length_mm'] for v in test['trace']])
    axes[1].plot(t,length**2,'o-',color='#24566b')
    axes[1].set(xlabel='Time [s]',ylabel='Equivalent wet length squared [mm2]',title=f"Fed strip: L² scales with t (R²={test['length_squared_fit_r2']:.6f})")
    axes[1].grid(alpha=.15)
    labels=['Paper dt / 2\n(R1 substrate)','Paper dx / 2\n(uniform sheet)','Coupled dt / 2\n(final fiber drop)']
    values=[summary['validation'][k]['density_relative_l1']*100 for k in ['paper_time','paper_space','coupled_drop_time']]
    bars=axes[2].bar(labels,values,color=['#86a5a0','#507c85','#aa8055'])
    axes[2].bar_label(bars,fmt='%.3f%%',padding=3)
    axes[2].set(ylabel='Density relative L1 [%]',title='Limited refinement checks',ylim=(0,max(values)*1.4))
    fig.savefig(OUT/'paper_diagnostics.png',dpi=145);plt.close(fig)

    # The two recordings use identical source timestamps and post-lift duration.
    oldtimes=np.array([r['t'] for r in oldmeta['trace']])
    newtimes=np.array([r['t'] for r in newmeta['trace']])
    assert oldtimes.shape==newtimes.shape and np.max(abs(oldtimes-newtimes))<1e-8
    a=[f.convert('RGB') for f in ImageSequence.Iterator(Image.open(ROOT/'paper-stroke-legacy/progress.gif'))]
    b=[f.convert('RGB') for f in ImageSequence.Iterator(Image.open(ROOT/STROKE/'progress.gif'))]
    assert len(a)==len(b), 'Paired animation must use the same frame sampling'
    paired=[]
    for oldframe,newframe in zip(a,b):
        frame=Image.new('RGB',(1280,270),BG)
        label(frame,(20,12),'v0.12',22);label(frame,(655,12),'v0.13',22)
        frame.paste(oldframe,(15,52));frame.paste(newframe,(650,52))
        label(frame,(20,245),'After 2.41 s the brush supplies no more ink; only the paper continues evolving.',15)
        paired.append(frame)
    paired[0].save(OUT/'wetting_comparison.gif',save_all=True,append_images=paired[1:]+[paired[-1]]*14,duration=90,loop=0)
    print(json.dumps(summary,indent=2))

def material_pixel():return 50e-6

if __name__=='__main__':main()
