"""Honest reference / control / iteration plates, from saved renders only."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageSequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .reference_evidence import text,fit

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reference-review';BG=(245,242,232)
ROUNDS=[('r0','v0.13 control'),('r1','Less lateral transport; stronger retention'),
        ('r2','Liquid contact over fixed paper relief'),('r3','Contact follows each hair bending'),
        ('r4','96 finer tips; tapered shaft compliance'),('r5','More mobile pigment; retain short breaks')]

def result(r,k):return Image.open(ROOT/'reference-rounds'/r/k/'result.png').convert('RGB')
def run(r,k):return json.loads((ROOT/'reference-rounds'/r/k/'run.json').read_text())

def main():
    OUT.mkdir(exist_ok=True)
    im=Image.new('RGB',(1480,100+6*228+75),BG)
    text(im,(25,14),'Five changes, with the v0.13 control retained',30)
    text(im,(25,59),'Wet / higher water charge',22);text(im,(763,59),'Lower water / higher ink concentration',22)
    for i,(r,description) in enumerate(ROUNDS):
        y=103+i*228;text(im,(25,y),r.upper()+' | '+description,20)
        for j,k in enumerate(['wet','dry']):
            crop=result(r,k).crop((8,34,335,127)).resize((654,186),Image.Resampling.NEAREST)
            im.paste(crop,(25+j*738,y+33))
    text(im,(25,1480),'Same hand programs and initial inventories within each column. Actual deposited ink can change with contact.',17)
    text(im,(25,1507),'Common optics and paper background; no contrast matching, ink-mask editing or post-render blur.',17)
    im.save(OUT/'all_rounds.png')

    board=Image.new('RGB',(1710,895),BG)
    text(board,(25,16),'Real reference | v0.13 control | v0.14 candidate R5',31)
    titles=['Museum crop: appearance features','Simulation: v0.13 control','Simulation: R5 candidate']
    for i,title in enumerate(titles):text(board,(25+565*i,71),title,21)
    text(board,(25,109),'Wet ink: a defined core, with local edge variation',21)
    ref=fit(Image.open(ROOT/'reference-v014/wet_stroke.png'),(530,260));board.paste(ref,(25,155))
    for i,r in enumerate(['r0','r5']):
        a=result(r,'wet').crop((8,34,335,127)).resize((530,151),Image.Resampling.NEAREST)
        board.paste(a,(590+i*565,170))
    text(board,(25,384),'Lower-water texture: interrupted marks and variable white openings',21)
    ref=fit(Image.open(ROOT/'reference-v014/dry_rock.png'),(530,365));board.paste(ref,(25,430))
    for i,r in enumerate(['r0','r5']):
        a=result(r,'dry').crop((8,34,335,127)).resize((530,151),Image.Resampling.NEAREST)
        board.paste(a,(590+i*565,446))
        detail=result(r,'dry').crop((112,51,224,108)).resize((448,228),Image.Resampling.NEAREST)
        board.paste(detail,(590+i*565,620))
    text(board,(25,853),'Museum scans: Shitao / The Met / Public Domain. Feature studies, not identical strokes or calibrated paper tests.',17)
    board.save(OUT/'reference_comparison.png')

    metrics=json.loads((OUT/'visual_metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
    ref=metrics['reference_wet'];axes[0].plot(ref['x'],ref['median_profile'],color='#292620',lw=2,label='real wet-stroke crop')
    for r,color in [('r0','#b17b60'),('r3','#9a9d56'),('r4','#507f98'),('r5','#327065')]:
        d=metrics['rounds'][r]['wet']['edge'];axes[0].plot(d['x'],d['median_profile'],label=r.upper(),color=color)
    axes[0].set(xlabel='Cross-stroke position / half-maximum width',ylabel='Contrast / local peak',title='Wet cross-sections: interior mismatch remains')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    vals=[metrics['rounds'][r]['dry']['longest_horizontal_pale_run_px'] for r,_ in ROUNDS]
    bars=axes[1].bar([r.upper() for r,_ in ROUNDS],vals,color=['#aa8068']+['#849294']*4+['#327065'])
    axes[1].bar_label(bars,padding=3);axes[1].set(ylabel='Pixels in the fixed synthetic ROI',title='Longest horizontal pale run: a limited diagnostic',ylim=(0,max(vals)*1.18))
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'feature_diagnostics.png',dpi=150);plt.close(fig)

    # Paired animation retains identical timestamps, without independent speedup.
    stacks=[];times=[]
    for r,k in [('r0','wet'),('r0','dry'),('r5','wet'),('r5','dry')]:
        stacks.append([a.convert('RGB') for a in ImageSequence.Iterator(Image.open(ROOT/'reference-rounds'/r/k/'progress.gif'))])
        times.append(np.array(run(r,k)['times']))
    assert len(set(map(len,stacks)))==1
    assert all(t.shape==times[0].shape and np.max(abs(t-times[0]))<1e-8 for t in times)
    frames=[]
    for group in zip(*stacks):
        a=Image.new('RGB',(730,370),BG)
        text(a,(15,4),'v0.13 | wet / lower-water',19);text(a,(15,190),'R5 | wet / lower-water',19)
        for i,g in enumerate(group):a.paste(g,(15+(i%2)*360,33+(i//2)*184))
        frames.append(a)
    frames[0].save(OUT/'reference_progress.gif',save_all=True,append_images=frames[1:]+[frames[-1]]*12,duration=80,loop=0)

    summary={'candidate':'r5','references':'reference-v014/manifest.json','visual_metrics':metrics,
        'validation':json.loads((OUT/'validation.json').read_text()),'rounds':{},
        'limits':'Appearance guided, uncalibrated. Finished museum scans differ from wet simulations. No evaporation, full rod model, or pixel reconstruction. Continuous interior bands remain.'}
    for r,_ in ROUNDS:
        summary['rounds'][r]={k:{q:run(r,k)[q] for q in ['source_water_nl','source_pigment_ng','post_lift_density_relative_l1','elapsed_s']} for k in ['wet','dry']}
    if all((ROOT/'reference-rock'/r/'result.png').exists() for r in ['r0','r5']):
        plate=Image.new('RGB',(1460,550),BG);text(plate,(25,14),'A three-stroke structural study after the Zhao rock',29)
        for i,(name,path) in enumerate([('Real crop | Zhao Mengfu',ROOT/'reference-v014/zhao_rock.png'),('v0.13 | same authored paths',ROOT/'reference-rock/r0/result.png'),('R5 | same authored paths',ROOT/'reference-rock/r5/result.png')]):
            text(plate,(25+480*i,70),name,21);a=fit(Image.open(path),(450,355));plate.paste(a,(25+480*i,117))
        text(plate,(25,502),'This is a small construction study. Stroke order and loading are hypotheses; the real artwork is not a rendering input.',17)
        plate.save(OUT/'rock_comparison.png')
        summary['rock']={r:json.loads((ROOT/'reference-rock'/r/'run.json').read_text())['diagnostics'] for r in ['r0','r5']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'candidate':'r5','rounds':len(ROUNDS),'rock':bool(summary.get('rock'))}))

if __name__=='__main__':main()
