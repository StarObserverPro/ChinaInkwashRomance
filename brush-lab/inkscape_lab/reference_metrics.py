"""Descriptive visual measurements, not physical inversion of museum scans."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d,label
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def edge_profile(a,xs,background_correct=False):
    samples=[];widths=[];halos=[];grid=np.linspace(-1.8,1.8,145)
    for x in xs:
        col=a[:,x].copy()
        if background_correct:
            bg=np.median(np.r_[col[:40],col[-40:]])
            col=np.maximum(0,1-col/max(bg,1e-10))
        col=gaussian_filter1d(col,.6)
        peak=float(np.max(col))
        if peak<.01:continue
        q=col/peak;yy=np.arange(len(q));half=np.flatnonzero(q>=.5)
        if len(half)<3:continue
        center=(half[0]+half[-1])*.5;width=half[-1]-half[0]
        # Normalize each outside flank to its own local maximum. A brush mark
        # can have two unequal dark rails; one global 90% threshold would
        # confuse that interior imbalance with an enormous blurry edge.
        mid=int(round(center));flanks=[col[:mid+1],col[mid:][::-1]];edge=[]
        for flank in flanks:
            fq=flank/max(float(flank.max()),1e-20)
            lo=np.flatnonzero(fq>=.1);hi=np.flatnonzero(fq>=.9)
            if len(lo) and len(hi):edge.append(float(hi[0]-lo[0]))
        if len(edge)!=2:continue
        samples.append(np.interp(grid,(yy-center)/width,q,left=0,right=0))
        widths.append(width);halos.append(float(np.mean(edge))/width)
    return {'x':grid.tolist(),'median_profile':np.median(samples,axis=0).tolist(),
            'median_10_90_edge_width_over_fwhm':float(np.median(halos)),
            'median_fwhm_pixels':float(np.median(widths)), 'columns':list(xs),
            'note':'Mean 10-to-90% width of the two outer flanks, each normalized to its own peak, divided by full-mark FWHM. Descriptive only: scan response, paper tone and physical width are not matched.'}

def stroke_metrics(d):
    opacity=-np.expm1(-d)
    # Same early, gently curved segment in every controlled probe. Inner white
    # run metric is not a reference fit: it only tracks the synthetic grooves.
    a=opacity[35:100,65:190]
    envelope=a>.06;closed=np.zeros_like(envelope)
    for x in range(a.shape[1]):
        ys=np.flatnonzero(envelope[:,x])
        if len(ys):closed[ys[0]:ys[-1]+1,x]=True
    white=closed & (a<.12)
    longest=0
    for row in white:
        lab,n=label(row)
        if n:longest=max(longest,int(np.bincount(lab)[1:].max()))
    return {'pale_interior_fraction':float(white.sum()/max(closed.sum(),1)),
        'longest_horizontal_pale_run_px':longest,'roi':[65,35,190,100],
        'opacity_threshold_envelope':.06,'opacity_threshold_pale':.12}

def main():
    out=ROOT/'reference-review';out.mkdir(exist_ok=True)
    im=np.asarray(Image.open(ROOT/'reference-v014/wet_stroke.png').convert('RGB'),float)
    lum=im@np.array([.2126,.7152,.0722])/255
    data={'reference_wet':edge_profile(lum,range(440,621,6),True),'rounds':{}}
    for folder in sorted((ROOT/'reference-rounds').glob('r*')):
        item={}
        for kind in ['wet','dry']:
            path=folder/kind/'fields.npz'
            if not path.exists():continue
            d=np.load(path)['density'];a=-np.expm1(-d)
            item[kind]={'edge':edge_profile(a,range(75,131,3)),**stroke_metrics(d)}
        if item:data['rounds'][folder.name]=item
    (out/'visual_metrics.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:{c:round(m['edge']['median_10_90_edge_width_over_fwhm'],3) for c,m in v.items()} for k,v in data['rounds'].items()}))

if __name__=='__main__':main()
