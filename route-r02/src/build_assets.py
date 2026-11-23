"""R02: Overscanned 2D carriers with independent ink and surface-coverage fields.

No brush generation. Cached flattened v0.19 paintings are adapted once offline.
The ink-driven mode does NOT use the old authored contour for its alpha for the central T05 mountain. Other multi-object source caches
use a loose semantic ROI to suppress unrelated fragments. The body envelope is a provisional image-derived estimate (closing + hole filling),
NOT a claim that surface coverage/depth can be inferred reliably from luminance.
A real brush adapter should supply coverage/ink maps with the shared UV domain.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageFilter

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'assets'; A.mkdir(exist_ok=True)
E=ROOT/'evidence';E.mkdir(exist_ok=True)
R01=ROOT/'baseline_assets'
N=1024; SRC=512
PAPER=np.array([245.,242.,232.])/255
INK=np.array([40.,53.,49.])/255
DOMAIN=[-.10,1.10,-.05,1.05]

def smooth(a,b,x):
    t=np.clip((x-a)/(b-a),0,1);return t*t*(3-2*t)

def pigment(path,case):
    im=Image.open(path).convert('RGB')
    if case=='T10':im=im.crop((0,35,im.width,im.height))
    arr=np.array(im.resize((SRC,SRC),Image.Resampling.LANCZOS),float)
    white=np.percentile(arr.reshape(-1,3),97,axis=0)
    return np.clip(1-np.mean(arr/white,axis=-1),0,1)

def disk(r):
    y,x=np.mgrid[-r:r+1,-r:r+1];return x*x+y*y<=r*r

def envelope(p):
    # Single fixed master, independent of camera and appearance A/B. Outer ink
    # is never multiplied by this field. Fill preserves enclosed interior white.
    raw=ndi.gaussian_filter(p,.65)>.036
    support=ndi.binary_closing(np.pad(raw,24),structure=disk(11))[24:-24,24:-24]
    labels,n=ndi.label(support);sz=np.bincount(labels.ravel());sz[0]=0
    keep=np.flatnonzero(sz>max(40,sz.max()*.004))
    support=np.isin(labels,keep)
    support=ndi.binary_fill_holes(support)
    dist=ndi.distance_transform_edt(support)
    # Only the inner envelope becomes paper-opaque. The original ink extends
    # beyond it and itself supplies the soft outermost visible edge.
    C=smooth(3.0,23.0,dist)
    return C,support

def resample(arr,u,v,fill=0):
    if arr.ndim==2:
        return ndi.map_coordinates(arr,[v*(arr.shape[0]-1),u*(arr.shape[1]-1)],order=1,mode='constant',cval=fill)
    return np.stack([resample(arr[...,k],u,v,fill) for k in range(arr.shape[-1])],axis=-1)

def save_straight(name,rgb,alpha):
    arr=np.dstack([rgb,np.clip(alpha,0,1)])
    Image.fromarray(np.clip(np.round(arr*255),0,255).astype('uint8')).save(A/name,optimize=True)

def save_fields(name,I,C):
    # Mathematically equivalent to INK over an optional PAPER surface, then over
    # the already composited scene. Do not multiply fringe ink by C again.
    alpha=I+(1-I)*C
    premul=INK*I[...,None]+PAPER*((1-I)*C)[...,None]
    straight=np.divide(premul,alpha[...,None],out=np.zeros_like(premul),where=alpha[...,None]>1e-8)
    save_straight(name,straight,alpha)
    return alpha,premul

def main():
    scene=json.loads((ROOT/'scene_r01.json').read_text())
    uu,vv=np.meshgrid(np.linspace(DOMAIN[0],DOMAIN[1],N),np.linspace(DOMAIN[2],DOMAIN[3],N))
    rows=[]
    for p in scene['patches']:
        id=p['id'];sv=p['sourceV']
        originalA=np.asarray(Image.open(R01/p['textureA']).convert('RGBA'),float)/255
        originalB=np.asarray(Image.open(R01/p['textureB']).convert('RGBA'),float)/255
        legacyC=resample(originalA[:,:,3],uu,vv)
        legacyA=resample(originalA[:,:,:3],uu,vv)
        legacyB=resample(originalB[:,:,:3],uu,vv)
        # Prevent zero-RGB outside the source image bleeding in before premultiply.
        legacyA[legacyC==0]=0;legacyB[legacyC==0]=0
        p['edgeVariants']={}
        for mode in ['legacy','feather','ink']:
            p['edgeVariants'][mode]={'A':f'{id}_{mode}_a.png','B':f'{id}_{mode}_b.png' if id=='M02' else f'{id}_{mode}_a.png'}
        save_straight(p['edgeVariants']['legacy']['A'],legacyA,legacyC)
        if id=='M02':save_straight(p['edgeVariants']['legacy']['B'],legacyB,legacyC)
        # Uniform alpha feather is deliberately a control, not the proposed fix.
        featherC=ndi.gaussian_filter(legacyC,5.0)
        valid=legacyC>.01
        nearest=ndi.distance_transform_edt(~valid,return_distances=False,return_indices=True)
        featherA=legacyA[tuple(nearest)];featherB=legacyB[tuple(nearest)]
        save_straight(p['edgeVariants']['feather']['A'],featherA,featherC)
        if id=='M02':save_straight(p['edgeVariants']['feather']['B'],featherB,featherC)
        # Use uncropped source ink, not ink already cut by R01's silhouette.
        src=ROOT/'frozen_cache'/p['file']
        raw=pigment(src,p['case'])
        clean=np.clip((raw-.019)/.981,0,1)
        # The central T05 source is already an isolated mountain. Other legacy
        # paintings contain unrelated strokes: a loose semantic ROI suppresses
        # distant fragments; it is not used as the visible alpha silhouette.
        roi=np.ones_like(clean)
        if p['case']!='T05':
            ry,rx=np.mgrid[0:SRC,0:SRC]/(SRC-1)
            oldRegion=resample(originalA[:,:,3],rx,ry*sv)>.5
            outside=ndi.distance_transform_edt(~oldRegion)
            roi=1-smooth(9,32,outside)
            clean*=roi
        body,support=envelope(clean)
        # The RGB-only source cannot recover what was cropped at image borders.
        # Fade its border, do not invent more mountain geometry or new ink.
        sy,sx=np.mgrid[0:SRC,0:SRC]/(SRC-1)
        guard=smooth(0,.035,sx)*smooth(0,.035,1-sx)*smooth(0,.015,sy)
        body*=guard
        rawI=resample(clean,uu,vv/sv)
        edgeguard=resample(guard,uu,vv/sv)
        foot=smooth(0,sv*.20,sv-vv)
        I=np.clip(rawI*p['strength']*edgeguard*foot,0,.94)
        C=resample(body,uu,vv/sv)*foot
        # Fixed alpha/outer ink in A/B. Only inner colour changes, exactly as R01.
        alpha,premul=save_fields(p['edgeVariants']['ink']['A'],I,C)
        if id=='M02':
            newraw=pigment(ROOT/'frozen_cache/T05_R2.png','T05')
            q=np.clip((newraw-.019)/.981,0,1)
            qI=resample(q,uu,vv/sv)*p['strength']*edgeguard*foot
            m=resample(np.asarray(Image.open(R01/'M02_update_region.png'),float)/255,uu,vv)
            m*=smooth(.94,1,C)
            Ib=np.clip(I+(qI-I)*m,0,.94)
            # Keep transmittance alpha invariant; change premultiplied colour
            # only in C=1 interior so it remains physically the same coverage.
            aB=alpha.copy()
            rgbA=np.divide(premul,alpha[...,None],out=np.zeros_like(premul),where=alpha[...,None]>1e-8)
            rgbB=rgbA+(INK-PAPER)*(Ib-I)[...,None]
            save_straight(p['edgeVariants']['ink']['B'],rgbB,aB)
            Image.fromarray(np.clip(m*255,0,255).astype('uint8')).save(A/'M02_update_region.png')
        # Diagnostics are authored fields, not geometry or inferred depth.
        p['fieldTexture']=f'{id}_fields.png'
        field=np.stack([C,I,legacyC,np.ones_like(C)],-1)
        Image.fromarray(np.round(field*255).astype('uint8')).save(A/p['fieldTexture'],optimize=True)
        p['textureA']=p['edgeVariants']['ink']['A'];p['textureB']=p['edgeVariants']['ink']['B']
        p['domain']=DOMAIN;p['grid']=[48,52] if p['relief'] else [36,24]
        p['coverage_source']='legacy RGB cache adapter: loose semantic ROI for multi-object source caches, support closing + hole fill; not brush-native'
        white=(C>=.999)&(I<.018)&(vv<sv*.85)
        restored=(legacyC<.005)&(I>.025)&(C<.5)
        rows.append({'patch':id,'transparent_border':bool(np.all(alpha[0]==0) and np.all(alpha[-1]==0) and np.all(alpha[:,0]==0) and np.all(alpha[:,-1]==0)),
          'white_opaque_pixel_count':int(white.sum()),'restored_ink_outside_old_contour_pixels':int(restored.sum()),
          'ink_fringe_pixel_count':int(((C==0)&(I>.002)).sum()),
          'support_width_factor':1.2,'support_height_factor':1.1,
          'white_probe_uv':[[float(uu[y,x]),float(vv[y,x])] for y,x in np.argwhere(white)[::max(1,int(white.sum()/12))][:12]],
          'max_premul_minus_alpha':float((premul-alpha[...,None]).max())})
    scene['version']='SHUIMO_ROUTE_R02'
    scene['coverage_note']='Carrier geometry only transports coordinates; ink, interior coverage and extent are separate. Legacy/feather are comparison modes, not geometry truth.'
    scene['limits']=[
      'Existing low-resolution, paper-baked brush caches; no new brush solver or brush-native alpha.',
      'Image-derived interior coverage is a provisional adapter and can fill some exterior gaps or miss open white regions.',
      'Fixed, nonintersecting depth-layer order over the promised route; arbitrary intersecting transparent surfaces are not solved.',
      'Hand-authored proxy depth, reference views and route remain unchanged in intent.',
      'No 3D ink aerosol simulation, new-view repainting, near scenery, binocular VR or 360-degree claim.'
    ]
    (A/'scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2))
    (E/'asset_checks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (ROOT/'CHECKPOINT.json').write_text(json.dumps({'version':scene['version'],'stage':'assets_compiled','next':'bundle and browser regression','files':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in A.glob('*') if f.is_file()}},ensure_ascii=False,indent=2))
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
