"""Audit actual simulation inventories and paired numerical/seed variations."""
from pathlib import Path
import json
import numpy as np
from .landscape_engine import ROOT,fingerprint
from .technique_studies import save_json
from .technique_controls import compile_gesture

def validate():
    records=[];errors=[]
    for p in sorted((ROOT/'atlas18').glob('r*/*/s*/run.json')):
        r=json.loads(p.read_text());records.append(r);label=str(p.relative_to(ROOT))
        if r['fingerprint']!=fingerprint(r['recipe'],r['config'],r['seed'],r['detail']):errors.append(label+': fingerprint')
        for op in r['recipe']['operations']:
            if op['kind']=='stroke':compile_gesture(op['gesture'])
        with np.load(p.parent/'fields.npz') as fields:
            for key in fields.files:
                a=fields[key]
                if not np.isfinite(a).all() or a.min() < -1e-18:errors.append(label+': field '+key)
        for k in ['water_relative_error','pigment_relative_error']:
            if abs(r['diagnostics'][k])>1e-8:errors.append(label+': '+k)
        if r['descriptors']['border_opacity_max']>.01:errors.append(label+': border clipping')
    primary={}
    for r in records:
        if r['seed']==0 and r['detail']==1:
            if r['id'] not in primary or primary[r['id']]['round']<r['round']:primary[r['id']]=r
    variations=[];refinements=[]
    for r in records:
        if r['id'] not in primary:continue
        base=primary[r['id']]
        if base['recipe']!=r['recipe']:continue
        if r['seed'] and r['detail']==1:
            variations.append(dict(id=r['id'],seed=r['seed'],base_output=base['output'],output=r['output'],
                ink_area_ratio=r['descriptors']['ink_pixels']/max(base['descriptors']['ink_pixels'],1),
                mean_opacity_ratio=r['descriptors']['mean_ink_opacity']/max(base['descriptors']['mean_ink_opacity'],1e-12)))
        if r['seed']==0 and r['detail']>1:
            a=np.load(ROOT/base['output']/'fields.npz')['density'];b=np.load(ROOT/r['output']/'fields.npz')['density']
            refinements.append(dict(id=r['id'],detail=r['detail'],density_l1_relative=float(abs(a-b).sum()/max(abs(b).sum(),1e-15)),
                opacity_l1_relative=float(abs(np.expm1(-a)-np.expm1(-b)).sum()/max(abs(np.expm1(-b)).sum(),1e-15)),
                note='Time and travel step halved together; paper grid unchanged. This is a sensitivity check, not proof of convergence.'))
    stats=dict(render_count=len(records),preset_count=len(primary),round_counts={str(i):sum(r['round']==i for r in records) for i in range(3)},
        max_water_error=max(abs(r['diagnostics']['water_relative_error']) for r in records),
        max_pigment_error=max(abs(r['diagnostics']['pigment_relative_error']) for r in records),errors=errors,
        seed_variations=variations,step_refinements=refinements,
        note='Conservation checks numerical accounting. These numbers do not establish physical calibration or likeness to historical brushwork.')
    save_json(ROOT/'atlas18/validation.json',stats);print(json.dumps(stats,ensure_ascii=False,indent=2))
    if errors:raise SystemExit('Validation failed')
    return stats

if __name__=='__main__':validate()
