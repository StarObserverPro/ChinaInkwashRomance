"""Inventory, independent controls, source replay and time-step diagnostics."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from .brush import InkCanvas
from .technique_controls import TechniqueBrush,TechniqueSettings,BrushLoading,InkMaterial,InkPaperInteraction
from .technique_studies import save_json

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path/'run.json').read_text())
def fields(path):return np.load(ROOT/path/'fields.npz')

def main():
    out=ROOT/'technique-validation';out.mkdir(exist_ok=True);result={}
    # The new API allows an empty brush, water-only ink, prewet paper and refill.
    c=InkCanvas(36,30,seed=505);b=TechniqueBrush(c,loading=BrushLoading(body_saturation=0,tip_saturation=0,
        body_concentration_kg_m3=0,tip_concentration_kg_m3=0),
        technique=TechniqueSettings(evaporation_m_s=2e-7))
    mask=np.zeros((30,36));mask[9:21,9:27]=1
    b.prewet(.3,mask);before=b.paper_added_water;b.prewet(.3,mask);assert b.paper_added_water==before
    b.relax(.35);b.substrate.check();d=b.diagnostics()
    assert d['evaporated_nl']>0 and abs(d['water_relative_error'])<1e-10 and d['pigment_relative_error']==0
    b.load_profile(BrushLoading(body_saturation=.2,tip_saturation=.2,body_concentration_kg_m3=0,tip_concentration_kg_m3=0))
    first=b.volume.copy();b.load_profile(BrushLoading(body_saturation=.35,tip_saturation=.4,body_concentration_kg_m3=12,tip_concentration_kg_m3=40))
    assert np.all(b.volume>=first);d=b.diagnostics()
    assert max(abs(d['water_relative_error']),abs(d['pigment_relative_error']))<1e-10
    result['empty_prewet_evaporation_refill']=d
    # A backend comparison: entry capture moves pigment only, not water.
    a=fields('technique-rounds/r2/pulse');z=fields('technique-rounds/r3/pulse')
    samewater=float(abs(a['water']-z['water']).max());assert samewater<1e-26
    result['entry_capture_water_unchanged_max_m3']=samewater
    # Concentration is not a disguised wetness control in this constant-mu study.
    low=fields('technique-controls-r5/ink_light');high=fields('technique-controls-r5/ink_dark')
    error=float(abs(low['water']-high['water']).max());ratio=1.3/.45
    relative=float(np.abs(high['density']-low['density']*ratio).sum()/high['density'].sum())
    assert error<1e-24 and relative<1e-10
    result['concentration_control']={'max_water_difference_m3':error,'pigment_scaling_relative_error':relative,
        'note':'Viscosity held fixed. A measured viscosity-concentration relation is not implemented.'}
    # Replay actual source through a fresh paper state; no brush trajectory solve.
    folder=ROOT/'technique-rounds/r5/leaf';meta=json.loads((folder/'run.json').read_text());tape=np.load(folder/'source.npz')
    H,W=map(int,tape['shape']);canvas=InkCanvas(W,H,seed=505,paper_color=(245,240,225));canvas.density=canvas.density.astype(float)
    replay=TechniqueBrush(canvas,technique=TechniqueSettings(**meta['technique']),ink=InkMaterial(**meta['ink']),
        interaction=InkPaperInteraction(**meta['interaction']),paper_overrides=meta['paper'],loading=BrushLoading(**meta['loading']))
    for i,dt in enumerate(tape['dt']):
        lo,hi=tape['offsets'][i:i+2];ids=tape['ids'][lo:hi]
        np.add.at(replay.film_h.ravel(),ids,tape['water'][lo:hi]/replay.fluid.pixel_m**2)
        np.add.at(replay.film_mass.ravel(),ids,tape['pigment'][lo:hi]);replay._surface(float(dt))
    replay.flush_paper();replay.relax(meta['hold_s']);replay.substrate.check()
    original=np.load(folder/'fields.npz')['density'];error=float(abs(original-canvas.density).max())
    assert error<1e-7
    result['source_replay_max_density_error']=error
    # Refinement is recorded as a sensitivity result, not hidden by a loose pass threshold.
    base=fields('technique-rounds/r5/pulse');fine=fields('technique-validation/r5-half-step')
    d0=read('technique-rounds/r5/pulse');d1=read('technique-validation/r5-half-step')
    rel=float(abs(base['density']-fine['density']).sum()/base['density'].sum())
    v0=d0['diagnostics']['deposited_nl_including_rupture'];v1=d1['diagnostics']['deposited_nl_including_rupture']
    result['time_refinement']={'density_relative_L1':rel,'deposition_relative_change':abs(v1-v0)/v0,
        'base_dt_s':.0015,'refined_dt_s':.00075,'note':'One probe only. Does not establish spatial convergence or material accuracy.'}
    runs=[]
    for group in ['technique-rounds','technique-controls','technique-controls-r5','technique-applications','technique-validation/half-step','technique-validation/r5-half-step']:
        for f in (ROOT/group).rglob('run.json'):
            d=json.loads(f.read_text())['diagnostics'];water=abs(d['water_relative_error']);ink=abs(d['pigment_relative_error'])
            assert max(water,ink)<1e-9; runs.append({'path':str(f.parent.relative_to(ROOT)),'water_error':water,'pigment_error':ink})
    result['runs']=runs;result['max_water_error']=max(r['water_error'] for r in runs);result['max_pigment_error']=max(r['pigment_error'] for r in runs)
    save_json(out/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
