"""Package v0.17 code and retained evidence, excluding compiled/cache artifacts."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,hashlib,json,shutil
from .monte_carlo_studies import ROOT

def usable(p):
    rel=p.relative_to(ROOT)
    return p.is_file() and not any(t in ('__pycache__','.git') for t in rel.parts) and p.suffix not in ('.so','.pyc','.log')

def light(p):
    r=p.relative_to(ROOT)
    return len(r.parts)==1 or r.parts[0] in ('inkscape_lab','docs','technique-presets') or (r.parts[0]=='study17' and len(r.parts)==2 and ('design' in r.name or 'feedback' in r.name)) or (r.parts[0]=='review17' and r.name in ('summary.json','metrics.csv'))

def main(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    records={}
    for name,predicate in [('shuimo_v0.17_code.zip',light),('shuimo_v017_sequential_experiments.zip',lambda p:True)]:
        files=[p for p in sorted(ROOT.rglob('*')) if usable(p) and predicate(p)]
        manifest={'version':'0.17','accepted_default':'C06','candidate':'B08','files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
        with ZipFile(out/name,'w',ZIP_DEFLATED,compresslevel=6) as z:
            for p in files:z.write(p,str(p.relative_to(ROOT)))
            z.writestr('package-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        with ZipFile(out/name) as z:
            bad=z.testzip()
            if bad:raise RuntimeError(f'Archive verification failed: {bad}')
        records[name]={'bytes':(out/name).stat().st_size,'members':len(files)+1,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
    for a,b in [('comparison.png','shuimo_v017_comparison.png'),('robustness.png','shuimo_v017_robustness.png'),('review.html','shuimo_v017_review.html')]:
        shutil.copyfile(ROOT/'review17'/a,out/b);records[b]={'bytes':(out/b).stat().st_size,'sha256':hashlib.sha256((out/b).read_bytes()).hexdigest()}
    (out/'checksums.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');main(p.parse_args().destination)
