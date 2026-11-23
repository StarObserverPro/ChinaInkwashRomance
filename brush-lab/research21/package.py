from pathlib import Path
import zipfile,json,hashlib,sys
R=Path(__file__).resolve().parents[1];D=R.parent/'deliverables-v021';name=sys.argv[1];p=D/f'shuimo_v021_{name}.zip'
D.mkdir(exist_ok=True)
assert not p.exists(),'Versions are immutable'
runs=list((R/'stroke21'/name).glob('*/physical/run.json'));assert len(runs)>=3
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for directory in ['inkscape_lab','research21','stroke21/'+name]:
  for f in (R/directory).rglob('*'):
   if not f.is_file() or '__pycache__' in f.parts or f.suffix in ['.so','.pyc','.log'] or f.name.startswith('progress'):continue
   z.write(f,str(f.relative_to(R)),compress_type=zipfile.ZIP_STORED if f.suffix in ['.gz','.npz','.png'] else zipfile.ZIP_DEFLATED)
 for directory in ['assets','references19','reference-v018']:
  for f in (R/directory).glob('*'):
   if f.is_file():z.write(f,str(f.relative_to(R)))
 for n in ['selected_v016.json','stroke20/readings.json']:
  z.write(R/n,n)
 for f in (R/'stroke20').glob('*'):
  if f.is_file() and f.suffix in ['.png','.npy']:z.write(f,str(f.relative_to(R)))
 z.writestr('VERSION.json',json.dumps(dict(version='0.21',round=name,runs=len(runs),physics='unchanged v0.16 closure; v0.19 session',notes='Actual fields, states, source, input gestures; no texture masks.')))
index=R/'stroke21/versions.json';data=json.loads(index.read_text()) if index.exists() else {};data[name]=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,runs=len(runs));index.write_text(json.dumps(data,indent=2));print(data[name])
