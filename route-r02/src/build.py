"""Bundle scene data and local images into a dependency-free offline HTML."""
from pathlib import Path
import json,base64
ROOT=Path(__file__).resolve().parents[1]
scene=json.loads((ROOT/'assets/scene.json').read_text(encoding='utf-8'))
names={n for p in scene['patches'] for v in p['edgeVariants'].values() for n in v.values()}|{p['fieldTexture'] for p in scene['patches']}|{'fog.png'}
data={'scene':scene,'images':{n:'data:image/png;base64,'+base64.b64encode((ROOT/'assets'/n).read_bytes()).decode() for n in sorted(names)}}
text=(ROOT/'src/index.template.html').read_text(encoding='utf-8').replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')).replace('__ENGINE__',(ROOT/'src/engine.js').read_text(encoding='utf-8'))
out=ROOT/'SHUIMO_ROUTE_R02.html';out.write_text(text,encoding='utf-8',newline='\n')
print(out,out.stat().st_size)
