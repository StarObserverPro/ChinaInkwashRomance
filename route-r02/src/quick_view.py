from playwright.sync_api import sync_playwright
from pathlib import Path
import base64,os
R=Path(__file__).resolve().parents[1]
(R/'evidence').mkdir(exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_EXECUTABLE'),headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'])
 page=b.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
 page.on('pageerror',lambda e:print(e))
 page.set_content((R/'SHUIMO_ROUTE_R02.html').read_text(encoding='utf-8'),wait_until='load')
 page.wait_for_function('window.__inkDemo?.ready',timeout=45000)
 print(page.evaluate('__inkDemo.getInfo()'))
 for name,opts in [('new',{'s':.4,'compare':False,'edge':'ink'}),('old',{'edge':'legacy'}),('feather',{'edge':'feather'}),('compare',{'edge':'ink','compare':True}),('solo',{'edge':'ink','compare':False,'solo':True,'backdrop':True})]:
  page.evaluate('(v)=>__inkDemo.setState(v)',opts)
  s=page.evaluate('__inkDemo.capture()')
  (R/f'evidence/{name}.png').write_bytes(base64.b64decode(s.split(',')[1]))
 page.evaluate('__inkDemo.setState({s:.4,solo:false,compare:true,backdrop:false})')
 page.screenshot(path=str(R/'evidence/compare_page.png'),full_page=True)
 b.close()
