"""R02 focused regression. Run: xvfb-run -a python src/test_browser.py.
Tests sample renderer output; not a perceptual or hardware performance study.
"""
from pathlib import Path
import base64,json,os
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1]; E=R/'evidence'
result={'version':'SHUIMO_ROUTE_R02','checks':[],'errors':[], 'scope':'Automated asset/browser checks, not perceptual validation or hardware-performance certification.'}
def add(name,passed,**data):
 row={'name':name,'pass':bool(passed),**data};result['checks'].append(row)
 (E/'browser_tests.partial.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(name,passed,flush=True)
def main():
 scene=json.loads((R/'assets/scene.json').read_text());ar=json.loads((E/'asset_checks.json').read_text())
 add('new_alpha_zero_on_all_carrier_borders',all(r['transparent_border'] for r in ar))
 add('original_ink_retained_outside_old_cutout',all(r['restored_ink_outside_old_contour_pixels']>0 for r in ar),counts={r['patch']:r['restored_ink_outside_old_contour_pixels'] for r in ar})
 aa=np.array(Image.open(R/'assets/M02_ink_a.png'));ab=np.array(Image.open(R/'assets/M02_ink_b.png'))
 add('A_B_share_exact_alpha',np.array_equal(aa[:,:,3],ab[:,:,3]),max_difference=int(abs(aa[:,:,3].astype(int)-ab[:,:,3].astype(int)).max()))
 changed=np.any(aa[:,:,:3]!=ab[:,:,:3],axis=2);f=np.array(Image.open(R/'assets/M02_fields.png'))
 add('A_B_changes_only_inside_opaque_core',not np.any(changed&(f[:,:,0]<239)),changed_fraction=float(changed.mean()))
 # Nonintersecting world-Z slabs justify the restricted scene's sorting.
 u,v=np.meshgrid(np.linspace(0,1,300),np.linspace(0,1,300));slabs=[]
 for p in scene['patches']:
  r=p['relief'];z=p['z']-r*np.sin(np.pi*u)*np.sin(np.pi*v)*(.76+.24*np.sin(3*np.pi*u))-.18*r*np.sin(5*np.pi*u)*np.sin(np.pi*v)
  slabs.append({'id':p['id'],'min_z':float(z.min()),'max_z':float(z.max())})
 slabs += [{'id':p['id'],'min_z':p['z'],'max_z':p['z']} for p in scene['fog']]
 slabs.sort(key=lambda p:-p['min_z']);gaps=[a['min_z']-b['max_z'] for a,b in zip(slabs,slabs[1:])]
 add('authored_depth_slabs_do_not_intersect',min(gaps)>0,minimum_gap=min(gaps),slabs=slabs)
 with sync_playwright() as pw:
  b=pw.chromium.launch(executable_path=os.environ.get('CHROMIUM_EXECUTABLE'),headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist'])
  page=b.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
  requests=[];page.on('request',lambda r:requests.append(r.url));page.on('pageerror',lambda e:result['errors'].append(str(e)))
  page.set_content((R/'SHUIMO_ROUTE_R02.html').read_text(encoding='utf-8'),wait_until='load')
  page.wait_for_function('window.__inkDemo?.ready===true',timeout=45000)
  result['environment']={'browser':b.version,'rendering':'Chromium + ANGLE SwiftShader','loading':'HTML injected verbatim','runtime':page.evaluate('__inkDemo.getInfo()')}
  add('webgl_initialized',page.evaluate('__inkDemo.getError()')==0)
  err=page.evaluate('__inkDemo.referenceResidual()');add('overscan_keeps_reference_alignment',err<1e-7,max_projection_delta_px=err)
  def state(**kw):page.evaluate('(v)=>__inkDemo.setState(v)',kw)
  def capture(name):
   data=page.evaluate('__inkDemo.capture()');(E/f'{name}.png').write_bytes(base64.b64decode(data.split(',')[1]))
  state(s=.4,compare=False,edge='ink',fog=False,playing=False,solo=False,inspect='paint',backdrop=False)
  hashes={}
  for edge in ['legacy','feather','ink']:
   state(edge=edge);hashes[edge]=page.evaluate('__inkDemo.pixelHash()');capture(edge+'_scene')
  add('three_edge_modes_produce_distinct_output',len(set(hashes.values()))==3,hashes=hashes)
  state(edge='ink',s=.5,fog=True)
  h0=page.evaluate('__inkDemo.pixelHash()');page.wait_for_timeout(2000);h1=page.evaluate('__inkDemo.pixelHash()')
  add('pause_two_seconds_is_pixel_stable',h0==h1,hash_before=h0,hash_after=h1)
  pairs=[]
  for s in [.17,.50,.83]:
   state(s=s);a=page.evaluate('__inkDemo.pixelHash()');state(s=.97);state(s=.02);state(s=s);bb=page.evaluate('__inkDemo.pixelHash()');pairs.append({'s':s,'before':a,'after':bb})
  add('reverse_revisit_is_pixel_stable',all(x['before']==x['after'] for x in pairs),samples=pairs)
  state(s=0,direction=1,playing=False);page.click('#reverse');st=page.evaluate('__inkDemo.getState()');add('reverse_endpoint_does_not_teleport',st['s']==0 and not st['playing'])
  state(s=.32,playing=False,direction=1);page.click('#play');page.wait_for_function('__inkDemo.getState().s>.3201',timeout=10000);page.click('#play');a=page.evaluate('__inkDemo.getState().s');page.click('#reverse');page.wait_for_function('(s)=>__inkDemo.getState().s<s',arg=a,timeout=10000);state(playing=False);add('play_pause_reverse_controls',True)
  state(s=.4,edge='ink',fog=False,solo=False,compare=False,inspect='paint',geometry=False,region=False,appearance='A')
  # Near-white interior is tested over two very different actual backgrounds.
  candidates=np.argwhere((minimum_filter(f[:,:,0],size=7)==255)&(f[:,:,1]<13))
  probes=[]
  domain=next(p for p in scene['patches'] if p['id']=='M02')['domain']
  for y,x in candidates[::max(1,len(candidates)//25)]:
   uv=[float(domain[0]+x/1023*(domain[1]-domain[0])),float(domain[2]+y/1023*(domain[3]-domain[2]))]
   if uv[1]>next(p for p in scene['patches'] if p['id']=='M02')['sourceV']*.86:continue
   one=page.evaluate('(uv)=>__inkDemo.diagnosticProbe("M02",uv,[1,0,1])',uv)
   two=page.evaluate('(uv)=>__inkDemo.diagnosticProbe("M02",uv,[0,1,0])',uv)
   if one['visible'] and two['visible']:
    delta=max(abs(a-bb) for a,bb in zip(one['rgba'],two['rgba']));probes.append({'uv':uv,'over_magenta':one['rgba'],'over_green':two['rgba'],'delta':delta})
   if len(probes)>=8:break
  add('white_interior_is_background_independent',len(probes)>=4 and max(p['delta'] for p in probes)<=2,probes=probes)
  # Empty paper outside mountain is genuinely transparent, no rectangular depth matte.
  empties=[]
  for uv in [[.68,.045],[.92,.09],[.83,.04],[.55,.015]]:
   z=page.evaluate('(uv)=>__inkDemo.diagnosticProbe("M02",uv,[1,0,1])',uv)
   if z['visible']:empties.append(z)
  add('empty_carrier_has_no_visible_paper_rectangle',bool(empties) and all(max(abs(a-bb) for a,bb in zip(p['rgba'],[255,0,255,255]))<=1 for p in empties),probes=empties)
  # Continuous restricted route in every edge mode at extreme look poses.
  style=page.add_style_tag(content='.stage{width:320px!important;height:180px!important;min-height:0!important;}')
  smoke=page.evaluate('''()=>{let frames=0,bad=[];for(const edge of ['legacy','feather','ink'])for(let i=0;i<=6;i++)for(const fog of [false,true])for(const [yaw,pitch] of [[0,0],[-16,-7],[16,7]]){
 __inkDemo.setState({edge,s:i/6,fog,yaw,pitch,compare:false,inspect:'paint',playing:false});
 let err=__inkDemo.getError();if(err)bad.push({edge,i,fog,yaw,pitch,err});frames++;
 }return {frames,bad};}''')
  add('126_route_edge_fog_pose_smoke_frames',not smoke['bad'],**smoke)
  style.evaluate('(s)=>s.remove()');page.wait_for_timeout(80)
  state(s=.4,yaw=0,pitch=0,fog=True,edge='ink',compare=False,solo=False)
  ids=[p['id'] for p in page.evaluate('__inkDemo.layerOrder()')]
  add('fog_interleaves_with_mountains',ids.index('fog_valley')>ids.index('M02') and ids.index('fog_valley')<ids.index('M03'),order=ids)
  before=page.evaluate('__inkDemo.pixelHash()');state(fog=False);after=page.evaluate('__inkDemo.pixelHash()');add('fog_toggle_changes_compositing',before!=after)
  for inspect in ['coverage','ink','alpha']:
   state(inspect=inspect,compare=False);capture('field_'+inspect)
  add('coverage_ink_alpha_diagnostics_render',page.evaluate('__inkDemo.getError()')==0)
  for name,opts in [('new_no_fog',dict(s=.4,fog=False,compare=False)),('new_with_fog',dict(s=.4,fog=True,compare=False)),('comparison',dict(s=.4,fog=False,compare=True)),('solo_magenta',dict(s=.4,fog=False,compare=False,solo=True,backdrop=True)),('carrier_mesh',dict(s=.4,fog=False,compare=False,solo=True,geometry=True)),('end',dict(s=1,fog=False,compare=False))]:
   state(inspect='paint',edge='ink',solo=False,backdrop=False,geometry=False,appearance='auto',**opts) if not any(k in opts for k in ['solo','backdrop','geometry']) else state(**{'inspect':'paint','edge':'ink','solo':False,'backdrop':False,'geometry':False,'appearance':'auto',**opts})
   capture(name)
   if name=='comparison':page.screenshot(path=str(E/'comparison_page.png'),full_page=True)
  state(inspect='paint',compare=True,solo=False,backdrop=False,geometry=False)
  page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(100);page.evaluate('__inkDemo.render()')
  dims=page.evaluate('({viewport:innerWidth,document:document.documentElement.scrollWidth})');add('mobile_no_horizontal_overflow',dims['document']<=dims['viewport'],**dims);page.screenshot(path=str(E/'mobile_page.png'),full_page=True)
  add('no_external_requests',not requests,count=len(requests))
  add('no_javascript_or_gl_errors',not result['errors'] and page.evaluate('__inkDemo.getError()')==0,errors=result['errors'])
  b.close()
 result['all_checks_passed']=all(x['pass'] for x in result['checks']);(E/'browser_tests.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('ALL CHECKS PASSED',result['all_checks_passed'],len(result['checks']))
 if not result['all_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
