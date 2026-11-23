"""Build a self-contained, offline viewer from actual simulation results."""
from pathlib import Path
import base64,json,hashlib
from .landscape_engine import ROOT
from .technique_studies import save_json
from .landscape_review import board

SOURCES=[
 dict(id='CUHK',title='香港中文大学文物馆：16 种皴法示范目录',url='https://www.artmuseum.cuhk.edu.hk/en/training/resource/detail/37',
      note='已核对馆方资源页及官方播放列表条目。未逐帧分析视频；不能据此声称测得真实手势。'),
 dict(id='Met-Shen-Zhou',title='沈周《溪山秋色图》／The Met',url='https://www.metmuseum.org/art/collection/search/45683',
      note='Public Domain。已查看图像与局部，比较线群、苔点、坡面和留白的关系。'),
 dict(id='Met-Shitao-Dry',title='石涛《山水人物图》／The Met',url='https://www.metmuseum.org/art/collection/search/49171',note='Public Domain。干皴与连贯墨脊的定性对照。'),
 dict(id='Met-Shitao-Wet',title='石涛《芙蓉荷石图》／The Met',url='https://www.metmuseum.org/art/collection/search/41492',note='Public Domain。湿墨交叠与局部边缘扩展的定性对照。'),
 dict(id='MFA-Sue-Yang',title='Sue Yang／MFA Boston：Chinese Brush Painting',url='https://www.mfa.org/programs/community-programs/art-making-at-home/chinese-brush-painting',
      note='查看了笔迹教学照片；作品及教学由 Sue Yang 提供。未声称开放许可。'),
 dict(id='HKMoA-Pamphlet',title='香港艺术馆《心静出水》小册子',url='https://hk.art.museum/Exhibition_publication/XBZ-e-pamphlet_p1.pdf',
      note='查看了龚贤山水、梅清黄山等图例，作为组织关系的辅助参照；不将整本图册作为纹理素材。'),
 dict(id='HKMoA-Ink',title='香港艺术馆：笔墨与皴法教学入口',url='https://hk.art.museum/tc/web/ma/exhibitions-and-events/virtually-at-hkmoa.html',note='教学入口与相关示范链接；未声称已观看全部影片。'),
 dict(id='DPM-Chen-Shizeng',title='故宫博物院：陈师曾山水图',url='https://www.dpm.org.cn/collection/paint/231032.html',note='馆方文字说明勾、皴、擦、点、染的反复组织，作积墨练习的文献背景。'),
 dict(id='EDB-Chinese-Painting',title='香港教育局：中国绘画',url='https://www.edb.gov.hk/attachment/tc/curriculum-development/kla/arts-edu/resources/va-chinese-arts/chinese_Painting.pdf',note='中国绘画及山水笔墨的教育资料；分类背景。'),
 dict(id='NYCU-Cun',title='魏德乐：《中国山水画山石皴法合成技术之研究》',url='https://ir.lib.nycu.edu.tw/bitstream/11536/64557/1/381201.pdf',note='参考控制线、含水含墨与皴线组织的分层思路。本文的接触与纸内输运仍是自写降阶模型。'),
 dict(id='VCB',title='Chu & Tai：Virtual Chinese Brush',url='https://cse.hkust.edu.hk/VCB/',note='参考笔体弯曲、铺毫和纸面摩擦问题。未集成其 3D 笔体实现。'),
 dict(id='WetHair-Paper',title='Fei 等：A Multi-Scale Model for Simulating Liquid-Hair Interactions',url='https://www.cs.columbia.edu/cg/liquidhair/',note='液体与毛发耦合的物理参考。没有将自己的降阶方程冒充该论文原模型。'),
 dict(id='WetHair-Repo',title='raymondyfei/libWetHair',url='https://github.com/raymondyfei/libWetHair',note='作者原始实现，包含弹性杆、液体及毛发上的降阶流动。本版未编译或集成。'),
 dict(id='MyPaint-Repo',title='mypaint/libmypaint',url='https://github.com/mypaint/libmypaint',note='笔触引擎和可编辑笔刷的架构参考，不把它当作纸内流体模型；本版未集成。'),
]

SELECTION_OVERRIDES={'S01':dict(round=0,reason='R1 的补液与淡染形成三块过重刷带，压住了披麻线群；保留较轻的 R0，R1 作为未采用候选留档。')}

def dataurl(path):
    p=Path(path);mime='image/gif' if p.suffix=='.gif' else 'image/jpeg' if p.suffix in ('.jpg','.jpeg') else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()

def collect():
    byid={}
    for p in sorted((ROOT/'atlas18').glob('r*/*/s0/run.json')):
        r=json.loads(p.read_text());r['path']=p.parent;byid.setdefault(r['id'],[]).append(r)
    items=[]
    for sid,versions in sorted(byid.items()):
        versions.sort(key=lambda r:r['round']);pick=SELECTION_OVERRIDES.get(sid)
        chosen_index=next((i for i,x in enumerate(versions) if pick and x['round']==pick['round']),len(versions)-1)
        latest=versions[chosen_index];r=latest['recipe'];v=[]
        for old in versions:
            v.append(dict(round=old['round'],image=dataurl(old['path']/'result.png'),recipe=old['recipe'],
                stages=dataurl(old['path']/'stages.gif') if (old['path']/'stages.gif').exists() else None))
        items.append(dict(id=sid,title=r['title'],en=r['en'],family=r['family'],status=r['status'],note=r.get('note',''),
            check=r['check'],recipe=r,round=latest['round'],versions=v,accepted_index=chosen_index,selection_note=pick['reason'] if pick else '采用最近一次已查看的修订。',output=latest['output'],fingerprint=latest['fingerprint'],
            source_ids=r['reference_ids'],reference_url=r.get('reference_url'),revisions=r.get('revision_notes',[])))
    return items

def build():
    items=collect();out=ROOT/'atlas18';out.mkdir(exist_ok=True)
    final=ROOT/'landscape-presets';final.mkdir(exist_ok=True)
    for r in items:save_json(final/(r['id']+'.json'),r['recipe'])
    save_json(out/'accepted.json',[{k:r[k] for k in ['id','title','family','round','output','fingerprint','status','selection_note']} for r in items])
    save_json(ROOT/'reference-v018/sources.json',SOURCES)
    refs=[]
    for title,path,source,note in [
        ('沈周：坡向线群与苔点','reference-v018/shen-hemp-detail.png','Met-Shen-Zhou','皴线服从体块，长短与疏密共同变化。'),
        ('沈周：山石转面','reference-v018/shen-angular-detail.png','Met-Shen-Zhou','轮廓、立面与坡脚皴擦相互衔接。'),
        ('石涛：干皴局部','reference-v014/dry_rock.png','Met-Shitao-Dry','断续细痕与连贯墨脊并存，不能只增加孤立墨点。'),
        ('石涛：湿墨局部','reference-v014/wet_dot.png','Met-Shitao-Wet','湿墨相接，边界仍有局部结构；本模型的湿染还偏刷带。'),
        ('Sue Yang：真实练笔照片','technique-references/mfa-sue-yang-step06.jpg','MFA-Sue-Yang','比较提按、粗细、墨头与收笔；未从照片反演手势。')]:
        refs.append(dict(title=title,image=dataurl(ROOT/path),source=source,note=note))
    stats=json.loads((out/'validation.json').read_text()) if (out/'validation.json').exists() else {}
    payload=json.dumps(dict(items=items,sources=SOURCES,refs=refs,stats=stats),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template=(ROOT/'assets/landscape-atlas-template.html').read_text();html=template.replace('/*ATLAS_DATA*/',payload)
    target=ROOT/'shuimo_v0.18_atlas.html';target.write_text(html)
    paths=[ROOT/r['output'] for r in items]
    board(paths,out/'all-60.png','山水笔法图谱 · v0.18',5,cell=(360,330))
    chosen=['A01','A02','A07','A12','B01','B04','C01','C02','C05','C07','C08','C12','D03','D08','E03','E08']
    lookup={r['id']:r for r in items}
    board([ROOT/lookup[i]['output'] for i in chosen],ROOT/'shuimo_v0.18_preview.png','山水笔法 · 统一模型的 16 个代表样本',4)
    board([ROOT/r['output'] for r in items if r['family']=='组合练习'],ROOT/'shuimo_v0.18_landscapes.png','组合练习 · 披麻、斧劈、米点、折带',2,cell=(640,570))
    comparisons=[]
    for sid in ['C02','C08','C12','D08','E08']:
        versions=lookup[sid]['versions']
        for v in versions:comparisons.append(ROOT/f'atlas18/r{v["round"]}/{sid}/s0')
    # Explicit row labels in the HTML handle unequal revision counts.
    save_json(out/'coverage.json',[dict(id=r['id'],title=r['title'],family=r['family'],status=r['status'],check=r['check'],source_ids=r['source_ids'],note=r['note']) for r in items])
    print(json.dumps(dict(html=str(target),bytes=target.stat().st_size,items=len(items)),ensure_ascii=False))

if __name__=='__main__':build()
