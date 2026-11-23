"""Build review plates from recorded simulations, without editing ink fields."""
from pathlib import Path
import base64
import json
import html
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'meniscus-review'
BG = (246, 244, 236)
INK = (37, 47, 49)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def label(im, xy, text, size=19, fill=INK):
    font = ImageFont.truetype(FONT, size) if Path(FONT).exists() else ImageFont.load_default()
    ImageDraw.Draw(im).text(xy, text, fill=fill, font=font)


def read(name):
    return json.loads((ROOT / name / 'run.json').read_text())


def pic(name):
    return Image.open(ROOT / name / 'result.png').convert('RGB')


def data_uri(path):
    path = Path(path)
    mime = 'image/gif' if path.suffix == '.gif' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def make_review():
    OUT.mkdir(exist_ok=True)
    runs = {n: read(n) for n in ['meniscus-final', 'meniscus-final-dry']}
    tests = {n: read('meniscus-validation/' + n) for n in ['slow', 'fast', 'fine', 'no_pores']}
    d0 = np.load(ROOT / 'meniscus-validation/slow/fields.npz')['density']
    d1 = np.load(ROOT / 'meniscus-validation/fine/fields.npz')['density']
    stats = {
        'wet': runs['meniscus-final']['diagnostics'],
        'dry': runs['meniscus-final-dry']['diagnostics'],
        'probes': {n: r['diagnostics'] for n, r in tests.items()},
        'refinement_relative_paper_volume_difference': abs(tests['fine']['diagnostics']['paper_nl'] / tests['slow']['diagnostics']['paper_nl'] - 1),
        'refinement_density_relative_l1': float(abs(d1-d0).sum() / abs(d0).sum()),
        'substrate': json.loads((ROOT / 'meniscus-validation/substrate/validation.json').read_text()),
    }
    for key, name in [('wet', 'meniscus-final'), ('dry', 'meniscus-final-dry')]:
        tr = runs[name]['trace']
        stats[key]['pause_absorption_nl'] = sum(r['absorbed_m3'] for r in tr if r['event'] == 3) * 1e12
        stats[key]['peak_retained_noncontact_bridges'] = max(r['retained_noncontact_bridges'] for r in tr)
    rock_file = ROOT / 'meniscus-rock/run.json'
    if rock_file.exists():
        stats['rock'] = json.loads(rock_file.read_text())['diagnostics']
    (OUT / 'validation_summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False) + '\n')

    im = Image.new('RGB', (1380, 1070), BG)
    label(im, (30, 20), 'v0.12 | Visual feedback and corrections', 30)
    label(im, (30, 65), 'Recorded simulation outputs. Iterations change the model; they are not controlled material comparisons.', 17)
    entries = [
        ('meniscus-round3', 'R3 | Long-lived anchored bridges', 'Long drag marks and isolated drops: rejected.'),
        ('meniscus-round4', 'R4 | Explicit contact-line update', 'Periodic bead rows. Image only; ledger save failed.'),
        ('meniscus-round5', 'R5 | Implicit contact-line update', 'Bead rows decrease, but breakup still repeats.'),
        ('meniscus-round6', 'R6 | Sliding wet contact', 'Continuous bands return; long parallel gaps remain.'),
        ('meniscus-final', 'R8 | 3-D tip motion + contact taper', 'Fewer bead rows. Comb-like bands remain a defect.'),
        ('meniscus-final-dry', 'R8 | Same path, less initial liquid', 'Wider pale lanes; patch-like dry breakup is weak.'),
    ]
    for i, (name, title, note) in enumerate(entries):
        x, y = 30 + (i % 2) * 675, 115 + (i // 2) * 305
        label(im, (x, y), title, 22)
        im.paste(pic(name), (x, y + 42))
        label(im, (x, y + 245), note, 17)
    label(im, (30, 1030), 'R1/R7 are incomplete previews. R2 failed pigment conservation/positivity and is excluded as evidence.', 17)
    im.save(OUT / 'iteration_comparison.png')

    im = Image.new('RGB', (1320, 750), BG)
    label(im, (30, 20), 'Same path, anatomy and paper | Wet / less ink', 30)
    for i, (name, key, title) in enumerate([
        ('meniscus-final', 'wet', 'Initial fraction 0.48'),
        ('meniscus-final-dry', 'dry', 'Initial fraction 0.26'),
    ]):
        x = 30 + i * 650
        label(im, (x, 76), title, 22)
        label(im, (x, 110), f"Liquid absorbed: {stats[key]['paper_nl']:.2f} nL", 18)
        source = pic(name)
        im.paste(source, (x, 145))
        label(im, (x, 360), 'Pause (4x)', 20)
        label(im, (x + 300, 360), 'Lift (4x)', 20)
        im.paste(source.crop((305, 70, 365, 142)).resize((240, 288), Image.Resampling.NEAREST), (x, 400))
        im.paste(source.crop((392, 36, 465, 98)).resize((292, 248), Image.Resampling.NEAREST), (x + 300, 400))
    label(im, (30, 712), 'Crops use nearest-neighbor magnification. No image noise, contrast remapping or painted white masks.', 17)
    im.save(OUT / 'wet_dry_detail.png')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for name, color, title in [('meniscus-final', '#24566b', 'wet'), ('meniscus-final-dry', '#b46c31', 'less ink')]:
        tr = runs[name]['trace']
        t = np.array([r['t'] for r in tr])
        xy = np.array([r['hand'] for r in tr])
        dt = np.diff(t)
        speed = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt * .05]
        if title == 'wet':
            ax[0].plot(t, speed, color='#53666b', label='hand speed')
            second = ax[0].twinx()
            second.plot(t, [r['force'] for r in tr], color='#c59148', label='load command response')
            second.set_ylabel('Dimensionless load', color='#986a2f')
        ax[1].plot(t, [r['filled_pores'] for r in tr], color=color, label=title)
        ax[2].plot(t, [r['retained_noncontact_bridges'] for r in tr], color=color, label=title)
        ax[3].plot(t, np.cumsum([r['absorbed_m3'] for r in tr]) * 1e12, color=color, label=title)
    for a in ax:
        a.axvspan(1.10, 1.38, color='#d6b970', alpha=.22)
        a.axvspan(1.93, 2.21, color='#8fb5b9', alpha=.19)
        a.grid(alpha=.15)
    ax[0].set_title('Recorded state: pause 1.10–1.38 s (yellow); release 1.93–2.21 s (blue)', loc='left')
    ax[0].set_ylabel('Hand speed [mm/s]')
    ax[1].set_ylabel('Filled pores [count]')
    ax[1].legend(loc='upper left', ncol=2)
    ax[2].set_ylabel('Retained bridges\nwithout solid contact')
    ax[3].set_ylabel('Cumulative paper\nabsorption [nL]')
    ax[3].set_xlabel('Simulation time [s]')
    fig.savefig(OUT / 'physical_diagnostics.png', dpi=145)
    plt.close(fig)

    im = Image.new('RGB', (1200, 760), BG)
    label(im, (30, 20), 'Mechanism and numerical probes', 30)
    es = [('slow', '15 mm/s'), ('fast', '45 mm/s'), ('fine', '15 mm/s, half sample spacing + dt'), ('no_pores', '15 mm/s, three-hair pores disabled')]
    for i, (name, title) in enumerate(es):
        x, y = 30 + (i % 2) * 590, 90 + (i // 2) * 300
        label(im, (x, y), title, 19)
        source = pic('meniscus-validation/' + name)
        im.paste(source.resize((510, 250), Image.Resampling.NEAREST), (x, y + 32))
        label(im, (x, y + 255), f"Paper: {tests[name]['diagnostics']['paper_nl']:.3f} nL", 17)
    label(im, (30, 710), f"Refinement: volume difference {stats['refinement_relative_paper_volume_difference']:.2%}; pixel-density L1 {stats['refinement_density_relative_l1']:.2%}.", 18)
    im.save(OUT / 'mechanism_probes.png')

    def figure(file, caption):
        return f'<figure><img src="{data_uri(file)}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
    sections = [
        '<section id="rounds"><h2>先看改动是否消掉了伪影</h2><p>模型经历八个工作阶段，其中 R1、R7 只有预览，R2 出现数值失效；它们不作为成功证据。下图保留了中间的失败形态。R4 只完成成图，账本保存失败。</p>' + figure(OUT / 'iteration_comparison.png', '各轮使用了不同的模型和部分参数，不能据此比较供墨量。') + '</section>',
        '<section id="motion" hidden><h2>同一路径：前进、加速、停顿、转向、提笔</h2><p>以下动画来自实际计算过程。墨色直接由颜料面密度得到。湿笔与较干笔只改变初始储液比例，纸、笔毛样本和路径相同。</p><div class="pair">' + figure(ROOT / 'meniscus-final/progress.gif', '初始储液比例 0.48') + figure(ROOT / 'meniscus-final-dry/progress.gif', '初始储液比例 0.26') + '</div>' + figure(OUT / 'wet_dry_detail.png', '停顿与提笔局部。末端小珠和长条白缝仍是问题。') + '</section>',
        '<section id="physics" hidden><h2>把视觉变化和状态变量对应起来</h2><p>停顿时纸内吸液量继续增加；提笔时非接触液桥短暂存在。这里的“非接触”指毛尖未与纸面固体接触，仍可能有湿润连接，不能等同于整笔已完全离纸。最终湿笔在完全离纸阶段没有测得额外吸液，因此这次没有证明明显的离纸后拖尾浸润。</p>' + figure(OUT / 'physical_diagnostics.png', '记录的液量、孔隙状态与动作时间。守恒不等于模型已经符合真实毛笔。') + figure(OUT / 'mechanism_probes.png', '速度、步长和三毛孔隙机制对照。关闭孔隙仍保留两毛液桥。') + f'<p class="warning">步长与采样减半：落纸总液量变化 {stats["refinement_relative_paper_volume_difference"]:.2%}，但密度场逐像素相对 L1 差为 {stats["refinement_density_relative_l1"]:.2%}。细白缝的位置尚未收敛。</p></section>',
    ]
    rock = ROOT / 'meniscus-rock/result.png'
    if rock.exists():
        sections.append('<section id="rock" hidden><h2>原来的九笔山石</h2><p>沿用之前临摹的九笔动作，替换供墨与接触模型。它用于观察同一机制放回皴笔后的表现；没有对原画做描图或像素贴合。</p>' + figure(rock, '本轮九笔结果。细长条带仍限制了自然的皴擦质感。') + '</section>')
    tabs = [('rounds', '逐轮对照'), ('motion', '干湿与动画'), ('physics', '物理诊断')]
    if rock.exists(): tabs.append(('rock', '九笔山石'))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>水墨 v0.12 · 多轮视觉反馈</title><style>
body{margin:0;background:#f2f0e7;color:#253339;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px 70px}h1{font-size:32px;line-height:1.3}h2{font-size:23px}p{max-width:960px}nav{position:sticky;top:0;background:#f2f0e7;padding:12px 0;display:flex;gap:8px;flex-wrap:wrap;z-index:1}button{font:inherit;padding:8px 18px;border:1px solid #bac5c1;border-radius:24px;background:#faf9f4;cursor:pointer}button[aria-selected=true]{background:#24566b;color:#fff}figure{margin:20px 0;background:#faf9f4;padding:12px;border-radius:10px}img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:#536466;margin-top:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.warning{border-left:4px solid #ad7342;padding:12px 18px;background:#eee2d2}.tag{font-size:13px;letter-spacing:1px;color:#6a7771}@media(max-width:740px){.pair{grid-template-columns:1fr}main{padding:20px 12px}h1{font-size:26px}}</style><main>
<div class="tag">COMPUTED BRUSH STUDY · 0.12 · 2026-09-28</div><h1>离散毛间供墨，与连续纸面湿润</h1>
<p>这是按毛细液桥、弹性笔毛和薄膜流重新实现的实验版本。多轮修正消除了部分喷点和周期性断裂；长条梳齿感、局部步长敏感性和真实前后缘湿润仍未解决，不能把它称为已经验证的毛笔模拟。</p><nav>'''
    page += ''.join(f'<button data-tab="{key}" aria-selected="{str(i == 0).lower()}">{title}</button>' for i, (key, title) in enumerate(tabs))
    page += '</nav>' + ''.join(sections)
    page += '''<p>模型与数值限制、复现命令和论文出处见随附说明。图像全部来自本次求解器输出。</p></main><script>
document.querySelectorAll('button[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('main>section').forEach(s=>s.hidden=s.id!==b.dataset.tab);document.querySelectorAll('button[data-tab]').forEach(x=>x.setAttribute('aria-selected',String(x===b)));});
</script></html>'''
    (OUT / 'shuimo_v012_review.html').write_text(page)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    make_review()
