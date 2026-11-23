"""Build the portable, offline study gallery with Python's standard library.

The gallery embeds rendered PNGs and logged brush paths.  Its path playback is
explicitly a geometric diagram, not a re-simulation of ink or an artist's hand.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import struct
from typing import Any


def _json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _png(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    width, height = (1000, 700)
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        width, height = struct.unpack(">II", raw[16:24])
    return {"src": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
            "width": width, "height": height, "file": path.name}


def _sources(examples_dir: Path) -> Any:
    for parent in (examples_dir, examples_dir.parent, examples_dir.parent.parent):
        for name in ("references/sources.json", "sources.json"):
            source = parent / name
            if source.is_file():
                records = _json(source, [])
                if isinstance(records, dict) and ("assets" in records or "text_sources" in records):
                    return [
                        {**item, "source_group": group}
                        for key, group in (("assets", "教材与名作图像"), ("text_sources", "研究文本"))
                        for item in records.get(key, [])
                        if isinstance(item, dict)
                    ]
                return records
    return []


def _numbered_files(folder: Path, suffix: str) -> list[tuple[int, Path]]:
    pattern = re.compile(r"round_(\d+)" + re.escape(suffix) + r"$")
    return sorted(
        (int(match[1]), path)
        for path in folder.glob("round_*" + suffix)
        if (match := pattern.fullmatch(path.name))
    )


def _ledger(path: Path, identifier: str, label: str, image: str | None = None) -> dict[str, Any]:
    log = _json(path, [])
    metadata = {k: v for k, v in log.items() if k != "strokes"} if isinstance(log, dict) else {}
    # Both the original flat ledgers and the explicit v0.3 metadata envelope
    # are accepted. The nested envelope wins when both specify the same key.
    if isinstance(metadata.get("metadata"), dict):
        nested = metadata.pop("metadata")
        metadata = {**metadata, **nested}
    strokes = log.get("strokes", []) if isinstance(log, dict) else log
    return {"id": identifier, "label": metadata.get("title", label),
            "image": metadata.get("image", metadata.get("image_file", image)),
            "metadata": metadata, "strokes": strokes if isinstance(strokes, list) else []}


def _painting_display_copy(ledger: dict[str, Any], target_samples: int = 160) -> dict[str, Any]:
    """Make a small display copy; the source ledger retains every replay sample.

    Preserve endpoints and both sides of each action/contact transition. The
    remaining budget is spread uniformly over the other samples. Transition
    samples take priority if their count alone exceeds the soft display target.
    """
    fields = set(("u t local_t event_index event_phase phase hand_x hand_y tip_x tip_y "
                  "nominal_pressure_target pressure_target pressure internal_force "
                  "contact_gate commanded_contact contact angle_target azimuth tilt "
                  "ink_load water_load flow_state observed_density observation_x "
                  "observation_y feedback_multiplier speed").split())
    strokes = []
    for raw in ledger["strokes"]:
        if not isinstance(raw, dict) or raw.get("kind") != "painting_stroke":
            strokes.append(raw)
            continue
        source = raw.get("trace", [])
        if not isinstance(source, list):
            strokes.append(raw)
            continue
        trace = [row for row in source if isinstance(row, dict)]
        keep = {0, len(trace) - 1} if trace else set()
        for i in range(1, len(trace)):
            if any(trace[i].get(key) != trace[i - 1].get(key)
                   for key in ("event_index", "commanded_contact")):
                keep.update((i - 1, i))
        candidates = [i for i in range(len(trace)) if i not in keep]
        budget = min(len(candidates), max(0, target_samples - len(keep)))
        if budget:
            keep.update(candidates[min(len(candidates) - 1,
                                       int((j + .5) * len(candidates) / budget))]
                        for j in range(budget))
        shown = [{key: value for key, value in trace[i].items() if key in fields}
                 for i in sorted(keep)]
        compact = {**raw, "trace": shown,
                   "points": [[row["tip_x"], row["tip_y"]] for row in shown
                              if "tip_x" in row and "tip_y" in row],
                   "display_trace": {"source_samples": len(trace),
                                     "shown_samples": len(shown),
                                     "integration_fields_omitted": True}}
        if isinstance(raw.get("driver"), dict):
            compact["driver"] = {key: value for key, value in raw["driver"].items()
                                 if key != "samples"}
        strokes.append(compact)
    return {**ledger, "strokes": strokes,
            "metadata": {**ledger["metadata"], "display_trace_decimated": True,
                         "display_trace_target_samples": target_samples,
                         "full_trace_file": ledger["metadata"].get("full_ledger", "painting_strokes.json")}}


def build_gallery(examples_dir: Path, output: Path) -> Path:
    """Write a self-contained Chinese HTML gallery and return its path.

    Missing images, reviews or stroke logs are omitted rather than preventing
    export. Integer-numbered ``round_*.png`` images are sorted numerically;
    the largest-numbered ``round_*_strokes.json`` supplies the landscape log.
    ``painting_strokes.json`` is initially selected when available, followed by
    ``drive_strokes.json`` and
    ``dynamics_strokes.json``. Logs can be lists or objects with ``strokes``. Backgrounds
    and dimensions belong to each log (or individual stroke), never another
    iteration. ``dynamics_*.png``, ``drive_*.png`` and ``painting_*.png`` have
    separate study sections.
    Coordinates default to PNG pixels; a log can override this using ``width``
    and ``height``, or ``canvas: {width, height}``. Painting traces are reduced
    to a display-only copy; original JSON ledgers are never modified.
    """
    examples_dir, output = Path(examples_dir), Path(output)
    rounds = []
    for number, path in _numbered_files(examples_dir, ".png"):
        item = _png(path)
        if item:
            rounds.append({**item, "round": number, "label": f"第 {number} 轮"})
    studies = []
    for name, title, caption in (
        ("brush_atlas.png", "一笔的状态", "压力、含水、含墨与干湿变化，以实际渲染观察笔迹。"),
        ("physics.png", "接触与扩散", "观察简化模型中的接触、输运与纸面响应。"),
        ("forms.png", "由笔到形", "将笔触组织为山石、树木与远近层次。"),
    ):
        item = _png(examples_dir / name)
        if item:
            studies.append({**item, "title": title, "caption": caption})
    variants = [image for path in sorted(examples_dir.glob("variant_*.png"))
                if (image := _png(path))]
    dynamics = [image for path in sorted(examples_dir.glob("dynamics_*.png"))
                if (image := _png(path))]
    dynamic_titles = {
        "dynamics_01_continuity.png": "连续调制、分笔重起与真实离纸",
        "dynamics_02_boundaries.png": "起笔、停顿与收笔",
        "dynamics_04_ledger.png": "四笔动态记录样张",
    }
    for image in dynamics:
        revision = re.fullmatch(r"dynamics_03_cun_r(\d+)\.png", image["file"])
        image["title"] = (f"皴法中的动态变化 · 第 {revision[1]} 轮" if revision
                          else dynamic_titles.get(image["file"], image["file"]))
    drive = [image for path in sorted(examples_dir.glob("drive_*.png"))
             if (image := _png(path))]
    drive_titles = {
        "drive_01_mechanics.png": "同一动作，不同毛束跟随",
        "drive_02_layers.png": "机械、供墨与纸面响应的分层对照",
        "drive_03_events.png": "动作事件与驱动响应",
        "drive_04_ledger.png": "动作驱动样张与笔触账本",
        "drive_03_cun_r0.png": "皴法动作语法 · 第 0 轮",
        "drive_03_cun_r1.png": "皴法动作语法 · 第 1 轮",
    }
    for image in drive:
        image["title"] = drive_titles.get(image["file"], image["file"])
    painting = [image for path in sorted(examples_dir.glob("painting_*.png"))
                if (image := _png(path))]
    painting_titles = {
        "painting_00_form.png": "山石形势与共用体面",
        "painting_01_r0.png": "起、行、转、收与山石成形 · 第 0 轮",
        "painting_01_r1.png": "起、行、转、收与山石成形 · 第 1 轮",
        "painting_02_process.png": "同一画面中的落笔与承接",
        "painting_03_boundaries.png": "起落与转折如何留在画中",
        "painting_04_feedback.png": "画面反馈如何改变同一笔",
        "painting_05_comparison.png": "三轮成形比较",
    }
    for image in painting:
        revision = re.fullmatch(r"painting_01_r(\d+)\.png", image["file"])
        image["title"] = (f"起、行、转、收与山石成形 · 第 {revision[1]} 轮" if revision
                          else painting_titles.get(image["file"], image["file"]))
    painting_reviews = []
    for name, title in (
        ("painting_process.md", "每一笔如何成形 · 模型与笔序"),
        ("painting_feedback.md", "视觉观察与修改依据"),
        ("painting_validation.md", "验证结果与当前边界"),
        ("painting_visual_review.md", "独立视觉审查"),
    ):
        document = examples_dir.parent / "docs" / name
        if document.is_file():
            painting_reviews.append({"title": title, "file": name,
                                     "text": document.read_text(encoding="utf-8")})
    drive_reviews = []
    for name, title in (
        ("drive_response_theory.md", "离散动作与有状态响应 · 模型说明"),
        ("drive_feedback.md", "视觉观察与调整"),
        ("drive_numerical_review.md", "数值验证与边界"),
        ("drive_visual_review.md", "独立视觉审查"),
    ):
        document = examples_dir.parent / "docs" / name
        if document.is_file():
            drive_reviews.append({"title": title, "file": name,
                                  "text": document.read_text(encoding="utf-8")})
    logs = _numbered_files(examples_dir, "_strokes.json")
    ledgers = []
    if logs:
        number, log_path = logs[-1]
        ledgers.append(_ledger(log_path, f"round-{number}", f"山水 · 第 {number} 轮", f"round_{number}.png"))
    dynamic_log = examples_dir / "dynamics_strokes.json"
    if dynamic_log.is_file():
        ledgers.append(_ledger(dynamic_log, "dynamics", "连续一笔 · 动态样张"))
    drive_log = examples_dir / "drive_strokes.json"
    if drive_log.is_file():
        ledgers.append(_ledger(drive_log, "drive", "离散动作 · 驱动与跟随"))
    painting_log = examples_dir / "painting_strokes.json"
    if painting_log.is_file():
        ledgers.append(_painting_display_copy(
            _ledger(painting_log, "painting", "每一笔正在成形 · 画面与笔势")))
    payload = {"rounds": rounds, "studies": studies, "variants": variants,
               "dynamics": dynamics, "drive": drive, "drive_reviews": drive_reviews,
               "painting": painting, "painting_reviews": painting_reviews,
               "ledgers": ledgers,
               "reviews": _json(examples_dir / "reviews.json", []),
               "sources": _sources(examples_dir)}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Safe even when a source title or tag contains HTML/script delimiters.
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_TEMPLATE.replace("__GALLERY_DATA__", encoded), encoding="utf-8")
    return output


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>水墨山水 · 计算笔记</title>
<style>
:root{color-scheme:light;--paper:#f5f2eb;--panel:#fffdf7;--ink:#292d29;--muted:#666b62;--line:#d9dacd;--accent:#795c42;--signal:#a74836}
[hidden]{display:none!important}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}button,input,select{font:inherit;color:inherit}button,select,input[type=number],input[type=text]{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:7px 12px}button{cursor:pointer}button:hover{border-color:var(--accent)}button[aria-pressed=true],button.primary{background:var(--ink);border-color:var(--ink);color:var(--panel)}button:disabled{opacity:.4;cursor:default}input[type=range]{accent-color:var(--accent);min-width:130px}input[type=checkbox]{accent-color:var(--accent)}input[type=number]{width:100px}input[type=text]{width:180px;max-width:100%}a{color:var(--accent);text-underline-offset:4px}main{width:min(1232px,100%);margin:auto;padding:48px 40px 64px}header{display:flex;justify-content:space-between;align-items:flex-end;gap:32px;padding-bottom:28px;border-bottom:1px solid var(--line)}h1,h2,h3,p,figure{margin:0}h1,h2{font-family:"Iowan Old Style","Songti SC",SimSun,serif;font-weight:400}h1{font-size:42px;letter-spacing:.14em}h2{font-size:28px;letter-spacing:.08em}h3{font-size:17px;font-weight:500}.eyebrow{font-size:11px;letter-spacing:.18em;color:var(--muted);margin-bottom:10px}.intro{max-width:485px;color:var(--muted);font-size:14px}section{padding-top:38px}.section-head{display:flex;justify-content:space-between;align-items:baseline;gap:24px;flex-wrap:wrap;margin-bottom:17px}.section-index{font-size:12px;color:var(--accent);margin-right:12px;letter-spacing:.06em}.note,.caption{font-size:13px;color:var(--muted)}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;margin-bottom:14px}.controls label{display:flex;align-items:center;gap:8px}.stage{position:relative;overflow:hidden;background:var(--panel);border:1px solid var(--line)}.stage img{display:block;width:100%;height:auto}.stage .overlay-image{position:absolute;inset:0;width:100%;height:100%;object-fit:fill}.split-line{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--signal);pointer-events:none}.split-line:after{content:"↔";position:absolute;top:48%;left:-17px;width:35px;height:35px;line-height:35px;text-align:center;background:var(--panel);border:1px solid var(--signal);border-radius:50%;color:var(--signal)}.corner{position:absolute;top:13px;padding:4px 9px;background:#fffdf7e8;font-size:12px;pointer-events:none}.corner.left{left:13px}.corner.right{right:13px}.compare-tools{display:flex;align-items:center;gap:12px;margin-top:12px}.compare-tools input{flex:1}.review-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:22px;margin-top:22px}.review{border-top:1px solid var(--line);padding-top:13px}.review h3{margin-bottom:6px}.review p,.review li{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.review ul{padding-left:20px;margin:6px 0}.study-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.study img{width:100%;display:block;background:var(--panel);border:1px solid var(--line)}.study figcaption{padding-top:10px}.study:first-child:nth-last-child(3){grid-column:1/-1}.stroke-layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:20px;align-items:start}.canvas-wrap{background:var(--panel);border:1px solid var(--line);width:100%;overflow:hidden}.canvas-wrap canvas{display:block;width:100%;height:auto}.inspector{min-width:0}.inspector canvas{width:100%;height:155px;display:block;background:var(--panel);border:1px solid var(--line);margin-top:12px}.inspector dl{display:grid;grid-template-columns:78px minmax(0,1fr);font-size:12px;margin:12px 0;gap:5px 10px}.inspector dt{color:var(--muted)}.inspector dd{margin:0;overflow-wrap:anywhere}.playback{display:flex;gap:12px;align-items:center;margin-top:12px}.playback input{flex:1;min-width:60px}.tabular{font-variant-numeric:tabular-nums}.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:20px}.variants img{width:100%;display:block;border:1px solid var(--line)}.variants figcaption{font-size:12px;color:var(--muted);padding-top:6px}.parameters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;padding-top:8px}.parameters h3{margin-bottom:5px}.parameters p{font-size:13px;color:var(--muted)}.sources{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:18px 32px}.sources li{font-size:13px;overflow-wrap:anywhere}.sources p{color:var(--muted);margin-top:3px}.source-type{display:block;font-size:11px;color:var(--muted)}footer{margin-top:42px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}[hidden]{display:none!important}.empty{padding:32px;text-align:center;color:var(--muted);border:1px solid var(--line)}:focus-visible{outline:2px solid var(--signal);outline-offset:3px}
@media(max-width:800px){main{padding:25px 20px 38px}header{display:block}h1{font-size:34px}.intro{margin-top:15px;max-width:none}.stroke-layout{grid-template-columns:1fr}.inspector{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}.inspector h3,.inspector .controls,.inspector>.note{grid-column:1/-1}.inspector canvas{height:180px}.parameters{grid-template-columns:1fr}.sources{grid-template-columns:1fr}}
@media(max-width:520px){main{padding:22px 14px 34px}h1{font-size:28px}h2{font-size:24px}.study-grid{grid-template-columns:1fr}.study:first-child:nth-last-child(3){grid-column:auto}.section-head{gap:8px}.inspector{display:block}.controls{gap:9px}.controls label{flex-wrap:wrap}.review-grid{grid-template-columns:1fr}.compare-tools{font-size:12px}.playback{flex-wrap:wrap}.playback input{min-width:150px}}
.inspector dl.stroke-purpose{display:block;border-left:2px solid var(--accent);padding-left:12px}.stroke-purpose dt{margin-top:9px}.stroke-purpose dd{font-size:13px}.dynamic-panel{margin-top:22px;border-top:1px solid var(--line);padding-top:18px}.dynamic-panel h3{margin-bottom:8px}.trace-curves{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px}.trace-curves canvas{display:block;width:100%;height:210px;background:var(--panel)}.sample-controls{display:flex;align-items:center;gap:12px;margin-top:12px}.sample-controls input{flex:1;min-width:80px}.sample-state{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px 16px;margin:14px 0}.sample-state dt{font-size:12px;color:var(--muted)}.sample-state dd{font-size:13px;margin:0;font-variant-numeric:tabular-nums}.contact-note{margin-top:9px;overflow-wrap:anywhere}.spec-details{margin-top:16px;font-size:13px}.spec-details summary{cursor:pointer;color:var(--accent)}.spec-details pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;background:var(--panel);border:1px solid var(--line);max-height:420px;overflow:auto}.controls select{max-width:100%}.dynamic-figures{display:grid;gap:24px}.dynamic-figures figure{min-width:0}@media(max-width:600px){.trace-curves{grid-template-columns:1fr}.sample-state{grid-template-columns:repeat(2,minmax(0,1fr))}.sample-controls{flex-wrap:wrap}}
</style>
</head>
<body>
<main>
<header><div><p class="eyebrow">INK LANDSCAPE / COMPUTATIONAL STUDIES</p><h1>水墨山水 · 计算笔记</h1></div><p class="intro">每一笔在已有的形与墨中起、行、转、收。记录笔势、接触、水墨和画面如何共同演变，把每次调整留在图像与笔序中，供观察、比较和复现。</p></header>
<section id="rounds-section">
<div class="section-head"><h2><span class="section-index">01</span>阶段对照</h2><p class="note">图像为各轮真实渲染；切换不会重新生成。</p></div>
<div class="controls" id="round-controls" aria-label="选择渲染轮次"></div>
<div class="stage" id="stage"><img id="round-image" alt="水墨山水阶段渲染"><img id="before-image" class="overlay-image" alt="初始轮次对照" hidden><div id="split-line" class="split-line" hidden></div><span id="left-label" class="corner left" hidden>第 0 轮</span><span id="right-label" class="corner right" hidden>第 2 轮</span></div>
<div class="compare-tools" id="compare-tools" hidden><span>初始</span><input id="compare-range" type="range" min="0" max="100" value="50" aria-label="拖动比较初始与最终轮次"><span>最终</span></div>
<div id="reviews" class="review-grid"></div>
</section>
<section id="studies-section"><div class="section-head"><h2><span class="section-index">02</span>从一笔到山形</h2></div><div id="studies" class="study-grid"></div></section>
<section id="dynamics-section"><div class="section-head"><h2><span class="section-index">03</span>连续一笔与起收笔</h2><p class="note">样张为真实渲染；一笔内的函数变化保留连续状态。</p></div><div id="dynamics" class="dynamic-figures"></div></section>
<section id="drive-section"><div class="section-head"><h2><span class="section-index">04</span>离散动作与跟随响应</h2><p class="note">动作给出驱动目标；毛束、供墨与纸面按各自的状态响应。</p></div><div id="drive" class="dynamic-figures"></div><div id="drive-reviews"></div></section>
<section id="painting-section"><div class="section-head"><h2><span class="section-index">05</span>每一笔正在成形</h2><p class="note">当前墨迹的密度参与下一次提按；起落与转折放在山石形势中观察。</p></div><div id="painting" class="dynamic-figures"></div><div id="painting-reviews"></div></section>
<section id="strokes-section">
<div class="section-head"><h2><span class="section-index">04</span>笔序与剖面</h2><p class="note">轨迹与轮廓的几何示意；不会重新计算墨液。</p></div>
<div class="controls"><label>笔触记录 <select id="ledger-select" aria-label="选择笔触账本"></select></label><label>标签前缀 <input id="tag-filter" type="text" list="tags" placeholder="全部笔触" aria-label="按标签前缀过滤笔触"><datalist id="tags"></datalist></label><label><input type="checkbox" id="show-result" checked> 叠加结果图</label><span id="filtered-count" class="note tabular" aria-live="polite"></span></div>
<p id="ledger-note" class="note" style="margin-bottom:14px" hidden></p>
<div class="stroke-layout"><div><div class="canvas-wrap"><canvas id="paths" width="1000" height="700" role="img" aria-label="按绘制顺序显示的笔触路径，圆点表示选中轨迹的首点"></canvas></div><div class="playback"><button id="play" type="button">播放笔序</button><input id="progress" type="range" min="0" max="1" value="0" aria-label="已显示的笔触数量"><span id="progress-text" class="note tabular">0 / 0</span></div><p class="note" style="margin-top:9px">细线：已显示的路径；朱色：选中路径与轨迹首点。绿圈表示当前笔尖，实心与空心对应触纸和离纸：动作账本依据触纸/提离指令代理，连续函数账本依据 trace.contact。褐色虚线和空心方框表示手部驱动路径与位置。正载荷区间和断墨不用于判断分笔。</p></div>
<aside class="inspector"><h3>这一笔</h3><div class="controls" style="margin-top:9px"><button id="prev-stroke" type="button" aria-label="上一笔">←</button><label>序号 <input id="stroke-index" type="number" min="1" value="1" step="1"></label><button id="next-stroke" type="button" aria-label="下一笔">→</button></div><dl id="stroke-purpose" class="stroke-purpose" hidden></dl><canvas id="detail" width="600" height="310" role="img" aria-label="选中笔触的路径局部放大，圆点表示轨迹首点"></canvas><dl id="stroke-fields"></dl><p class="note">数值沿用日志单位；未记录的量不推测。局部图按路径范围自动缩放。</p></aside></div>
<div id="dynamic-panel" class="dynamic-panel" hidden><h3 id="trace-title">一笔内部的连续变化</h3><p id="trace-description" class="note">曲线直接读取 trace 采样值。滑块只查看记录，不把函数片段拆成新的起笔或收笔。</p><div class="sample-controls"><label for="trace-index">采样位置</label><input id="trace-index" type="range" min="0" max="1" value="0" step="1"><span id="trace-position" class="note tabular" aria-live="polite"></span></div><dl id="sample-state" class="sample-state"></dl><p id="contact-events" class="note contact-note"></p><p id="driver-events" class="note contact-note" hidden></p><div class="trace-curves"><canvas id="trace-pressure" role="img" aria-label="实际采样压力随归一化笔程 u 的变化"></canvas><canvas id="trace-speed" role="img" aria-label="实际采样速度随归一化笔程 u 的变化"></canvas></div><details class="spec-details"><summary id="spec-summary">查看这一笔的函数定义与边界参数（原始 spec）</summary><pre id="stroke-spec"></pre></details></div>
</section>
<section id="variants-section"><div class="section-head"><h2><span class="section-index">04</span>固定规则，改变随机种子</h2></div><div class="variants" id="variants"></div></section>
<section><div class="section-head"><h2><span class="section-index">05</span>如何阅读参数</h2></div><div class="parameters"><div><h3>接触与提按</h3><p>路径定义行笔方向，压力与笔宽影响接触范围。起笔、行笔、收笔的压力变化形成粗细与端部形态。</p></div><div><h3>水、墨与纸</h3><p>含水与含墨控制输运和沉积；干湿差异影响边缘、渗化和断续。这里的参数属于可计算的简化模型。</p></div><div><h3>笔序与形态</h3><p>一笔放入何处、朝向哪里，以及它与邻笔的疏密关系，共同构成山石与树木。随机性应依附于这些约束。</p></div></div></section>
<section id="sources-section"><div class="section-head"><h2><span class="section-index">06</span>参考与边界</h2><p class="note">来源按项目记录展示；名作与教材不等于本模型的验证结果。</p></div><ul class="sources" id="sources"></ul><p id="sources-empty" class="note" hidden>本次导出未包含来源目录。请参阅代码包中的研究说明及 references/sources.json。</p></section>
<footer><span>离线可浏览 · 图像、参数与路径随本文件保存</span><span>程序化实验，不声称复现真实毛笔或传统笔法的全部经验。</span></footer>
</main>
<script id="gallery-data" type="application/json">__GALLERY_DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("gallery-data").textContent);
const $=id=>document.getElementById(id);
const node=(name,cls,text)=>{const n=document.createElement(name);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n};
const fmt=value=>{if(value===null||value===undefined)return "未记录";if(typeof value==="number")return Number.isInteger(value)?String(value):Number(value.toFixed(4)).toString();if(Array.isArray(value)){if(!value.length)return "[]";if(value.every(v=>typeof v==="number")){if(value.length<=3)return value.map(fmt).join(", ");return fmt(Math.min(...value))+" → "+fmt(Math.max(...value))+"（"+value.length+" 点）";}return JSON.stringify(value);}if(typeof value==="object")return JSON.stringify(value);return String(value)};
const rounds=DATA.rounds||[];
let comparing=false;
const initial=rounds[0],final=rounds[rounds.length-1];
function setRound(item){comparing=false;$("round-image").src=item.src;$("round-image").alt=item.label+"水墨山水渲染";["before-image","split-line","left-label","right-label","compare-tools"].forEach(id=>$(id).hidden=true);$("round-controls").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.round===String(item.round))));}
rounds.forEach(item=>{const b=node("button","",item.label);b.type="button";b.dataset.round=item.round;b.onclick=()=>setRound(item);$("round-controls").appendChild(b)});
if(final)setRound(final);else $("rounds-section").hidden=true;
if(initial&&final&&initial.round!==final.round){const b=node("button","","拖动对照");b.type="button";b.setAttribute("aria-pressed","false");b.onclick=()=>{setRound(final);comparing=true;$("before-image").src=initial.src;["before-image","split-line","left-label","right-label","compare-tools"].forEach(id=>$(id).hidden=false);$("left-label").textContent=initial.label;$("right-label").textContent=final.label;$("round-controls").querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));split()};$("round-controls").appendChild(b)}
function split(){const value=Number($("compare-range").value);$("before-image").style.clipPath="inset(0 "+(100-value)+"% 0 0)";$("split-line").style.left=value+"%";}
$("compare-range").addEventListener("input",split);
let drag=false;$("stage").addEventListener("pointerdown",e=>{if(!comparing)return;drag=true;$("stage").setPointerCapture(e.pointerId);moveSplit(e)});$("stage").addEventListener("pointermove",e=>{if(drag)moveSplit(e)});$("stage").addEventListener("pointerup",()=>drag=false);$("stage").addEventListener("pointercancel",()=>drag=false);
function moveSplit(e){const rect=$("stage").getBoundingClientRect();$("compare-range").value=Math.max(0,Math.min(100,(e.clientX-rect.left)/rect.width*100));split()}
function imageFigure(item,title,caption){const f=node("figure","study"),im=node("img");im.src=item.src;im.alt=title;im.loading="lazy";f.appendChild(im);const cap=node("figcaption");cap.appendChild(node("h3","",title));if(caption)cap.appendChild(node("p","caption",caption));f.appendChild(cap);return f}
DATA.studies.forEach(item=>$("studies").appendChild(imageFigure(item,item.title,item.caption)));if(!DATA.studies.length)$("studies-section").hidden=true;
DATA.variants.forEach((item,i)=>$("variants").appendChild(imageFigure(item,item.file,"")));if(!DATA.variants.length)$("variants-section").hidden=true;
(DATA.dynamics||[]).forEach(item=>$("dynamics").appendChild(imageFigure(item,item.title||item.file,item.caption||"")));if(!DATA.dynamics?.length)$("dynamics-section").hidden=true;
(DATA.drive||[]).forEach(item=>$("drive").appendChild(imageFigure(item,item.title||item.file,item.caption||"")));
(DATA.drive_reviews||[]).forEach(review=>{const details=node("details","spec-details"),summary=node("summary","",review.title||review.file),content=node("pre","",review.text||"");details.appendChild(summary);details.appendChild(content);$("drive-reviews").appendChild(details)});
if(!DATA.drive?.length&&!DATA.drive_reviews?.length)$("drive-section").hidden=true;
(DATA.painting||[]).forEach(item=>$("painting").appendChild(imageFigure(item,item.title||item.file,item.caption||"")));
(DATA.painting_reviews||[]).forEach(review=>{const details=node("details","spec-details"),summary=node("summary","",review.title||review.file),content=node("pre","",review.text||"");details.appendChild(summary);details.appendChild(content);$("painting-reviews").appendChild(details)});
if(!DATA.painting?.length&&!DATA.painting_reviews?.length)$("painting-section").hidden=true;
function simpleText(value){if(typeof value==="string")return value;if(Array.isArray(value))return value.map(simpleText).join("；");if(value&&typeof value==="object")return Object.entries(value).map(([k,v])=>k+"："+simpleText(v)).join("；");return fmt(value)}
let reviews=DATA.reviews;if(reviews&&!Array.isArray(reviews))reviews=reviews.rounds||reviews.reviews||Object.entries(reviews).map(([k,v])=>typeof v==="object"&&!Array.isArray(v)?{title:k,...v}:{title:k,summary:v});
if(Array.isArray(reviews))reviews.forEach((review,i)=>{const r=typeof review==="string"?{summary:review}:review;if(!r||typeof r!=="object")return;const a=node("article","review");a.appendChild(node("h3","",r.title||r.name||(r.round!==undefined?"第 "+r.round+" 轮":"观察 "+(i+1))));const entries=Object.entries(r).filter(([k])=>!["title","name","round","image","path","filename"].includes(k));const labels={summary:"观察",observations:"观察",issues:"问题",changes:"调整",feedback:"反馈",next:"下一步",limitations:"局限",metrics:"记录",verdict:"结论",improvements:"改进",remaining:"待改进",decision:"处理",focus:"关注点",status:"状态"};entries.forEach(([k,v])=>{if(v===null||v===undefined)return;const p=node("p","",(labels[k]||k)+"："+simpleText(v));a.appendChild(p)});$("reviews").appendChild(a)});
const point=p=>Array.isArray(p)?[Number(p[0]),Number(p[1])]:p&&typeof p==="object"?[Number(p.x??p[0]),Number(p.y??p[1])]:null;
const numeric=v=>v!==null&&v!==undefined&&v!==""&&Number.isFinite(Number(v));
function normalizeTrace(raw){
  if(Array.isArray(raw))return raw.filter(v=>v&&typeof v==="object"&&!Array.isArray(v));
  if(!raw||typeof raw!=="object")return [];
  if(Array.isArray(raw.samples))return normalizeTrace(raw.samples);
  const columns=Object.entries(raw).filter(([key,value])=>Array.isArray(value));
  if(!columns.length)return [];
  const count=Math.max(...columns.map(([key,values])=>values.length));
  return Array.from({length:count},(_,index)=>Object.fromEntries(columns.map(([key,values])=>[key,values[index]])));
}
function normalizeStrokes(values){return(values||[]).filter(raw=>raw&&typeof raw==="object").map((raw,index)=>{
  const trace=normalizeTrace(raw.trace),pts=raw.points||raw.path||raw.xy||raw.trajectory||trace.map(v=>[v.tip_x,v.tip_y]);
  return{raw,index,trace,tag:String(raw.tag||raw.category||raw.name||raw.kind||"未标注"),points:Array.isArray(pts)?pts.map(point).filter(p=>p&&p.every(Number.isFinite)):[]};
}).filter(s=>s.points.length>0)}
const ledgers=(DATA.ledgers||[]).filter(item=>Array.isArray(item.strokes)&&item.strokes.length);
const imageCatalog=[...rounds,...(DATA.studies||[]),...(DATA.variants||[]),...(DATA.dynamics||[]),...(DATA.drive||[]),...(DATA.painting||[])];
let ledger=null,strokes=[],filtered=[],selected=0,shown=0,playing=false,lastTime=0,fraction=0;
let width=1000,height=700,background=null,backgroundFile=null,sampleIndex=0,trace=[];
function filename(value){if(value&&typeof value==="object")value=value.file||value.image;return typeof value==="string"?value.split(/[\\/]/).pop():null}
function surfaceFor(s){
  const raw=s?.raw||{},meta=ledger?.metadata||{},canvas=raw.canvas||{},shared=meta.canvas||meta.size||{};
  const file=filename(raw.image||raw.image_file||canvas.image||ledger?.image||shared.image),image=imageCatalog.find(item=>item.file===file);
  return{image,width:Number(canvas.width||raw.canvas_width||meta.width||shared.width||(Array.isArray(shared)?shared[0]:0)||image?.width||1000),height:Number(canvas.height||raw.canvas_height||meta.height||shared.height||(Array.isArray(shared)?shared[1]:0)||image?.height||700)};
}
function setSurface(s){
  const surface=surfaceFor(s);width=surface.width;height=surface.height;
  const image=surface.image;$("show-result").disabled=!image;
  if(!image){background=null;backgroundFile=null;return}
  if(image.file===backgroundFile)return;
  backgroundFile=image.file;background=new Image();background.onload=renderPaths;background.src=image.src;
}
function dimensions(canvas,aspect){
  const cssWidth=Math.max(1,canvas.getBoundingClientRect().width||canvas.parentElement.getBoundingClientRect().width),ratio=Math.min(window.devicePixelRatio||1,2);
  const cssHeight=canvas.id==="detail"?155:canvas.id.startsWith("trace-")?210:Math.max(1,cssWidth/aspect);
  const w=Math.max(1,Math.round(cssWidth*ratio)),h=Math.max(1,Math.round(cssHeight*ratio));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}return[w,h,ratio];
}
function drawPath(ctx,s,scaleX,scaleY,offsetX=0,offsetY=0){
  if(!s?.points.length)return;ctx.beginPath();s.points.forEach((p,i)=>{const x=p[0]*scaleX+offsetX,y=p[1]*scaleY+offsetY;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  if(s.points.length===1){const p=s.points[0];ctx.lineTo(p[0]*scaleX+offsetX+.1,p[1]*scaleY+offsetY+.1)}ctx.stroke();
}
function hasDriver(s){return ["gesture_stroke","painting_stroke"].includes(s?.raw?.kind)}
function handPath(s){return{points:(s?.trace||[]).filter(row=>numeric(row.hand_x)&&numeric(row.hand_y)).map(row=>[Number(row.hand_x),Number(row.hand_y)])}}
function drawHand(ctx,s,sx,sy,ox=0,oy=0){
  if(!hasDriver(s))return;ctx.save();ctx.strokeStyle="#795c42";ctx.lineWidth=Math.max(1,ctx.lineWidth*.8);ctx.setLineDash([7,5]);drawPath(ctx,handPath(s),sx,sy,ox,oy);ctx.restore();
}
function markStart(ctx,s,sx,sy,ox=0,oy=0,r=4){
  if(!s?.points.length)return;const first=s.points[0],last=s.points[s.points.length-1];
  ctx.beginPath();ctx.arc(first[0]*sx+ox,first[1]*sy+oy,r,0,Math.PI*2);ctx.fill();
  if(s.points.length>1){const prev=s.points[s.points.length-2],angle=Math.atan2((last[1]-prev[1])*sy,(last[0]-prev[0])*sx),x=last[0]*sx+ox,y=last[1]*sy+oy;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-2.4*r*Math.cos(angle-.4),y-2.4*r*Math.sin(angle-.4));ctx.lineTo(x-2.4*r*Math.cos(angle+.4),y-2.4*r*Math.sin(angle+.4));ctx.closePath();ctx.fill()}
}
function contactValue(value){if(value===true||value===false)return value;if(numeric(value))return Number(value)>0;return null}
function markSample(ctx,sx,sy,ox=0,oy=0,r=6){
  const sample=trace[sampleIndex];if(!sample||!numeric(sample.tip_x)||!numeric(sample.tip_y))return;
  const x=Number(sample.tip_x)*sx+ox,y=Number(sample.tip_y)*sy+oy;ctx.strokeStyle="#3e5d56";ctx.fillStyle="#3e5d56";ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke();if(contactValue(hasDriver(filtered[selected])?sample.commanded_contact:sample.contact)===true){ctx.beginPath();ctx.arc(x,y,r*.46,0,Math.PI*2);ctx.fill()}
  if(numeric(sample.hand_x)&&numeric(sample.hand_y)){const hx=Number(sample.hand_x)*sx+ox,hy=Number(sample.hand_y)*sy+oy;ctx.strokeStyle="#795c42";ctx.strokeRect(hx-r*.75,hy-r*.75,r*1.5,r*1.5)}
}
function sameSurface(s){const other=surfaceFor(s);return other.width===width&&other.height===height&&(other.image?.file||null)===backgroundFile}
function renderPaths(){
  if(!strokes.length)return;const canvas=$("paths"),[w,h]=dimensions(canvas,width/height),ctx=canvas.getContext("2d"),sx=w/width,sy=h/height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#fffdf7";ctx.fillRect(0,0,w,h);
  if($("show-result").checked&&background?.complete&&background.naturalWidth)ctx.drawImage(background,0,0,w,h);
  ctx.lineCap="round";ctx.lineJoin="round";ctx.strokeStyle=$("show-result").checked?"rgba(62,93,86,.52)":"rgba(50,57,47,.54)";ctx.lineWidth=Math.max(1,w/1200);
  for(let i=0;i<Math.min(shown,filtered.length);i++)if(sameSurface(filtered[i]))drawPath(ctx,filtered[i],sx,sy);
  const chosen=filtered[selected];if(chosen){ctx.strokeStyle="#a74836";ctx.fillStyle="#a74836";ctx.lineWidth=Math.max(1.5,w/650);drawHand(ctx,chosen,sx,sy);drawPath(ctx,chosen,sx,sy);markStart(ctx,chosen,sx,sy,0,0,Math.max(3,w/260));markSample(ctx,sx,sy,0,0,Math.max(5,w/160))}
  $("progress").value=shown;$("progress-text").textContent=shown+" / "+filtered.length;
}
function addField(name,value){$("stroke-fields").appendChild(node("dt","",name));$("stroke-fields").appendChild(node("dd","tabular",fmt(value)))}
function renderDetail(){
  const s=filtered[selected],canvas=$("detail"),[w,h]=dimensions(canvas,2),ctx=canvas.getContext("2d");ctx.clearRect(0,0,w,h);ctx.fillStyle="#fffdf7";ctx.fillRect(0,0,w,h);if(!s)return;
  const bounds=[...s.points,...handPath(s).points],xs=bounds.map(p=>p[0]),ys=bounds.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const margin=28,scale=Math.min((w-2*margin)/Math.max(1,x1-x0),(h-2*margin)/Math.max(1,y1-y0)),ox=(w-(x1-x0)*scale)/2-x0*scale,oy=(h-(y1-y0)*scale)/2-y0*scale;
  ctx.strokeStyle="#a74836";ctx.fillStyle="#a74836";ctx.lineWidth=3;ctx.lineCap="round";ctx.lineJoin="round";drawHand(ctx,s,scale,scale,ox,oy);drawPath(ctx,s,scale,scale,ox,oy);markStart(ctx,s,scale,scale,ox,oy,5);markSample(ctx,scale,scale,ox,oy,8);
}
function traceCurve(canvas,key,label,targetKey=null,axis="u",options={}){
  const[w,h,ratio]=dimensions(canvas,2),ctx=canvas.getContext("2d");
  const series=options.series||[{key:targetKey,name:"驱动目标",color:"#795c42",dash:[6,4]},{key,name:"实际响应",color:key==="pressure"?"#a74836":"#3e5d56",dash:[]}].filter(line=>line.key);
  const separate=Boolean(options.separateScales),samples=trace.filter(row=>numeric(row[axis])&&series.some(line=>numeric(row[line.key])));
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#fffdf7";ctx.fillRect(0,0,w,h);ctx.fillStyle="#292d29";ctx.font=12*ratio+"px system-ui";ctx.fillText(label,12*ratio,20*ratio);
  if(!samples.length){ctx.fillStyle="#666b62";ctx.fillText("trace 未记录此曲线",12*ratio,55*ratio);return}
  const us=samples.map(row=>Number(row[axis]));let x0=Math.min(...us),x1=Math.max(...us);if(x1===x0)x1=x0+1;
  function bounds(lines){const values=lines.flatMap(line=>samples.filter(row=>numeric(row[line.key])).map(row=>Number(row[line.key])));let lo=values.length?Math.min(...values):0,hi=values.length?Math.max(...values):1;if(hi===lo){const pad=Math.max(.01,Math.abs(lo)*.05);lo-=pad;hi+=pad}return[lo,hi]}
  const common=bounds(series);series.forEach(line=>{line.range=separate?bounds([line]):common});
  let legendX=12*ratio,legendY=39*ratio;
  if(series.length>1)series.forEach(line=>{const extent=ctx.measureText(line.name||line.key).width+43*ratio;if(legendX+extent>w-12*ratio&&legendX>12*ratio){legendX=12*ratio;legendY+=18*ratio}ctx.strokeStyle=line.color;ctx.lineWidth=1.8*ratio;ctx.setLineDash((line.dash||[]).map(v=>v*ratio));ctx.beginPath();ctx.moveTo(legendX,legendY-4*ratio);ctx.lineTo(legendX+20*ratio,legendY-4*ratio);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=line.color;ctx.fillText(line.name||line.key,legendX+25*ratio,legendY);legendX+=extent});
  const left=55*ratio,right=w-(separate?55:20)*ratio,top=series.length>1?legendY+18*ratio:35*ratio,bottom=h-32*ratio,x=value=>left+(value-x0)/(x1-x0)*(right-left),y=(value,line)=>bottom-(value-line.range[0])/(line.range[1]-line.range[0])*(bottom-top);
  ctx.strokeStyle="#d9dacd";ctx.lineWidth=ratio;ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);if(separate)ctx.lineTo(right,top);ctx.stroke();
  ctx.fillStyle=separate?series[0].color:"#666b62";ctx.textAlign="right";ctx.fillText(fmt(series[0].range[1]),left-7*ratio,top+4*ratio);ctx.fillText(fmt(series[0].range[0]),left-7*ratio,bottom+4*ratio);
  if(separate&&series[1]){ctx.fillStyle=series[1].color;ctx.textAlign="left";ctx.fillText(fmt(series[1].range[1]),right+7*ratio,top+4*ratio);ctx.fillText(fmt(series[1].range[0]),right+7*ratio,bottom+4*ratio)}
  ctx.fillStyle="#666b62";ctx.textAlign="left";ctx.fillText(fmt(x0),left,bottom+19*ratio);ctx.textAlign="right";ctx.fillText(fmt(x1),right,bottom+19*ratio);ctx.textAlign="center";ctx.fillText(axis==="local_t"?"本笔 t（s）":axis==="t"?"t（s）":"u",(left+right)/2,bottom+19*ratio);
  ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
  series.forEach(line=>{ctx.strokeStyle=line.color;ctx.lineWidth=1.8*ratio;ctx.setLineDash((line.dash||[]).map(v=>v*ratio));ctx.beginPath();let active=false;samples.forEach(row=>{if(!numeric(row[line.key])){active=false;return}const px=x(Number(row[axis])),py=y(Number(row[line.key]),line);active?ctx.lineTo(px,py):ctx.moveTo(px,py);active=true});ctx.stroke()});
  ctx.setLineDash([]);const selectedSample=trace[sampleIndex];if(selectedSample&&numeric(selectedSample[axis])){const px=x(Number(selectedSample[axis]));ctx.strokeStyle="#795c42";ctx.lineWidth=ratio;ctx.beginPath();ctx.moveTo(px,top);ctx.lineTo(px,bottom);ctx.stroke();series.forEach(line=>{if(numeric(selectedSample[line.key])){ctx.fillStyle=line.color;ctx.beginPath();ctx.arc(px,y(Number(selectedSample[line.key]),line),3.3*ratio,0,Math.PI*2);ctx.fill()}})}
  ctx.restore();ctx.textAlign="left";
}
function driverEvents(s){return s?.raw?.driver?.events||s?.raw?.program?.events||s?.raw?.program?.actions||[]}
function eventName(event){return event?.kind||event?.action||event?.name||"未记录"}
function traceEvents(){
  const events=[],gesture=hasDriver(filtered[selected]),key=gesture?"commanded_contact":"contact";let previous=null;
  trace.forEach(row=>{const state=contactValue(row[key]);if(state===null)return;if(state!==previous){const at=numeric(row.t)?"t="+fmt(Number(row.t)):numeric(row.u)?"u="+fmt(Number(row.u)):"时刻未记录";events.push((previous===null?"初始":"")+(state?"触纸":"离纸")+" "+at);previous=state}});
  $("contact-events").textContent=gesture?(events.length?"指令接触边界（touch/lift 代理，记录中的采样时刻）："+events.join("；")+"。正载荷区间数和断墨不等于分笔数。":"trace 未记录指令接触状态；不从正载荷或断墨推测分笔。"):events.length?"接触边界（记录中的采样时刻）："+events.join("；")+"。未在函数分段点增添起收边界。":"trace 未记录接触状态；不从压力曲线推测触纸或离纸时刻。";
}
function renderSample(){
  if(!trace.length)return;sampleIndex=Math.max(0,Math.min(trace.length-1,sampleIndex));const sample=trace[sampleIndex],s=filtered[selected],gesture=hasDriver(s),painting=s?.raw?.kind==="painting_stroke",timeAxis=painting&&trace.every(row=>numeric(row.local_t))?"local_t":"t";$("trace-index").value=sampleIndex;
  $("trace-position").textContent=(sampleIndex+1)+" / "+trace.length+" · "+(gesture?(timeAxis==="local_t"?"本笔 t=":"t=")+fmt(sample[timeAxis])+" s":"u="+fmt(sample.u));$("sample-state").replaceChildren();
  const labels=gesture?{t:painting?"该笔刷累计时间（s）":"时间 t（s）",...(painting?{local_t:"本笔时间（s）"}:{}),event_index:"动作序号（从 0 起）",event_phase:"动作阶段",phase:"动作进度 0–1",...(painting?{nominal_pressure_target:"名义载荷目标",observed_density:"已见密度（模型值）",observation_x:"密度观测位置 x",observation_y:"密度观测位置 y",feedback_multiplier:"画面反馈倍率"}:{}),pressure_target:painting?"画面反馈后目标":"驱动载荷目标",pressure:"实际接触压力",internal_force:"内部载荷状态",contact_gate:"接触门控",commanded_contact:"指令接触状态",contact:"正载荷状态",angle_target:"目标方位（rad）",azimuth:"实际方位（rad）",tilt:"实际倾角（rad）",ink_load:"含墨 ink_load",water_load:"含水 water_load",flow_state:"供墨响应",hand_x:"手部 x",hand_y:"手部 y",tip_x:"笔尖 x",tip_y:"笔尖 y"}:{u:"归一化笔程 u",t:"时间 t（s）",pressure:"压力 pressure",speed:"速度（px/s）",tilt:"倾角（rad）",azimuth:"方位（rad）",contact:"接触状态",ink_load:"含墨 ink_load",water_load:"含水 water_load",tip_x:"笔尖 x",tip_y:"笔尖 y"};
  if(gesture){const event=driverEvents(s)[Number(sample.event_index)],group=node("div");group.appendChild(node("dt","","当前动作"));group.appendChild(node("dd","",eventName(event)));$("sample-state").appendChild(group)}
  Object.entries(labels).forEach(([key,label])=>{const group=node("div");group.appendChild(node("dt","",label));const state=contactValue(sample[key]),value=key==="commanded_contact"?(state===true?"指令触纸":state===false?"指令离纸":"未记录"):key==="contact"?(state===true?(gesture?"有正载荷":"触纸"):state===false?(gesture?"无正载荷":"离纸"):"未记录"):fmt(sample[key]);group.appendChild(node("dd","",value));$("sample-state").appendChild(group)});
  if(painting){
    traceCurve($("trace-pressure"),"pressure","提按 · 名义、画面反馈与毛束响应",null,timeAxis,{series:[{key:"nominal_pressure_target",name:"名义目标",color:"#8b8980",dash:[2,3]},{key:"pressure_target",name:"反馈后目标",color:"#795c42",dash:[6,4]},{key:"pressure",name:"实际压力",color:"#a74836",dash:[]}]});
    traceCurve($("trace-speed"),"observed_density","已见密度与画面反馈 · 左右轴分别读数",null,timeAxis,{separateScales:true,series:[{key:"observed_density",name:"已见密度（左）",color:"#3e5d56",dash:[]},{key:"feedback_multiplier",name:"载荷倍率（右）",color:"#795c42",dash:[6,4]}]});
  }else if(gesture){traceCurve($("trace-pressure"),"pressure","接触压力 · 目标与响应","pressure_target","t");traceCurve($("trace-speed"),"azimuth","毛束方位（rad）· 目标与响应","angle_target","t")}
  else{traceCurve($("trace-pressure"),"pressure","压力 pressure · trace 实值");traceCurve($("trace-speed"),"speed","速度 speed（px/s）· trace 实值")}
  $("trace-pressure").setAttribute("aria-label",painting?"名义载荷、画面反馈后的目标载荷与毛束实际压力随时间对照":gesture?"目标载荷与实际接触压力随时间的对照":"实际采样压力随归一化笔程的变化");$("trace-speed").setAttribute("aria-label",painting?"左轴已见密度与右轴载荷反馈倍率，两者分别读数":gesture?"目标方位与毛束实际方位随时间的对照":"实际采样速度随归一化笔程的变化");renderDetail();renderPaths();
}
function updateSelection(resetSample=true){
  selected=Math.max(0,Math.min(filtered.length-1,selected));$("stroke-index").value=filtered.length?selected+1:0;$("stroke-index").max=filtered.length;$("stroke-index").disabled=!filtered.length;$("prev-stroke").disabled=selected<=0;$("next-stroke").disabled=selected>=filtered.length-1;$("stroke-fields").replaceChildren();$("stroke-purpose").replaceChildren();$("stroke-purpose").hidden=true;
  const s=filtered[selected],gesture=hasDriver(s),painting=s?.raw?.kind==="painting_stroke";setSurface(s);trace=s?.trace||[];if(resetSample)sampleIndex=0;$("dynamic-panel").hidden=!trace.length;
  if(s){
    if(painting){const context=s.raw.context||{};for(const[key,label]of[["role","这一笔的作用"],["face","所在体面"],["start_reason","为何起在这里"],["turn_reason","转折如何成形"],["end_reason","为何收在这里"]]){if(context[key]!==undefined&&context[key]!==null){$("stroke-purpose").appendChild(node("dt","",label));$("stroke-purpose").appendChild(node("dd","",simpleText(context[key])))}}$("stroke-purpose").hidden=!$("stroke-purpose").children.length}
    addField("draw order",s.index+1);addField("kind",s.raw.kind||"stroke");addField("tag",s.tag);addField("轨迹首点",s.raw.start||s.points[0]);if(painting){const contacted=trace.filter(row=>contactValue(row.commanded_contact)===true);if(contacted.length){addField("首次触纸采样",[contacted[0].tip_x,contacted[0].tip_y]);addField("末次触纸采样",[contacted[contacted.length-1].tip_x,contacted[contacted.length-1].tip_y])}}addField("points",s.raw.display_trace?s.points.length+" / "+s.raw.display_trace.source_samples+" 个采样点（显示 / 完整）":s.points.length+(trace.length?" 个采样点":" 个控制点"));addField("画布",width+" × "+height);
    if(trace.length){addField("基准笔宽",s.raw.brush?.width??s.raw.spec?.width);if(!gesture){addField("起笔参数",s.raw.spec?.boundary?.enter);addField("收笔参数",s.raw.spec?.boundary?.leave)}if(gesture){addField("指令接触段",s.raw.diagnostics?.commanded_contact_episodes);addField("正载荷区间数",s.raw.diagnostics?.positive_load_episodes)}else addField("接触段数",s.raw.diagnostics?.contact_episodes);addField("总时长(s)",s.raw.diagnostics?.time);if(!gesture)addField("弧长(px)",s.raw.diagnostics?.arc_length)}
    if(gesture){addField("动作数",driverEvents(s).length);addField("驱动 hash",s.raw.driver?.driver_hash||s.raw.driver_hash);if(painting)addField("画面反馈",s.raw.feedback);addField("机械参数",s.raw.brush);addField("供墨参数",s.raw.ink);addField("纸面参数",s.raw.paper)}
    if(!trace.length)for(const key of["pressure","width","water","ink","dry","tilt","fade_bottom"]){let value=s.raw[key];if(key==="pressure"&&value===null)value="默认提按包络（见 physics.md）";if(value===undefined&&Array.isArray(s.raw.points)&&s.raw.points[0]&&typeof s.raw.points[0]==="object"&&!Array.isArray(s.raw.points[0])){const values=s.raw.points.map(p=>p[key]).filter(v=>v!==undefined);if(values.length)value=values}addField(key,value)}
  }else addField("筛选结果","无匹配笔触");
  if(trace.length){$("trace-index").max=trace.length-1;$("trace-index").disabled=trace.length<2;
    $("trace-title").textContent=painting?"这一笔如何随已有形墨继续":gesture?"离散动作、驱动目标与实际响应":"一笔内部的连续变化";
    $("trace-description").textContent=painting?"名义目标经当前局部密度减载，再由毛束接触产生实际压力；形体作用由笔序方案给定。已见密度为模型值，载荷倍率无量纲，按第二图左右轴分别读数。曲线为抽样显示，精确重算使用包内完整账本；滑块不重新作画。":gesture?"虚线来自动作编译后的驱动目标，实线来自毛束的有状态响应。动作切换不重置笔内状态；接触门控为 0 时不沉积。曲线与轨迹均读取保存的记录，滑块不重新求解。":"曲线直接读取 trace 采样值，函数定义直接读取 spec。滑块只查看记录，不把函数片段拆成新的起笔或收笔。停驻时 u 可不变，而 t 与笔内状态继续演化。";
    $("spec-summary").textContent=painting?"查看成形意图、动作、画面反馈与承接状态":gesture?"查看动作程序、响应参数与驱动事件":"查看这一笔的函数定义与边界参数（原始 spec）";
    $("stroke-spec").textContent=gesture?JSON.stringify({...(painting?{context:s.raw.context,feedback:s.raw.feedback,initial_state:s.raw.initial_state,final_state:s.raw.final_state}:{}),program:s.raw.program,brush:s.raw.brush,ink:s.raw.ink,paper:s.raw.paper,driver_hash:s.raw.driver?.driver_hash||s.raw.driver_hash,events:driverEvents(s)},null,2):s.raw.spec?JSON.stringify(s.raw.spec,null,2):"本条记录未保存 spec。";
    $("driver-events").hidden=!gesture;
    if(gesture)$("driver-events").textContent=(painting?"本笔动作时间表（s，从本笔起点计）：":"动作时间表（s）：")+driverEvents(s).map((event,index)=>index+" "+eventName(event)+" ["+fmt(event.t_start??event.t0)+", "+fmt(event.t_end??event.t1)+"]").join("；")+"。";
    traceEvents();renderSample()
  }else{renderDetail();renderPaths()}
}
function stopPlayback(){playing=false;$("play").textContent="播放笔序";lastTime=0}
function applyFilter(){
  stopPlayback();const prefix=$("tag-filter").value.trim().toLowerCase();filtered=strokes.filter(s=>s.tag.toLowerCase().startsWith(prefix));selected=0;shown=0;$("progress").max=filtered.length||1;$("progress").disabled=!filtered.length;$("play").disabled=!filtered.length;$("filtered-count").textContent=filtered.length+" / "+strokes.length+" 笔";updateSelection();
}
function selectLedger(){
  stopPlayback();ledger=ledgers.find(item=>item.id===$("ledger-select").value)||ledgers[0];strokes=normalizeStrokes(ledger?.strokes);$("tags").replaceChildren();$("ledger-note").hidden=!ledger?.metadata?.display_trace_decimated;$("ledger-note").textContent=ledger?.metadata?.display_trace_decimated?"曲线与路径为抽样显示：每笔通常至多 "+fmt(ledger.metadata.display_trace_target_samples)+" 点，保留首尾及动作、接触切换边界。精确重算使用代码包 examples/"+(ledger.metadata.full_trace_file||"painting_strokes.json")+" 中的完整账本。":"";
  [...new Set(strokes.flatMap(s=>[s.tag,s.tag.split(/[\/:_.-]/)[0]]))].sort().forEach(tag=>{const option=node("option");option.value=tag;$("tags").appendChild(option)});$("tag-filter").value="";backgroundFile=null;background=null;applyFilter();
}
ledgers.forEach(item=>{const option=node("option","",item.label);option.value=item.id;$("ledger-select").appendChild(option)});
$("ledger-select").addEventListener("change",selectLedger);$("tag-filter").addEventListener("input",applyFilter);$("show-result").addEventListener("change",renderPaths);$("stroke-index").addEventListener("change",()=>{selected=(Number($("stroke-index").value)||1)-1;updateSelection()});$("prev-stroke").onclick=()=>{selected--;updateSelection()};$("next-stroke").onclick=()=>{selected++;updateSelection()};$("progress").addEventListener("input",()=>{stopPlayback();shown=Number($("progress").value);renderPaths()});$("trace-index").addEventListener("input",()=>{sampleIndex=Number($("trace-index").value);renderSample()});
function frame(time){if(!playing)return;if(lastTime){fraction+=(time-lastTime)/1000*90;const step=Math.floor(fraction);if(step){shown=Math.min(filtered.length,shown+step);fraction-=step;renderPaths()}}lastTime=time;if(shown>=filtered.length){stopPlayback();return}requestAnimationFrame(frame)}
$("play").onclick=()=>{if(playing){stopPlayback();return}if(shown>=filtered.length)shown=0;playing=true;fraction=0;lastTime=0;$("play").textContent="暂停";requestAnimationFrame(frame)};
if(ledgers.length){$("ledger-select").value=(ledgers.find(item=>item.id==="painting")||ledgers.find(item=>item.id==="drive")||ledgers.find(item=>item.id==="dynamics")||ledgers[ledgers.length-1]).id;selectLedger()}else $("strokes-section").hidden=true;
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(strokes.length)updateSelection(false)},100)});
let sources=DATA.sources;if(sources&&!Array.isArray(sources)){sources=sources.sources||sources.references||Object.entries(sources).map(([id,s])=>typeof s==="object"?{id,...s}:{title:id,note:s})}if(!Array.isArray(sources))sources=[];
function sourceLink(title,url){if(typeof url!=="string"||!/^https?:\/\//i.test(url))return node("span","",title);const link=node("a","",title);link.href=url;link.target="_blank";link.rel="noopener noreferrer";return link;}
sources.forEach(source=>{const s=typeof source==="string"?{title:source}:source;if(!s||typeof s!=="object")return;const li=node("li");const title=s.title||s.name||s.work||s.id||"参考资料",url=s.url||s.link||s.source_url;if(s.source_group)li.appendChild(node("span","source-type",s.source_group));li.appendChild(sourceLink(title,url));const meta=[s.author||s.artist||s.creator,s.institution||s.publisher,s.date||s.year].filter(Boolean).map(simpleText).join(" · ");if(meta)li.appendChild(node("span","source-type",meta));const note=s.relevance||s.note||s.notes||s.use||s.description;if(note)li.appendChild(node("p","",simpleText(note)));const links=node("p");for(const [key,label]of[["direct_image_url","原始图像"],["archive_record_url","图像档案"],["license_url","授权说明"]]){if(s[key]){if(links.children.length)links.appendChild(document.createTextNode(" · "));links.appendChild(sourceLink(label,s[key]))}}if(links.children.length)li.appendChild(links);if(s.attribution)li.appendChild(node("p","source-type",s.attribution));$("sources").appendChild(li)});if(!$("sources").children.length)$("sources-empty").hidden=false;
[...document.querySelectorAll("main > section")].filter(section=>!section.hidden).forEach((section,index)=>{const number=section.querySelector(".section-index");if(number)number.textContent=String(index+1).padStart(2,"0")});
</script>
</body>
</html>
'''


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples_dir", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(build_gallery(arguments.examples_dir, arguments.output))
