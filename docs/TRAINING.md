# 运行与训练

## 单笔调用

`sandboxes/brush-toolbox/learning/` 保留原模块结构。给 Python 的模块搜索路径加入此目录，物理核心调用 `wet_dry.load_engine`，用 `ink_adversary.AdversarialInkObservation` 可以叠加当前多尺度外观。

```python
from pathlib import Path
from brush_api import render_stroke
from ink_adversary import AdversarialInkObservation
from wet_dry import load_engine

model_dir = Path("sandboxes/brush-toolbox/learning/results")
physics = load_engine(model_dir / "wet-dry/calibration.json", track_contact_water=True)
engine = AdversarialInkObservation(
    physics, model_dir / "ink-adversary/protected-without-xingjie.pt",
    seed=19, local_budget=True,
)
result = render_stroke(request, engine=engine)
```

完整的 `StrokeRequest`、连画三笔和账量例子在 [smoke.py](../sandboxes/brush-toolbox/learning/smoke.py)。权重以 `torch.load(..., weights_only=True)` 读取。当前物理校准和学习权重用于研究复现，实物质量和照片的光学量仍需配对标定。

## 研究入口

| 内容 | 入口 |
| --- | --- |
| 笔的物理状态 | `brush_api.py`、`brush_intent.py`、`wet_dry.py`、`wet_paper.py` |
| 训练接纸和连续供墨 | `learned_contact_eval.py`、`learned_release_eval.py`、`contact_transition_eval.py` |
| 多尺度基线与留出 | `ink_adversary_eval.py`、`ink_adversary_audit.py` |
| 细纹表示与递推 | `ink_latent_eval.py`、`ink_trajectory_eval.py`、`ink_trajectory_audit.py` |
| 最新实写局部模型 | `ink_neighborhood_eval.py`、`ink_neighborhood_join.py`、`ink_neighborhood_refine.py`，用 `ink_neighborhood_metrics.py --assembled` 核对组合 |
| 空间原型和旧方法 | `route-r02/`、`brush-lab/`、`sandboxes/blender-ink-bloom/` |

实写训练脚本在 `learning` 目录运行，使用本地 `.build/` 和各脚本指定的资料位置。教学参考包括 [清华教材](https://www.tup.tsinghua.edu.cn/upload/books/yz/071995-01.pdf)、[馆方国画示范](https://education.asianart.org/resources/an-introduction-to-chinese-brushpainting-techniques/)、[Gathered 侧锋课](https://www.gathered.how/arts-crafts/art/beginners-guide-to-chinese-brush-painting)、[艺学绘示范](https://www.youtube.com/watch?v=AtMX5-s1XTE) 和 [兴杰运笔实拍](https://www.youtube.com/watch?v=wVZN4Prx47g&t=187s)。图号、裁窗及本地路径可查 `prepare.py`、`paired_pose_eval.py`、`axis_ink_eval.py` 和同目录的参考标注 JSON。部分历史跑法还用到未随库附带的本地配准和中间结果。

最新局部匹配按整课留出。从本地参考图重建训练样本库后，可运行上述四个入口。保留兴杰、艺学绘、Gathered 三折的接缝和有界校正成品，以及四折初始匹配权重。匹配权重包含网络参数和样本名，运行该路线需本地原图。仓库提供的三笔复跑可以独立执行。

Blender 原型的背景裁窗及原文件写在 [sources.json](../sandboxes/blender-ink-bloom/assets/sources.json)。可从 R02 项目景片按记录裁切出背景，运行 `bake_ink.py` 或 `bake_dry.py` 再生成数据帧，由 `build_blender.py` 读取。

## 供评审的后续入口

从仓库根目录执行下面的命令，可用已存的结构化输入跑六笔。输出包含逐笔纸、笔及账量，写入本地 `.build/`。

```sh
python -B sandboxes/text-to-brush/replay.py sandboxes/text-to-brush/input-B.json --engine learned --output sandboxes/text-to-brush/.build/review
```

下一阶段的[配对采样雏形](../sandboxes/brush-toolbox/learning/paired_capture.py)接收以下 JSON 字段，输出纸面像素尺寸、接纸区间、沿程速度和有符号的锋尖投影。动作帧先在视频和成迹上人工配准。该入口处理标注和缺测，接纸与墨量预测仍在下一阶段实现。

| 字段 | 含义 |
| --- | --- |
| `stroke_id`、`batch_id`、`finished_scan` | 单笔及整批留出标识，成迹图的本地相对路径或索引 |
| `paper_mm`、`scan_shape_px` | 纸宽高和成迹栅格行列数 |
| `hand`、`load_unit`、`contact_width_mm` | 手别、法向力单位和已知接纸宽度 |
| `frames` | 按时间排列的 `Frame` 字段对象，`time_s`、`xy_mm`、`touching` 必填 |
| `brush_load` | `water_mg` 和 `solids_mg` 按笔尖、左笔腹、右笔腹排列，可用 `null` 标缺测 |
