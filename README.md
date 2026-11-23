# 水墨江山

项目用中国画的一笔一笔生成墨迹，再把笔迹放入有景深的山水空间。当前研究集中在可独立调用的单笔模型。已有同一套干湿纸墨状态、逐笔库存和训练得到的原像素外观。侧锋的下侧浓边和转顿仍需与实写匹配。

- [项目范围、接口与下阶段](docs/SCOPE.md)
- [学习笔记与当前结论](docs/LEARNING.md)
- [本地运行、训练与资料来源](docs/TRAINING.md)
- [当前证据与轮换规则](docs/EVIDENCE.md)

### 复跑三笔

用 Python 3.12。在仓库根目录运行以下命令，可用同一支笔连续画三笔并检查余墨、纸面状态和总光学深度。校准参数与多尺度外观权重随仓库提供。

```sh
python -m pip install -r sandboxes/brush-toolbox/learning/requirements.txt
python -B sandboxes/brush-toolbox/learning/smoke.py
```

[入山 R02](route-r02/README.md) 是固定游线 WebGL 演示，使用项目自带的景片资产。[笔触实验室 v0.21](brush-lab/README-v021.md) 保留旧的材料模拟和笔法程序。[景片接口](route-r02/BRUSH_ADAPTER_CONTRACT.md) 记录场景接线的墨层、内部覆盖和画幅约定。
