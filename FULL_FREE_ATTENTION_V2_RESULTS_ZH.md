# Full-Free Attention v2：首个 Pilot 结果

日期：2026-09-06

## 目标

在不修改 Full-Free v1 冻结实现的前提下，让同一张 Cell 图同时控制：

- 输入边；
- Cell 宽度；
- Cell 深度；
- Attention 深度。

没有新增独立的 Attention Router。一个 Step 是否包含活跃 Compute Cell，直接决定该 Step 的 Attention 是否激活。为保证最低信息交换能力，8个 Step 中保留模型配置记录的3个均匀锚点 `[0, 4, 7]`。

联合预算以 Cell-equivalent MAC 为单位。在 `C=128, T=128, atom=64` 时，一个标准 Attention 约等价6个 Compute Cell，训练目标为32个等价单位。

## 训练结果

配置：Shakespeare char、seed 1337、1000 steps、32 Cell 最大容量、8 Attention 最大深度。

| 指标 | 结果 |
|---|---:|
| 动态 PPL（10-batch eval） | 8.595 |
| 动态 PPL（batch=128, 3-batch eval） | 8.621 |
| 平均 active Cells | 5.37 / 32 |
| 平均 active Attention | 3.008 / 8 |
| 平均 Cell 图深度（step 1000 eval） | 1.99 |
| 学到的高频 Attention 主干 | Step 0 / 4 / 7 |

可选 Attention 的逐 token 使用率全部低于0.25%。因此动态图已经形成非常清楚的三层公共主干，外加极少量长尾 Attention。

## 动态稀疏与物理执行的差异

虽然每个 token 平均只用3.008层 Attention，但 batch=128 时，极少量长尾 token 会覆盖所有可选层，导致朴素动态图仍要启动8/8层 Attention。只做“整批为空才跳层”没有加速。

按1%使用率阈值从模型路由中自动提炼 Familiar Fast Path 后，得到共享计划 `[0, 4, 7]`：

| 指标 | Full dynamic | Condensed Fast Path |
|---|---:|---:|
| PPL | 8.6210 | 8.6331 |
| NLL 变化 | — | +0.00140 |
| Attention 层数 | 最多8，token平均3.008 | 固定共享3层 |

这条 Fast Path 不是人工指定的 D3，而是先训练 Full-Free 图，再根据实际路由覆盖率凝聚得到。

## GTX 1650 FP16 实际速度

条件：batch=128、block=128、关闭只用于研究记录的图熵/`unique`/`.item()`诊断；输出一致性已验证。

AB/BA 交错5组中位数：

| 执行方式 | 延迟 |
|---|---:|
| Dense masked（8 Attention + 32 Cell 都计算） | 419.37 ms |
| 3-Attention Fast Path + queued active Cells | 346.83 ms |

- 实际 speedup：**1.209×**；
- 延迟下降：**17.30%**；
- Dense Cell 与 queued Cell logits 最大绝对误差：**0**；
- 动态 Attention 的物理跳层与 dense-mask logits 最大绝对误差：**0**。

## 当前结论与限制

已经证明：Attention 可以被纳入同一张自由计算图；图能够学出约3层 Attention 的公共主干；将主干凝聚为共享执行计划后，可以在普通GPU上产生可重复的真实加速。

尚未证明：v2 在质量上优于 v1 或 matched static。v2 PPL 约8.6，仍差于 Natural v1 约7.76，也差于 matched Static B=8 约7.96。本 pilot 的联合预算明显偏激进。

参数文件大小目前也没有下降：未激活的权重仍作为长尾回退保留。若导出仅含 `[0,4,7]` Attention 与高频 Cell 的部署模型，才会得到体积下降，但需要另做剪枝后质量验证。

下一步应跑联合预算 Pareto，而不是继续压到最稀疏：例如有效成本目标 `40 / 48 / 56 / 64`，寻找接近 v1 PPL 且仍能凝聚为3–5层共享 Fast Path 的甜点区。
