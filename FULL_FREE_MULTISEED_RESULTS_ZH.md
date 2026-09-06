# Full-Free v1 冻结后复现实验（3 seeds）

日期：2026-09-04  
冻结版本：`FULL_FREE_V1_FREEZE.json`

## 结论

在完全相同的 1000-step Shakespeare-char 协议下，Dynamic 在 3/3 个 seed 中都优于从各自动态图提取、再从头独立训练的 matched Static。三组平均 PPL 为 7.7396 vs 7.9205，平均配对差值为 -0.1809（相对改善约 2.28%）。

这已经排除了“只在 seed 1337 偶然胜出”的最直接解释，但还不是普适性证明：样本只有 3 个 seed，配对差值的 95% t 区间很宽且跨过 0；Static 也只是一个预先规定的提取基线，不等于对所有固定图做了全局结构搜索。

## Dynamic vs matched Static

| seed | Dynamic PPL | Static PPL | Dynamic - Static | Dynamic 平均 Cells | Static Cells |
|---:|---:|---:|---:|---:|---:|
| 1337 | 7.7554 | 7.8081 | -0.0528 | 22.73 | 23 |
| 2027 | 7.8646 | 8.2302 | -0.3656 | 22.39 | 22 |
| 3407 | 7.5988 | 7.7231 | -0.1242 | 24.46 | 24 |
| mean | 7.7396 | 7.9205 | -0.1809 | 23.19 | 23.00 |

Static 的节点数取该 seed 动态模型的平均 active cells 四舍五入；节点按使用率选择，边取对应节点最常见的输入组合。Static 与 Dynamic 使用同一 Cell/Attention 参数包络，但 Static 无 Router 推理开销。这里的“matched compute”指逻辑 Cell 计算量近似匹配，不包含动态图的 Router 开销，也不代表当前 Research Mode 已物理跳过未激活 Cell。

## 输入相关性干预

把一个 token 的图替换成同 token、其他上下文位置学到的图，PPL 均明显变差：

| seed | learned | same-token shuffle | same-token + position shuffle |
|---:|---:|---:|---:|
| 1337 | 7.7554 | 9.9293 | 9.8166 |
| 2027 | 7.8646 | 10.2148 | 10.1093 |
| 3407 | 7.5988 | 9.4354 | 9.3961 |

这说明有效路由不只是字符 ID 的静态查表；它依赖当前上下文形成的状态。不过该干预同时改变整张节点/边图，不能单独归因于宽度、深度或某一条边。

## Marginal Compute Utility

评估对每个目标 token 单独做反事实：上下文和其他 token 的图保持不变，只在目标位置按 Router 权重从低到高移除 Cell。每个 seed 抽样 512 个 token。

| seed | full NLL | keep 80% | keep 60% | keep 40% | corr(active, last-20% utility) | corr(active, last-4 utility) |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 2.2102 | 2.2818 | 2.5961 | 3.2983 | +0.0119 | +0.0115 |
| 2027 | 2.0100 | 2.0391 | 2.2660 | 2.8921 | -0.0088 | -0.0900 |
| 3407 | 2.0542 | 2.0595 | 2.2481 | 2.9953 | -0.0145 | -0.0442 |

表中相关性为 Pearson；Spearman 也都接近 0。由此得到两个不同结论：

1. 多算 Cell 整体有因果价值：逐步裁剪会稳定提高 NLL。
2. 当前 Router 没有证据表明会把更多 Cell 分给“边际计算收益更高”的 token。Level 2 仍未成立。

“从最低权重开始移除”是对最后分配 Cell 的可复现近似，不等于搜索每个 token 的最优子图。因此这是保守、明确的第一版边际效用检验。

## 可复现产物

- `full_free_multiseed.csv`：三 seed 配对汇总。
- `out-full-free-natural-seed*/full_free_summary.json`：结构统计。
- `out-full-free-natural-seed*/full_free_interventions.csv`：干预 PPL。
- `out-full-free-natural-seed*/marginal_compute_utility.csv`：逐 token 反事实曲线。
- `out-full-free-natural-seed*/marginal_compute_utility.json`：相关性摘要。
- `experiments/full_free_marginal_compute_utility.py`：单位置因果评估脚本。

冻结清单中的四个 SHA-256 已复核一致；全量 26 个单元测试通过。

## 下一实验

保持架构不变，运行全局平均 Cell 预算 `B = 8, 12, 16, 20, 23`。每个预算、每个 seed 都需要独立训练 Dynamic，并从该 Dynamic 提取等 Cell 数固定图后独立训练 Static。主图使用实测 `Average Active Cells` 为横轴而不是目标 B，以防 dual constraint 未精确命中目标。

首个协议校准端点已经完成：`B=8, seed=1337` 的 Dynamic 实测 7.64 Cells（token 范围 5–15），PPL 8.6393；matched Static 固定 8 Cells，PPL 7.9562。该低预算点是 Static 胜出 0.6831，说明现阶段的 Dynamic 并未形成全区间占优的 Pareto 曲线。必须继续测 12/16/20/23 才能定位交叉点，并区分“动态图只在高预算有效”和“低预算 dual 训练方法需要改进”。
