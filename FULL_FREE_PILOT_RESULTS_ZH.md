# Full-Free Dynamic Cell Graph：1000-step Pilot 结果

## 结论

单 seed、1000-step pilot 达到 Level 1（结构成功），但尚未达到 Level 2，也不能宣称 Level 3。

- 节点数、fan-in 和真实最长路径都随输入变化。
- Learned graph 明显优于 Fixed、Global Shuffle、Same-token Shuffle 和 Same-token+Position Shuffle。
- Natural 在纯 LM loss 下从全开自然变为部分稀疏，且没有 dead Cell。
- 动态模型在一个匹配 23 个活跃 Cell 的独立训练静态基线上略胜，但只有一个 seed、一个静态拓扑，不能称为超过 best-static Pareto。
- 计算量没有按 reference NLL 难度单调增加；当前没有证据说明模型会把更多计算给更困难 token。

## Natural 图演化

| Step | Active Cells | Active edges | Fan-in | Longest depth | Val loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 32.00 | 320.00 | 10.00 | 8.00 | 4.2293 |
| 500 | 28.72 | 122.03 | 4.25 | 6.28 | 2.2155 |
| 1000 | 22.73 | 58.44 | 2.57 | 5.34 | 2.0622 |

训练目标中 node/edge/depth/balance 权重全部为 0。最终理论 Cell MACs 从最大 524,288/token 降至约 372,472/token，即逻辑上跳过约 29%。这不是实际 GPU 节省，因为 Research Mode 仍稠密执行 Cell。

最终 validation 样本中：

- active Cells 均值 22.73，标准差 3.56，范围 12–32；
- depth 均值 5.34，标准差 1.39，范围 2–8；
- 没有完整空 Step，因此浅度来自激活边形成的较短路径，而不是后半段连续空层；
- dead-cell ratio 为 0；
- 形成 9 个 usage ≥80% 的 shared-trunk Cell，没有 ≤5% 的 rare Cell。

逐 Step 平均宽度：

```text
3.59, 3.63, 2.94, 2.97, 2.75, 2.43, 2.15, 2.26
```

## 因果干预

| 条件 | PPL |
|---|---:|
| Learned | **7.7554** |
| Fixed-most-common | 22.6191 |
| Global request shuffle | 22.3127 |
| Same-token shuffle | 9.9293 |
| Same-token + position shuffle | 9.8166 |
| 独立训练 matched static | 7.8081 |

Same-token 两种 shuffle 仍显著恶化，说明图与上下文存在对应关系，不能只用字符 identity 或粗位置解释。

共观察到 12,328 张完整图，graph entropy 为 8.816 nats；Top-1/4/8/16 coverage 分别为 3.67%/7.04%/9.29%/11.39%。没有出现可供 Address Predictor 使用的 Top-K condensation。

## Difficulty 与边际价值

使用独立训练静态模型的 token NLL 作为固定 difficulty proxy。最容易到最困难五组的平均 active Cell 为：

```text
23.53, 22.69, 22.51, 22.49, 22.43
```

平均边数为：

```text
71.33, 63.23, 57.00, 49.09, 51.28
```

趋势与“困难 token 获得更多计算”的理想现象相反。当前 Router 学到的是上下文相关信息流，但尚未学到可解释为难度驱动的资源分配。

Remove/Add 反事实和逐 Cell specialization 已导出。部分 Cell 删除显著增加 NLL，但盲目加入未激活 Cell 多数会降低质量，说明节点开关不是简单的“越多越好”；这一结果仍需更多 batch 和 seed 验证。

## Router 成本

- Router 参数：28,072，约占 1.09M 模型的 2.6%。
- 理论 Router MACs：212,224/token。
- 将 edge scorer 从逐 Cell 调度改为逐 Step 批量评分后，GTX 1650 上 Router-only CUDA latency 约 10.99 ms，完整模型约 147.63 ms，占 7.44%。

该比例已低于最初的约 20%，但仍不是零；Physical Mode 前不能宣称真实加速。

## 当前研究等级

- Level 1：通过（单 seed pilot）。
- Level 2：未通过；计算量与难度/边际收益关系尚不理想。
- Level 3：未判定；需要 3000+ steps、3 seeds、预算扫描以及多个 matched static 拓扑形成真正 Pareto。

