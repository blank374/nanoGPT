# Dynamic Cell Graph：Fixed Compute + Free Connectivity 正式结果

## 结论先行

在固定宽度、固定深度、固定 Attention、固定每步 2 个活跃 Cell 的条件下，学习到的边不是一张对所有输入都相同的静态图。

- 三个随机种子中，原始动态图均优于“每个 Cell 固定为最常见输入 mask”的同权重干预。
- 三个随机种子中，原始动态图均优于“跨 request 打乱完整 route”的同权重干预。
- 后续 Step 的 route entropy 明显大于零，且最常见 route 的覆盖率很低。
- 不同 Cell 的平均 fan-in、mask 分布和 source usage 不同。

因此，第一阶段的科学问题可以回答为：**模型确实在按输入重组信息流，而且已训练的计算权重会使用这种对应关系。**

但当前结果不能回答“动态图已经比最好的固定网络更强”。独立训练的固定图平均 PPL 略好于动态图。因此更准确的表述是：**输入条件化 routing 现象成立，任务收益尚未成立。**

## 实验设置

- 数据：nanoGPT Shakespeare char。
- 三个 seed：1337、2027、3407。
- 每组训练 1000 steps。
- 容量：4 Steps × 4 Cells。
- 每步严格固定激活前 2 个 Compute Cell，共 8 个活跃 Cell。
- Cell：128 → 64 → 128。
- 每步固定执行标准 causal attention。
- 深度、宽度、节点数、Attention 路径均固定；只有合法历史输入边可变。
- Balance loss、node cost 均关闭。
- 正式分析使用相同的 16 个 validation batches。

## 三随机种子结果

| Seed | Learned | Fixed-most-common | Shuffled routes | 独立训练 Fixed | Edge-budget |
|---:|---:|---:|---:|---:|---:|
| 1337 | 7.5400 | 8.7760 | 9.4636 | 7.5480 | 7.4973 |
| 2027 | 7.8367 | 9.6412 | 9.7740 | 7.7298 | 7.8054 |
| 3407 | 7.5662 | 9.3459 | 9.5118 | 7.4036 | 7.5581 |
| 平均 | **7.6476** | 9.2544 | 9.5831 | **7.5605** | 7.6203 |
| 样本标准差 | 0.1643 | 0.4398 | 0.1671 | 0.1635 | 0.1632 |

同权重因果干预的平均差异：

- 固定为最常见图：PPL 增加 1.6068，约增加 21.0%。
- 跨 request 打乱 route：PPL 增加 1.9355，约增加 25.3%。
- 两项判断均为 3/3 seeds 同方向。

这里 D/E 使用 Learned checkpoint 的同一组模型权重，只替换边 mask。它们说明性能不只依赖边的总体频率或 fan-in，还依赖 route 与当前输入的对应关系。

独立训练 Fixed 的平均 PPL 为 7.5605，比 Learned 的 7.6476 低 0.0872，约 1.15%。三个种子中 Learned 仅在 seed 1337 略优。因此不能把同权重干预的优势解释成动态图已经在端到端语言建模上胜过固定模型。

## 图是否真的多样

全图统计的三 seed 均值：

- graph entropy：4.5525 nats；
- Top-1 graph coverage：8.69%；
- Top-4 graph coverage：23.36%；
- Top-8 graph coverage：35.49%；
- token identity 与 graph mask 的估计互信息：3.2904 nats。

逐 Step 均值：

| Step | Graph entropy | Top-1 coverage |
|---:|---:|---:|
| 0 | 0.00 | 100% |
| 1 | 1.91 | 34% |
| 2 | 3.15 | 15% |
| 3 | 3.69 | 12% |

Step 0 只有 current backbone state 是合法来源，所以必然是静态的；随着可用历史来源增加，后续步骤的图多样性持续增大。这与实现中的 DAG 约束一致，不是异常退化。

逐 Cell 的三 seed 均值也呈现相同趋势：Step 1 两个 Cell 的 mask entropy 约 1.10/1.15，Step 2 为 2.26/2.30，Step 3 为 3.02/2.75；对应 Top-1 mask coverage 从约 52%/56% 降至约 17%/17%。平均 fan-in 也从 Step 1 的 2.39/2.16 增至 Step 3 的 3.22/2.94。

## 科学解释与边界

当前证据支持：

1. 路由分布不等价于一张全局静态子网；
2. 同一 Cell 对不同 token 使用不同历史信息；
3. route 与输入之间的配对对已训练模型的预测有因果作用；
4. 这一现象在三个初始化下重复出现。

当前证据不支持：

1. 动态边已经改善最终 validation PPL；
2. 这些边能带来实际 wall-clock 加速；
3. 自由宽度或自由深度已经有效；
4. token identity 互信息等价于高层语义 specialization。

样本数只有 3 个 seed，且只训练 1000 steps。结果足以作为进入下一阶段的工程门槛，但不适合做显著性或规模律结论。

## 下一阶段建议

可以进入“自由宽度、仍固定深度”的阶段，但建议保留本实验全部干预作为回归门槛。下一阶段除了 PPL，应同时报告 active-cell 数、实际 MACs/理论 FLOPs、宽度分布、按 token/位置/损失难度分桶的路由统计，并继续与等算力固定宽度模型比较。GPU kernel 裁剪仍可后置，先验证可变节点是否带来质量—计算量 Pareto 改善。

