# 自由通道动态宽度 MLP 实验记录

本文档总结当前 nanoGPT 分支中关于动态宽度 MLP 的阶段性工作。

## 1. 研究问题

核心问题是：

> MLP 的有效计算宽度，能否不由人工预先指定，而是在训练过程中由模型自己分化出来？

第一阶段已经验证了离散动态宽度 MLP。该阶段使用四个预设 hidden width：

```text
64 / 128 / 256 / 512
```

并结合：

```text
STE routing
temperature annealing
hard-routing-aware loss
compute cost
```

最终得到大致结果：

```text
dense baseline hard PPL ~= 7.17
dynamic hard PPL       ~= 7.19
mean hard width        ~= 205 / 512
```

这说明：

> 离散 hard dynamic width 是可行的。

当前阶段进一步去掉人工规定的宽度档位，让每一个 MLP hidden channel 自己决定是否参与当前 token 的计算。

## 2. 当前 Free-Channel 设计

Transformer 主结构保持不变：

- `n_embd = 128`
- 最大 MLP hidden width = `4 * n_embd = 512`
- 不修改 attention
- 不修改 QKV
- 不修改 residual stream width
- 不修改 embedding dimension
- 不修改 LayerNorm
- 不修改 block depth

第一版只改 MLP intermediate channels。

新增模块：

```python
class FreeChannelMLP(nn.Module):
```

结构为：

```text
[B, T, 128]
  -> c_fc
[B, T, 512]
  -> GELU
  -> channel-level gating
  -> c_proj
[B, T, 128]
```

完整参数矩阵仍然保留：

```python
c_fc:   128 -> 512
c_proj: 512 -> 128
```

对每个 token，router 产生 512 个 channel 的 gate：

```python
gate_logits = gate_network(x)      # [B, T, 512]
gate_prob = sigmoid(gate_logits / temperature)
```

每个 `gate_prob[..., i]` 表示：

> 第 i 个 MLP channel 对当前 token 是否值得参与计算。

注意：

- 不输出一个 width number
- 不再使用 `P(64), P(128), P(256), P(512)`
- 不使用 Top-K
- 不规定每个 token 必须激活多少 channel
- 不设置 prefix nested width
- 不人为指定 fast/slow/common/rare channels

所有 512 个 channel 在结构上尽量保持平等。

## 3. Routing 模式

当前实现支持两种 routing。

### Soft routing

```python
hidden = gelu(c_fc(x))
hidden = hidden * gate_prob
out = c_proj(hidden)
```

该模式主要用于训练稳定性和分析。

### STE hard routing

```python
hard_gate = (gate_prob > threshold).float()
gate = hard_gate + gate_prob - gate_prob.detach()
hidden = hidden * gate
```

STE 模式满足：

- forward 使用真实 0/1 channel mask
- backward 对 gate network 保持可微

每个 token 的实际 active width 定义为：

```python
active_channels = gate.sum(dim=-1)
```

因此不同 token 可以自然形成不同有效宽度，例如：

```text
token A -> 73 active channels
token B -> 146 active channels
token C -> 281 active channels
token D -> 447 active channels
```

## 4. 资源约束

当前主实验使用全局平均预算，而不是逐 token 固定惩罚。

定义：

```python
mean_gate_ratio = gate.mean()
budget_loss = (mean_gate_ratio - free_channel_target_ratio) ** 2
```

训练 loss：

```python
loss = task_ce + free_channel_budget_weight * budget_loss
```

这只约束全局平均 active ratio，不要求每个 token 都接近同一个宽度。

例如：

```text
token A = 50
token B = 100
token C = 280
token D = 500
```

只要整体平均接近预算即可。

当前默认目标：

```python
free_channel_target_ratio = 0.4
```

对应最大 hidden width 512 时：

```text
0.4 * 512 ~= 205 channels
```

另外保留一个简单 L1 cost 作为可选 ablation：

```python
channel_cost = gate_prob.mean()
loss += free_channel_cost_weight * channel_cost
```

当前主实验默认：

```python
free_channel_cost_weight = 0.0
```

也就是说，目前优先测试 global budget constraint，而不是逐 token 或逐 channel 的简单稀疏惩罚。

## 5. Temperature Annealing

沿用上一阶段已经验证有效的退火思路。

默认配置：

```python
free_channel_temperature = 2.0
free_channel_temperature_final = 0.5
free_channel_temperature_anneal_iters = 1000
```

gate 概率计算为：

```python
gate_prob = sigmoid(gate_logits / temperature)
```

目标是：

```text
训练早期 gate 较软
        ↓
训练后期逐渐接近 0 / 1
```

## 6. 监控指标

为了观察 collapse 和 token 间分化，当前实现记录：

- mean active channels
- median active channels
- std active channels
- min active channels
- max active channels
- gate entropy
- fraction gate > 0.5
- fraction gate > 0.9
- fraction gate < 0.1
- active width histogram
- P10/P25/P50/P75/P90 active width
- per-channel usage rate
- channel usage mean/std/min/max

active width histogram 的 bin 为：

```text
0-64
65-128
129-192
193-256
257-320
321-384
385-448
449-512
```

其中最重要的不是平均宽度，而是：

> 不同 token 是否真的形成不同 active channel count。

因此重点看：

```text
P90 - P10
```

如果所有 token 都集中在同一个宽度附近，则说明自主分化不理想。

## 7. 主要文件

实现文件：

- `model.py`
  - `FreeChannelMLP`
  - `GPTConfig` 中的 free-channel 配置项
  - free-channel budget/cost loss
  - free-channel runtime stats
- `train.py`
  - free-channel 训练配置接入
  - temperature annealing
  - eval/training 日志打印
- `config/train_shakespeare_char_free_channel.py`
  - Shakespeare char free-channel 默认训练配置
- `experiments/free_channel_analysis.py`
  - checkpoint 分析脚本
  - 导出 summary、layer、channel usage CSV

## 8. 单组 Free-Channel 1000 iter 实验

命令：

```bash
.venv/bin/python train.py config/train_shakespeare_char_free_channel.py \
  --device=cpu \
  --compile=False \
  --max_iters=1000
```

默认配置：

```python
free_channel_routing = "ste"
free_channel_threshold = 0.5
free_channel_target_ratio = 0.4
free_channel_budget_weight = 0.01
free_channel_cost_weight = 0.0
free_channel_temperature = 2.0
free_channel_temperature_final = 0.5
free_channel_temperature_anneal_iters = 1000
```

分析结果：

```text
PPL                 = 7.82
mean active width   = 151.21 / 512
active ratio        = 29.53%
P10/P50/P90 width   = 95 / 140 / 216
P90 - P10           = 121
channel usage std   = 0.0952
```

关键观察：

虽然目标预算是：

```text
0.4 * 512 ~= 205 channels
```

但当 `free_channel_budget_weight = 0.01` 时，模型最终没有贴住 205，而是收敛到约 150 active channels。

这说明：

1. 当前预算约束较弱。
2. task loss 自身可能倾向于诱导出约 150 channels 的稀疏工作点。
3. token 间确实出现了 active width 分化，`P10/P50/P90 = 95/140/216`。

## 9. Budget Weight 对照实验

为了检查 budget 约束强度，只改一个变量：

```python
free_channel_budget_weight
```

其他配置全部不变。

每组训练 1000 iter，扫：

```text
0
0.001
0.01
0.1
1.0
```

只看四个核心指标：

- PPL
- 平均 active width
- `P90 - P10`
- channel usage std

结果如下：

| budget_weight | PPL | 平均 active width | P90 - P10 | channel usage std |
|---:|---:|---:|---:|---:|
| 0.0 | 7.85 | 147.22 | 119 | 0.0926 |
| 0.001 | 7.84 | 147.33 | 123 | 0.0947 |
| 0.01 | 7.82 | 151.21 | 121 | 0.0952 |
| 0.1 | 7.86 | 183.97 | 96 | 0.1046 |
| 1.0 | 7.92 | 200.67 | 89 | 0.1132 |

汇总文件：

```text
out-free-channel-budget-summary.csv
```

各组分析输出：

```text
out-free-channel-budget-0p0/free_channel_summary.csv
out-free-channel-budget-0p001/free_channel_summary.csv
out-free-channel-budget-0p01/free_channel_summary.csv
out-free-channel-budget-0p1/free_channel_summary.csv
out-free-channel-budget-1p0/free_channel_summary.csv
```

## 10. 当前解释

这组 sweep 显示出三个区间。

### 极弱或无预算约束

```text
budget_weight = 0.0, 0.001, 0.01
```

这三组基本收敛到同一工作点：

```text
mean active width ~= 147-151
```

PPL 最好也出现在这个区间。

同时 token-level 宽度分化最大：

```text
P90 - P10 ~= 119-123
```

这说明在当前模型、数据和训练长度下，即使没有显式 budget loss，task loss 也会把 gate 推向一个相对稀疏的自由宽度解。

### 中等预算约束

```text
budget_weight = 0.1
```

平均宽度被拉高到：

```text
mean active width ~= 184
```

但 token 间分化被压缩：

```text
P90 - P10 = 96
```

PPL 与弱约束区相比略差，但差距不大。

### 强预算约束

```text
budget_weight = 1.0
```

平均宽度基本贴近目标：

```text
mean active width = 200.67
target width      ~= 204.8
```

但代价是：

```text
PPL 变差到 7.92
P90 - P10 降到 89
```

说明强 global budget 可以控制平均宽度，但会压平一部分 token-level 自主分化。

## 11. 当前阶段结论

目前可以得到几个初步结论。

1. Free-channel dynamic width 是可训练的。
2. 每个 token 的 active channel count 确实会分化。
3. 每个 channel 的 usage rate 也出现差异，说明 channel specialization 有初步迹象。
4. 弱 budget loss 不足以控制最终平均宽度。
5. 强 budget loss 可以控制平均宽度，但会降低 token 间宽度分化。
6. 在当前设置下，模型自然偏好的 active width 大约是 150/512。

更概括地说：

> 自由通道机制本身是成立的，但 resource objective 的形式和强度会显著影响“平均计算量”和“token 间分化”之间的平衡。

## 12. 当前能力边界：逻辑计算量 vs 实际加速

当前 free-channel / dynamic-width 已经具备“减少理论计算宽度”的能力，但还没有真正实现机器上的加速运算。

现在模型可以学到：

```text
某个 token 只需要激活约 150 个 channel，而不是完整 512 个 channel
```

但当前实现仍然是：

```text
先计算完整 512 个 hidden channel
        ↓
再用 gate / mask 把不用的 channel 置零
```

也就是说，逻辑上某个 token 只用了：

```text
150 / 512 channels
```

但底层矩阵乘法仍然执行了完整的：

```text
X_128 x W_128x512
```

然后再把不需要的 channel mask 掉。

因此当前结果只能说明：

```text
逻辑计算宽度下降了
```

还不能说明：

```text
实际 wall-clock latency 或 tokens/s 按比例改善了
```

这两个概念必须分开。

真正的加速需要把 mask 之前的 dense 计算改成：

```text
先确定哪些 channel 要用
        ↓
只读取这些 channel 对应的权重
        ↓
只计算这些 channel
```

例如原来第一层 MLP 是：

```text
X_128 x W_128x512
```

如果某个 token 实际只需要 150 个 channel，真正加速版应该接近：

```text
X_128 x W_128x150
```

第二层 projection 也应该只处理这些 active channel：

```text
H_150 x W_150x128
```

这样才是真的减少 FLOPs 和权重读取。

## 13. 工程难点：完全自由 channel 不容易高效

当前 free-channel 是逐 channel 独立 gating。它的表达能力强，但硬件执行不友好。

例如：

```text
token A -> channels 1, 8, 12, 30, ...
token B -> channels 2, 7, 19, 44, ...
token C -> channels 5, 9, 16, 25, ...
```

不同 token 激活的 channel 集合都不一样。这样的非结构化稀疏模式很难用普通矩阵乘法高效执行。

如果强行逐 token gather/scatter 权重，可能会出现：

- kernel launch overhead 高
- 内存访问不连续
- batch 内 token 很难合并
- GPU 利用率下降
- 理论 FLOPs 降了，但实际速度不一定变快

所以真正系统加速通常需要更结构化的 gating。

## 14. 下一阶段建议：Block-wise Hard Routing

下一步不建议立刻写复杂 sparse kernel，而是先做一个简单的 block-wise hard routing 原型。

把 512 个 hidden channel 切成固定大小的 block：

```text
512 channels
        ↓
32 个 block
每个 block = 16 channels
```

模型不再自由开关单个 channel，而是自由开关 channel block：

```text
token A -> 开 7 个 block  = 112 channels
token B -> 开 12 个 block = 192 channels
token C -> 开 21 个 block = 336 channels
```

这样相比离散宽度：

```text
64 / 128 / 256 / 512
```

仍然自由很多，因为 token 可以选择任意 block 组合；但相比完全自由 channel，它更容易做真实跳算。

一个最小可行原型可以是：

```text
BlockRouter(x)
        ↓
得到 [B, T, 32] 的 block gate
        ↓
按 active block gather 对应的 c_fc / c_proj 权重
        ↓
只对 active block 做 sliced Linear
        ↓
scatter/add 回 residual stream
```

第一版甚至可以先在 CPU 或小 batch 上实现清楚，不追求最优 kernel，只验证：

```text
active width 下降
        ↓
实际 forward 时间下降
```

## 15. 阶段划分

当前项目可以按四个阶段理解。

阶段 1：

```text
证明不同 token 不需要固定宽度
```

状态：

```text
已完成
```

阶段 2：

```text
证明自由宽度可以自己分化
```

状态：

```text
已初步完成
```

阶段 3：

```text
把“逻辑没用的 channel”真正不计算
```

状态：

```text
未完成
```

阶段 4：

```text
验证 wall-clock latency / tokens per second
```

状态：

```text
未完成
```

因此当前工作已经有了加速的算法基础，但还没有把它转化成实际机器上的加速。

最终目标不应只是报告：

```text
平均 active width 从 512 降到 150
```

而应进一步验证：

```text
实际生成速度从 100 token/s 提升到 180 token/s，同时 PPL 基本不变
```

## 16. 下一步实验建议

建议下一步做：

1. 多 seed 重复 budget sweep，确认当前趋势是否稳定。
2. 对 `budget_weight = 0.01 / 0.1 / 1.0` 跑更长训练。
3. 在同样 budget sweep 下比较 `soft` 和 `ste`。
4. sweep `free_channel_target_ratio`，例如 `0.25 / 0.4 / 0.6`。
5. 分析逐层 channel usage，看早层和晚层是否形成不同 specialization。
6. 与之前离散 hard dynamic-width 模型做 matched-width / matched-PPL 对比。
7. 进一步分析 token 类型、字符频率、上下文复杂度与 active width 的关系。
8. 实现 block-wise hard routing，并加入真正 sliced Linear 的原型。
9. 对 dense / mask-only / sliced block-wise 三种 forward 做 wall-clock benchmark。
