# Free-Fan-In Query 实验

本实验只改变 Query 的输入来源。Key、Value、causal attention、MLP 宽度和
Transformer 深度保持标准实现，并且训练 loss 只有语言模型交叉熵。

## 架构

每个 attention head 分别产生四个 source logit：

```text
[current, previous, earlier, anchor]
```

路由粒度是 token-level，shape 为 `[B,T,n_head,4]`。这是为了保持训练的
因果性；当前数据管线没有 prompt/request 边界，使用整个 sequence 的 pooled
表示会令早期 token 的 route 泄漏未来 token。

定义 residual 时间线 `r0=embedding`、`r1=Block0 输出`、……。第 `l` 层使用：

```text
current  = LN_l(r_l)
previous = LN_l(r_(l-1))
earlier  = LN_l(r_(l-2))
anchor   = LN_l(r0)
```

完全相同的浅层 alias 会被去除。因此四层模型实际 availability 是：

```text
L0: current, anchor
L1: current, anchor
L2: current, previous, anchor
L3: current, previous, earlier, anchor
```

历史 residual 使用当前层 `ln_1` 归一化，避免 selector 只利用 source scale
差异。Identity/linear 指归一化之后是否再使用独立的 `128 -> 128` source
projection。

QKV 参数仍保存在 packed `c_attn` 中。K/V 直接来自 `c_attn(current)`；每个
head 的 Q 使用相应的 Wq 行切片作用在该 head 的 source mixture 上。

## Selector

`binary` 在训练时使用 stochastic Bernoulli STE，评估时以 threshold 产生
hard mask。如果全部关闭，强制保留可用 source 中概率最大的一个。

`sparsemax` 不指定 top-k，直接产生和为 1 且可含精确零的竞争权重。hard
fan-in 是正支持集大小。

两者都没有 compute、fan-in、load-balancing 或 head-specialization loss。

## 训练

准备 Shakespeare 数据后运行：

```powershell
python train.py config/train_shakespeare_char_free_q_baseline.py
python train.py config/train_shakespeare_char_free_q_binary.py
python train.py config/train_shakespeare_char_free_q_sparse.py
```

三组配置均为 4 层、4 head、`n_embd=128`、标准 512-wide MLP。运行 linear
source projection 对照时可覆盖配置：

```powershell
python train.py config/train_shakespeare_char_free_q_binary.py `
  --free_q_source_projection=linear --out_dir=out-free-q-binary-linear
```

训练期间生成：

- `free_q_history.csv`：step、train loss、val PPL、全局 fan-in/source/mask 指标；
- `free_q_layers_history.csv`：逐层聚合的 source usage 与 fan-in；
- `free_q_heads_history.csv`：每个 layer/head 的 source usage、平均概率、fan-in
  distribution 和 mask entropy。

## Fixed、Shuffle、Oracle 与矩阵

训练结束后：

```powershell
python experiments/free_q_analysis.py `
  --out_dir=out-free-q-binary-identity `
  --baseline_out_dir=out-free-q-baseline --device=cuda
```

分析使用同一批固定抽样的 validation requests 比较：

- learned dynamic routing；
- 每个 layer/head 的 most-common fixed mask；
- 在 batch 的 request 维循环置换完整 token route 的 shuffled routing；
- fixed current、current+previous、current+earlier、all-available；
- validation subset 上逐 request、逐 layer/head 的 coordinate-wise oracle。

Oracle 固定其他 head 的已学习 hard mask，只枚举当前 head 的所有可用非空
mask；它不是不可计算的全 16-head 联合 oracle。

主要输出：

```text
free_q_summary.csv
free_q_source_usage_matrix.csv
free_q_source_usage_heatmap.png
free_q_fanin_by_head.csv
free_q_fanin_by_layer.csv
free_q_mask_diversity.csv
free_q_oracle_by_request_head.csv
free_q_routes.npz
```

`free_q_routes.npz` 保存每批输入 token、每层 hard mask 和 source probability。
Shuffle 在 request 维移动整条 `[T,H,S]` route，因此精确保留 hard source usage
和 fan-in distribution，只破坏 route 与原输入内容的对应。

`mask_entropy` 是经验 bitmask 分布熵；`conditional_mask_entropy_token` 是
`H(mask | token_id)`，两者之差作为当前 token 与 route 的互信息估计。它只
捕捉 token identity，不等同于完整上下文条件分析，因此 Fixed/Shuffle PPL
仍是判断 input dependence 的主要证据。

## 解释原则

只有完成足够训练后才能判断是否产生自主 head specialization。初始化或
smoke test 中的 mask 差异不构成实验结果。若 binary 全开，应先看 sparsemax；
若全部退化为 current，应比较固定多 source PPL，再检查 source normalization、
router collapse 与 linear projection 对照。

## 300-step CPU pilot（非正式结论）

为验证完整实验链路，在 CUDA venv 的基础 Python 缺失时使用 CPU 跑了三组
matched pilot：seed 1337、4 层、4 head、128 hidden、batch 8、context 64、
300 steps。随后在三组模型完全相同的 16×8 条 validation requests 上分析。

| 模型/干预 | PPL |
|---|---:|
| Standard Q baseline | 11.7751 |
| Binary learned | 11.8826 |
| Binary fixed most-common | 11.8725 |
| Binary shuffled | 11.8890 |
| Sparsemax learned | 11.8189 |
| Sparsemax fixed most-common | 11.9049 |
| Sparsemax shuffled | 11.9325 |

Sparsemax learned route 相对自身 fixed 和 shuffle 分别改善约 0.086 和 0.114 PPL，
但仍比标准 Q baseline 差约 0.044 PPL。因此 pilot 支持“路由具有输入相关价值”，
不支持“Free-Q 已优于标准 Q”。Binary 在当前训练长度下与 fixed/shuffle 基本
持平，更像静态结构搜索。

Sparsemax 的全局平均 fan-in 为 1.644：fan-in 1/2/3/4 的比例分别为
41.23%/53.92%/4.06%/0.79%。不同 head 已出现明显偏好，例如 L3-H1 的 earlier
usage 为 87.12%，而 L3-H3 的 current/previous usage 为 88.79%/67.25%；两者
平均 fan-in 分别为 1.563 和 1.850。经验 `H(mask)=1.770`，
`H(mask|token_id)=1.570`，说明 mask 并非固定，token identity 能解释其中一小部分。

这些数字只证明实现与分析能够捕捉预期现象。正式结论仍应使用原配置完成更长
训练、多 seed、置信区间及 linear projection 对照。
