# Fixed Compute + Free Connectivity 实验

本阶段只回答一个问题：Dynamic Cell Graph 是否会根据输入重组信息流，还是只会搜索出一张静态网络。GPU kernel 裁剪、自由宽度、自由深度和 Free-Q 均不在本阶段范围内。

## 受控架构

```text
4 Steps × 4 Compute Cells
每个 Cell: 128 -> 64 -> 128
每 Step 固定激活 C0、C1
每 Step 固定执行一个标准 causal attention
深度固定为 4 Steps
```

只有激活 Compute Cell 的历史输入边可以变化。节点路由被旁路，因此每个 token 的 Cell 数和理论计算量完全一致。Attention 不使用 routed Q；它是每 Step 都执行的标准固定模块。

路由是 token-level，并且只读取该 token 的 embedding/position state。Cell 只能选择当前 backbone state 或严格更早 Step 的 Cell 输出，因而不会产生图环或未来 token 泄漏。

## 三个训练条件和两个干预

| 条件 | 配置/来源 | Edge | Cost | Balance |
|---|---|---|---:|---:|
| A Fixed | `train_shakespeare_char_cell_graph_edges_fixed.py` | current-only 固定图 | 0 | 0 |
| B Learned | `train_shakespeare_char_cell_graph_edges_learned.py` | 动态学习 | 0 | 0 |
| C Budget | `train_shakespeare_char_cell_graph_edges_budget.py` | 动态学习 | 0.01 | 0 |
| D Most-common | B checkpoint 的评估干预 | 每 Cell 的训练后众数 mask | 与 B 相同 | 0 |
| E Shuffle | B checkpoint 的评估干预 | 在 request 维打乱完整 token route | 与 B 相同 | 0 |

B 是“无代价自然生长”条件：训练目标严格退化为语言模型交叉熵。A、B、C 均关闭 node budget 和 balance loss。C 只增加 edge cost。

D/E 不重新训练权重。它们冻结 B 的模型参数，仅替换评估时的边，使比较直接回答输入相关 routing 是否有价值。Shuffle 在 request 维置换完整 `[T,N,S]` route，保持 fan-in、Cell/source 使用频率和序列位置结构，只破坏 route 与原输入的对应。

## 运行三种子正式实验

```powershell
python experiments/run_cell_graph_free_edges.py
```

默认执行 seeds `1337 2027 3407`，每个训练条件 3000 steps。先查看命令而不执行：

```powershell
python experiments/run_cell_graph_free_edges.py --dry_run
```

也可以单独运行：

```powershell
python train.py config/train_shakespeare_char_cell_graph_edges_fixed.py
python train.py config/train_shakespeare_char_cell_graph_edges_learned.py
python train.py config/train_shakespeare_char_cell_graph_edges_budget.py
```

## 干预与结构分析

```powershell
python experiments/cell_graph_free_edges_analysis.py `
  --learned_out_dir=out-cell-graph-edges-learned-seed1337 `
  --fixed_out_dir=out-cell-graph-edges-fixed-seed1337 `
  --budget_out_dir=out-cell-graph-edges-budget-seed1337
```

输出包括：

- `cell_graph_edge_interventions.csv`：Learned、Most-common、Shuffle、canonical fixed 和独立训练 A/C 的 PPL；
- `cell_graph_edge_summary.json`：全图熵、Top-1/4/8 coverage 和两个核心成功判断；
- `cell_graph_edge_diversity_by_cell.csv`：每个 Cell 的 \(H(E_j)\)、Top-1/4/8 input-mask coverage、fan-in、source usage；
- `cell_graph_edge_diversity_by_step.csv`：逐 Step graph entropy 与 coverage；
- `cell_graph_edge_routes.npz`：原始 token、edge mask、most-common graph。
- `cell_graph_free_edges_multiseed.csv`：三 seed 的逐 seed 结果、均值、样本标准差和标准误。

另外计算 \(H(E\mid token\_id)\) 和估计的 token/mask mutual information。它只能解释 token identity，不能代替 Fixed/Shuffle PPL 干预。

## 判定规则

第一块地基要求至少同时观察到：

1. Learned PPL 小于 Fixed-most-common PPL；
2. Learned PPL 小于 Shuffled PPL；
3. 不同 Cell 的 source usage 明显不同；
4. 同一 Cell 的 edge-mask entropy 不接近零，且 Top-1 coverage 不接近 100%。

若 Learned 与 D/E 没有稳定差异，或者绝大多数 Cell 的 Top-1 mask coverage 接近 100%，应把当前结果解释为静态架构搜索，而不是输入条件化计算图。只有至少三个 seed 的方向一致后，才进入自由 width 阶段。
