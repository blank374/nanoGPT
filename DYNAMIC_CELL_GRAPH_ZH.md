# Dynamic Cell Graph MVP

这个实现把自由输入、自由计算宽度、自由深度和 Free-Q 合并成一个输入条件化的有序 DAG：

\[
G(x) = \{V(x), E(x)\}.
\]

它通过 `cell_graph=True` 启用，并与旧的动态深度、动态宽度、Free-Q 等实验开关互斥。旧模型路径保持不变。

## 图结构

- `n_layer` 表示最大 Step 数。
- `cell_graph_cells_per_step` 表示每个 Step 的物理 Cell 容量。
- 每个 Cell 只能读取 embedding anchor 和严格更早 Step 的 Cell，所以图天然无环。
- 统一的 `CellGraphRouter` 对每个 token 一次性产生全部节点门和边门。
- 路由只读取当前 token 的 embedding/position state，不对序列做未来信息池化。
- 某 Step 没有激活 Cell 时，当前状态原样跳过。
- 输出始终经固定 `ln_f` 和共享 `lm_head`，因此空图也有合法输出。

每个 Step 前 `cell_graph_attention_cells` 个 Cell 是 Attention Cell，其余是 Compute Cell。

Attention Cell 使用路由融合的历史状态形成 Q，使用当前阶段状态形成 K/V，并始终使用 causal attention。Compute Cell 固定为：

```text
n_embd -> cell_graph_atom_size -> n_embd
```

Cell 接口宽度始终为 `n_embd`；同时激活多个 Cell 增加的是计算宽度，而不是残差张量宽度。

## 三种自由度

- fan-in：激活节点所选择的入边数。
- width：一个 Step 中激活的 Cell 数。
- depth：激活边组成的最长有效路径。

最近一次前向的结构可通过 `model.cell_graph_route_records()` 取得；摘要位于 `model.last_cell_graph_stats`。

## 训练目标

总损失为：

\[
L=L_{LM}+\lambda_b L_{budget}+\lambda_e L_{edge}+\lambda_{bal}L_{balance}.
\]

- `cell_graph_target_node_ratio` 给出期望节点激活比例。
- `cell_graph_budget_weight` 控制节点预算约束。
- `cell_graph_edge_cost_weight` 惩罚过多候选边。
- `cell_graph_balance_weight` 防止所有输入长期只使用少数 Cell。
- 路由温度从 `cell_graph_temperature` 退火到 `cell_graph_temperature_final`。

节点和边在前向时使用硬二值决策，反向使用 straight-through estimator。每个 Cell 至少保留 anchor 作为输入回退，避免空 fan-in。

## 运行

```powershell
python train.py config/train_shakespeare_char_cell_graph.py
```

训练日志会报告 active cells、平均 width、最长路径 depth、fan-in、不同节点图数量，以及预算/边/均衡损失。

## 当前边界

这是研究正确性优先的 MVP。路由产生的图是真实的硬图，但当前 PyTorch 前向仍以稠密方式计算 Cell，然后应用硬门控；它还不会按 token 给 GPU 带来与理论 FLOPs 等比例的墙钟加速。下一步需要把常见图进行分组并编译成批量执行计划，或把路由粒度提升到 sequence/chunk，才能获得稳定的物理加速。

在执行优化之前，先按 [Fixed Compute + Free Connectivity 实验](CELL_GRAPH_FREE_EDGES_EXPERIMENTS_ZH.md) 验证边路由是否真正具有输入相关价值。
