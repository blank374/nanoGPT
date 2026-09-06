# Full-Free Dynamic Cell Graph v1

## 目标与隔离边界

Full-Free 用一张输入条件化 DAG 同时派生 fan-in、计算宽度、计算深度、skip 和路径。它不使用单独的 Width/Depth Router，也不预定义 Fast/Medium/Slow。

旧 Dynamic Cell Graph 保持为 `cell_graph_mode='legacy'`；新实验必须显式设置：

```python
cell_graph = True
cell_graph_mode = 'full_free'
```

因此已有 fixed-compute/free-edge checkpoint 和配置仍按原路径加载。

## v1 结构

默认容量是 8 Steps × 4 Compute Cells。每个 Step 先执行固定标准 causal attention，然后进行图计算：

```text
current hidden ─→ shared 128→32 graph context
                          ├→ local node sparsemax (4 Cells + Skip)
                          └→ local edge sparsemax

h_s = h_(s-1) + Σ_j a_(s,j) C_(s,j)(z_(s,j))
```

- `a=0` 是 sparsemax 产生的精确零；不使用 Top-k。也实现了可选 `entmax15`。
- Skip 与四个 Cell 共同竞争，因此一个 Step 可以有 0–4 个活跃 Cell。
- 所有 Cell 固定为 `128→64→128`，返回 128 维 residual delta。
- Cell 输出只求和，不 concat；representation dimension 永远是 128。
- 边只能连接 current state 和最近 3 个更早 Step 的 Cell 输出。
- 8×4 时合法候选边为 320 条，不生成旧实现的 1056 维扁平边向量。
- 输入先按 sparse edge weight 融合，再做 RMSNorm。
- `identity|linear` input projection 均可用，默认 identity。
- 可选 HALT 已接入同一局部竞争，默认关闭。
- Research Mode 仍稠密执行底层 Cell；日志中的 skipped MACs 是理论值，不是真实 GPU 加速。

## Natural pilot

```powershell
F:\Environment\conda_envs\pytorch\python.exe train.py `
  config/train_shakespeare_char_full_free_natural.py
```

Natural 的总目标严格为 LM cross entropy。以下权重均为 0：

```text
node budget / edge cost / depth cost / balance loss
```

`full_free_graph_history.csv` 在每次 eval 保存 active Cell/edge、逐 Step width、真实最长路径、图 entropy/coverage、usage entropy/Gini、dead-cell ratio 和理论 MACs。

## 单一 Budget

Budget 模式只约束 batch/token 平均 active Cell 数：

```text
L = L_LM + λ (C_soft - B) / C_max
λ ← max(0, λ + dual_lr (C_hard - B) / C_max)
```

`C_soft` 使用 sparsemax support 的平滑近似，避免误用每步恒定的 simplex 权重和；`C_hard` 是日志中的真实非零 Cell 数。λ 会随 checkpoint 保存和恢复。

单独运行 B=16：

```powershell
F:\Environment\conda_envs\pytorch\python.exe train.py `
  config/train_shakespeare_char_full_free_budget.py
```

## 完整三 seed 与 Pareto

1000-step 单 seed Natural pilot：

```powershell
F:\Environment\conda_envs\pytorch\python.exe `
  experiments/run_full_free_cell_graph.py --pilot
```

完整 3000-step、三 seed、预算 `4/6/8/12/16/24/32`：

```powershell
F:\Environment\conda_envs\pytorch\python.exe `
  experiments/run_full_free_cell_graph.py
```

每个预算点先训练动态模型，再根据其平均实际 Cell 数导出 `matched_static_graph.npz`，随后从头训练对应静态图。最终输出 `full_free_pareto.csv` 以及 PPL–active-cells、PPL–MACs 图。

## 因果干预与分析

```powershell
F:\Environment\conda_envs\pytorch\python.exe `
  experiments/full_free_cell_graph_analysis.py `
  --out_dir=out-full-free-natural-seed1337 `
  --static_out_dir=out-full-free-static-seed1337 `
  --reference_out_dir=out-full-free-static-seed1337
```

干预包括：

1. Learned graph；
2. Fixed-most-common node + edge graph；
3. 跨 request 的完整 node/edge weight shuffle；
4. Same-token shuffle；
5. Same-token + position-bucket shuffle；
6. 独立训练 matched static graph。

分析还输出：

- `full_free_difficulty_quintiles.csv`：固定 reference NLL 五分位的计算分配；
- `full_free_marginal_cells.csv`：逐 Cell remove/add 反事实 NLL；
- `full_free_specialization.csv`：usage、来源、fan-in、token、位置、难度和消融影响；
- `full_free_structure_tokens.csv`：逐 token path/branch/merge 结构；
- `top8_graphs.json`、DOT 和 edge-list CSV；
- Router 参数、理论 MACs、单独 CUDA latency 与 Router/model latency ratio；
- 原始 node/edge weights、硬 mask 和 depth 的压缩 NPZ。

没有提供 `--reference_out_dir` 时，脚本会明确标记 difficulty 为模型自身 NLL proxy；正式结论必须传入固定 Reference Model。

## 判定

- Level 1：动态图胜过 Fixed/Shuffle，且宽、深、fan-in 随输入变化。
- Level 2：不同 token 得到不同计算量，且预算下分配与边际收益一致。
- Level 3：Dynamic quality–compute Pareto 超过独立训练的 matched static Pareto；Physical Mode 还需进一步证明真实 GPU latency 降低。
