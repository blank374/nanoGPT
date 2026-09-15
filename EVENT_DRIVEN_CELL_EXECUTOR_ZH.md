# Event-Driven Dynamic Cell Graph：第一阶段实现

日期：2026-09-14

## 已实现

本阶段不修改冻结的 Full-Free v1，也不改变 `128→64→128` Compute Cell。新增的执行层把 Cell 视为事件目标：

```text
Router产生硬节点/边
        ↓
构造ready Cell events
        ↓
按cell_id进入队列
        ↓
同一Cell的请求/token打包
        ↓
一次Cell调用
        ↓
index_copy scatter回原位置
```

主要能力：

- `CellEventBatch`：携带 Cell ID、融合输入、active mask和依赖数量；
- 相同 Cell 的多个请求分片可合并为一次 packed 调用；
- `PersistentCellEventScheduler` 支持跨独立提交保留队列，并以事件阈值、最大等待时间或强制drain触发执行；
- 最大队列优先的确定性调度；
- 未激活 Cell/token 不执行；
- 依赖 ready-wave 构造和循环/死锁检测；
- 可选多 CUDA stream；
- 每一步和整张图的事件数、队列长度、fan-in、Cell launches统计；
- 与稠密门控、旧 physical queue 的三路交错测速；
- FP16 autocast 下 scatter 回目标状态 dtype；
- 推理结果与旧执行方式严格一致。

实现文件：

- `experiments/event_driven_cell_executor.py`
- `experiments/benchmark_event_driven_cells.py`
- `tests/test_event_driven_cell_executor.py`
- `experiments/full_free_attention_v2.py` 中的接入开关：
  - `event_driven_execution`
  - `event_parallel_streams`

## GTX 1650 FP16 结果

条件：batch=128、block=128，AB/BA交错计时。

### 稀疏 B=8 模型

平均 active Cells：5.364/32。

| 执行方式 | 中位延迟 | 相对稠密 |
|---|---:|---:|
| Dense Cells | 424.76 ms | 1.000× |
| 旧 physical queue | 400.43 ms | 1.061× |
| Event queue | **396.33 ms** | **1.072×** |

事件执行相对稠密 Cell 延迟下降6.69%，吞吐提高7.17%；相对旧 queue 进一步减少约1.0%延迟。三种输出的 logits 最大绝对误差均为0。

该批次产生87,885个 ready events，28/32个 Cell 队列非空。热点 Cell 队列约16,384个事件，长尾队列只有1–35个事件，说明下一阶段应合并或延迟长尾小队列。

多 CUDA stream 的中位结果为411.37 ms，相对对应稠密基准提升1.068×，没有超过单 stream。当前GPU上创建/同步streams和并发小GEMM没有额外收益，默认保持关闭。

### 七层质量模型

PPL 7.7589的质量模型平均 active Cells：22.546/32。

| 执行方式 | 中位延迟 | 相对稠密 |
|---|---:|---:|
| Dense Cells | 440.74 ms | 1.000× |
| 旧 physical queue | 440.59 ms | 1.000× |
| Event queue | **435.69 ms** | **1.012×** |

所有32个队列都非空，事件层只能带来约1.16%的 Cell 执行提升。加上此前固定跳过一层 Attention 的约1.8%，当前质量路径的端到端收益仍属于几个百分点，而不是激进模型的两位数收益。

## 当前边界

这是 correctness-first 的事件执行器，不是最终的完整异步服务运行时。持久调度器已经能跨独立事件提交攒批，但尚未接管同步的 `model.forward()`；真正的服务循环需要凭返回 ticket 恢复各请求状态。当前 v2 Router 和 residual state 在每个 Step 之间仍有同步屏障，Python列表、字典、gather/scatter也仍有开销。

下一阶段需要：

1. 将请求状态放入持久 state arena；
2. 用GPU环形ready queues保存跨forward事件；
3. 为每个请求/节点维护依赖计数；
4. 使用“队列达到阈值或等待超时”触发策略；
5. 将多个不同Cell的小队列改为grouped GEMM；
6. 将高频路径编译为CUDA Graph，长尾交给事件调度；
7. 在并发1/8/32/128请求下报告tokens/s与P50/P95延迟。

## 结论

第一阶段已经证明：现有 Full-Free Cell 可以在不改变模型输出的情况下转换为ready-event/Cell-queue执行；在真正稀疏的B=8模型上，事件调度产生了可测的7.17%吞吐提升。但质量模型仍然过密，因此执行器本身不能替代路由边际价值校准。
