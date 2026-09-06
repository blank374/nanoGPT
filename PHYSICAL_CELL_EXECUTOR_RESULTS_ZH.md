# Full-Free v1 Physical Cell Executor 初步结果

日期：2026-09-04  
设备：NVIDIA GeForce GTX 1650  
精度：FP16 autocast  
序列长度：128  
模型：B=8 seed1337（实测平均约 7.64/32 active Cells）

## 实现

`experiments/full_free_physical_executor.py` 在不修改冻结模型文件的前提下，为 Full-Free v1 临时安装物理执行路径：

1. Router 与 causal Attention 保持 dense；
2. 按 Cell 收集 node mask 非零的 token；
3. 每个非空 Cell queue 只执行一次 RMSNorm + 128→64→128 MLP；
4. 将结果 scatter 回 `[batch, time, channel]`；
5. 空 Cell 不启动 MLP kernel。

Dense 与 queued executor 的 NLL 完全相同，logits 最大差值为 0（FP16 测试），节点和边 mask 完全一致。

## 实测结果

| batch | token positions | Dense ms | Queue ms | speedup | 延迟变化 |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,024 | 151.30 | 163.18 | 0.927× | +7.85%（变慢） |
| 32 | 4,096 | 273.65 | 273.71 | 1.000× | +0.02%（持平） |
| 64 | 8,192 | 397.54 | 387.87 | 1.025× | -2.43% |
| 128 | 16,384 | 648.58 | 616.98 | 1.051× | -4.87% |

每个点使用相同 checkpoint、相同输入，经过 warmup 后取三次或五次重复的中位数。batch=64 和 batch=128 的每次重复均为 queued executor 更快。

## 结论

`logical sparsity → physical speedup` 已得到端到端实测支持，但存在明确的批量门槛：当前纯 PyTorch gather/scatter 原型在小 batch 上调度成本过高；当一次前向包含约 8K 以上 token positions 时，队列足够大，开始产生 2.4%–4.9% 的真实加速。

随后使用无损推理包做了更严格的 AB/BA 交错计时。batch=128、五轮中位数为 Dense 666.50 ms、Queue 628.93 ms，即 **1.060× / 延迟下降 5.64%**；NLL、logits 和图仍完全一致。该交错结果作为当前更可靠的速度数字。

这还不是最终 executor。当前仍有以下未裁剪成本：Router、Attention、源融合、source projection、Python 的 32 Cell 循环，以及每个 Cell 独立的 gather/scatter。下一步应把同一步的 Cell queues 合并为 grouped GEMM，并用设备端 prefix-sum/dispatch 消除 Python/CPU 同步。当前机器没有 `nvcc` 或 Triton，因此本轮只实现了可直接运行的 PyTorch基线。

## 结果文件

- `full_free_physical_executor_b8_batch32_fp16.json/csv`
- `full_free_physical_executor_b8_batch64_fp16.json/csv`
- `full_free_physical_executor_b8_batch128_fp16.json/csv`
- `full_free_physical_executor_benchmark_fp16.json/csv`
- `full_free_inference_package_b128_interleaved_fp16.json/csv`

## 无损部署体积

`experiments/export_full_free_inference.py` 删除 optimizer、GradScaler 和训练配置历史，只保留原始 FP32 权重与构图参数。Natural 与 B=8 的训练 checkpoint 均从 13,427,557 bytes 降到约 4,474,8xx bytes，减少 **66.67%**。逐 tensor 权重位级相同，逐 token NLL 位级相同，节点/边 mask 完全相同。
