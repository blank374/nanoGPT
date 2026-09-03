# 自组织计算路径实验

本实验不给模型预定义深度模板。每个 request 在进入第一个 Transformer Block
以前，由一个轻量 Router 一次性产生 `[B, n_layer, 5]` 决策：

```text
每层资源 = 0 / 64 / 128 / 256 / 512
四层理论路径 = 5^4 = 625
```

默认 `dynamic_resource_skip_mode='mlp'`：0 只跳过 MLP，Attention 保持不变。
`'block'` 也已支持，用于后续整 Block Skip 对照实验。

训练目标为动态路径真实标签 CE、完整 512×4 路径 CE、完整路径蒸馏和渐增的
expected compute penalty。代码不加入 path balance 或 path concentration loss，避免人为
制造“路径分化”。

## 快速验证

```powershell
python experiments/dynamic_resource_smoke.py
```

覆盖三条手工路径的 forward/backward、自动 Router 梯度、625 路径编码、MLP Skip
research/physical 数值等价和 Whole Block Skip。

## 训练

```powershell
python train.py config/train_shakespeare_char_dynamic_resource.py
```

训练过程写入 `out-shakespeare-char-dynamic-resource/dynamic_resource_history.csv`，包含
PPL、平均计算、活跃层、平均宽度、Unique Paths、Top-1/4/8/16 coverage、Router entropy
和 collapse warning。

## 综合评价与真实延迟

```powershell
python experiments/dynamic_resource_eval.py `
  --checkpoint=out-shakespeare-char-dynamic-resource/ckpt.pt
```

报告 Full/Dynamic PPL、路径分布、按首 token 模式估计的条件路径熵，以及 Router、
Attention、MLP、prefill、decode、total latency、tokens/s 和真实 speedup。理论 compute
saving 与实际 wall-clock speedup 分开报告。

## Compute–Quality Pareto sweep

```powershell
python experiments/dynamic_resource_pareto.py --penalties=0,0.01,0.05,0.10
```

四组实验使用相同模型、数据、batch 和训练步数，最终合并到
`out-dynamic-resource-pareto/pareto_summary.csv`。

