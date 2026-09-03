# 自动分化的动态深度快速路径（第一阶段）

本阶段只研究深度，不启用动态 MLP 宽度。模型为 4 层、`n_embd=128`、4 头；普通 MLP 的中间宽度固定为 `4×128=512`。Attention/QKV 公式没有改动。

## 通俗思路

路由器像“门口分诊”：读取整段请求的 embedding 平均表示，在进入 Transformer 主干前只判断一次该请求需要 D2、D3 还是 D4。熟悉、容易的请求走较短前缀；困难请求继续走更深前缀。一个请求后续生成的所有 token 复用同一深度和同一静态执行计划，不会每个 token 重新选择。

训练时仍执行 4 层，因为需要同时知道 D2、D3、D4 的答案质量。三个深度复用同一个最终 `ln_f + lm_head`，没有新增三个词表头。损失包含：

- D4 主任务 CE；
- D2/D3 的 early CE；
- D4 向 D2/D3 的蒸馏；
- 请求级 oracle 路由监督：选择 CE 不超过 D4 加容差的最浅路径；
- warmup 后逐渐增加的期望深度成本；
- 小量熵正则和 D4 探索，降低早期塌缩风险。

## 真正跳层的位置

部署路径不是“算完 4 层再 mask”。单一路径 batch 只启动前 N 个 block；混合 batch 使用逐层收缩或缓存的静态路由计划：前两层处理全批，D2 退出，第三层处理剩余请求，D3 退出，第四层只处理最后一组。输出最终恢复到原 batch 顺序。

`generate()` 在请求开始时选择一次深度。batch=1 将深度缓存为 Python 整数；混合 batch 缓存排序索引及每层退出数量，避免 decode 每步发生 GPU 到 CPU 的路由同步。

## 运行

```powershell
.venv-cuda\Scripts\python.exe train.py config\train_shakespeare_char_dynamic_depth.py
.venv-cuda\Scripts\python.exe experiments\dynamic_depth_eval.py
.venv-cuda\Scripts\python.exe -m unittest tests.test_dynamic_depth
```

可用 `--cost-bias` 调整部署时质量/速度工作点；它只改变深度分数中的成本偏置，不重新训练模型。

## GTX 1650 第一阶段结果

1000 步 Shakespeare char 实验在固定随机种子、100 个验证 batch 上的结果为：固定 D2/D3/D4 PPL 分别 8.107/7.938/7.896，动态 PPL 7.939；路由 D2 0%、D3 92.5%、D4 7.5%，平均深度 3.075，理论层计算减少 23.125%，路由熵 0.9061，top-1/top-3 路径覆盖率为 92.5%/100%。

真实 CUDA 结果并不等于理论 FLOPs：batch=1 的缓存深度 decode 达到 1.104×，但首次 prefill 为 0.871×；batch 8/32 的 decode 为 0.927×/0.917×。小模型只有四层，路由、分组和张量整理开销很容易吃掉一层节省。因此本阶段证明了质量可控的深度分化和单请求 decode 加速，但没有证明混合大 batch 端到端加速。

下一阶段若继续，应优先做按深度排队的请求调度、每个深度的 CUDA Graph、减少 Python/动态 shape 开销，并在 8–12 层模型上复测；之后再与 GPU 原子级 MLP 宽度组合。不要先把两种路由耦合，否则无法判断收益来自深度还是宽度。
