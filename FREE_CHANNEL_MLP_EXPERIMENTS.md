# Free-Channel Dynamic Width MLP Experiments

This note summarizes the current dynamic-width MLP work in this nanoGPT fork.

## 1. Research Question

The project asks whether MLP compute width can emerge from training instead of
being fixed by hand.

The first validated stage used a discrete dynamic-width MLP with four predefined
hidden widths:

```text
64 / 128 / 256 / 512
```

With STE routing, temperature annealing, hard-routing-aware loss, and compute
cost, that setup reached roughly:

```text
dense baseline hard PPL ~= 7.17
dynamic hard PPL       ~= 7.19
mean hard width        ~= 205 / 512
```

This established that discrete hard dynamic width is viable. The next stage
removes the predefined width choices and lets each MLP hidden channel decide
whether to participate for each token.

## 2. Current Free-Channel Design

The Transformer trunk is unchanged:

- `n_embd = 128`
- max MLP hidden width = `4 * n_embd = 512`
- attention, QKV, residual stream width, embeddings, LayerNorm, and depth are
  unchanged

Only the MLP intermediate channels are gated.

The new module is:

```python
class FreeChannelMLP(nn.Module):
```

Its structure is:

```text
[B, T, 128]
  -> c_fc
[B, T, 512]
  -> GELU
  -> per-channel gate
  -> c_proj
[B, T, 128]
```

The full parameter matrices remain present:

```python
c_fc:   128 -> 512
c_proj: 512 -> 128
```

For each token, a separate router produces:

```python
gate_logits = gate_network(x)      # [B, T, 512]
gate_prob = sigmoid(gate_logits / temperature)
```

Each channel is structurally symmetric. There are no predefined core channels,
fast channels, slow channels, rare channels, or prefix/nested channel groups.

## 3. Routing Modes

The implementation supports two modes.

Soft routing:

```python
hidden = gelu(c_fc(x))
hidden = hidden * gate_prob
out = c_proj(hidden)
```

STE routing:

```python
hard_gate = (gate_prob > threshold).float()
gate = hard_gate + gate_prob - gate_prob.detach()
hidden = hidden * gate
```

STE uses real 0/1 gates in the forward pass while keeping gradients through the
sigmoid probabilities.

No Top-K is used. No fixed active channel count is imposed. The active width is
measured as:

```python
active_channels = gate.sum(dim=-1)
```

## 4. Resource Objective

The main resource constraint is a global average budget:

```python
mean_gate_ratio = gate.mean()
budget_loss = (mean_gate_ratio - free_channel_target_ratio) ** 2
```

The training loss is:

```python
loss = task_ce + free_channel_budget_weight * budget_loss
```

A simple L1-style optional ablation is also supported:

```python
channel_cost = gate_prob.mean()
loss += free_channel_cost_weight * channel_cost
```

The current main experiments use:

```python
free_channel_target_ratio = 0.4
free_channel_cost_weight = 0.0
```

So the target average width is:

```text
0.4 * 512 ~= 205 channels
```

This is not a per-token requirement. Individual tokens may use much smaller or
larger widths.

## 5. Monitoring

The implementation records the following free-channel statistics:

- mean active channels
- median active channels
- std active channels
- min/max active channels
- gate entropy
- fraction of gates above `0.5`
- fraction of gates above `0.9`
- fraction of gates below `0.1`
- active width histogram
- P10/P25/P50/P75/P90 active width
- per-channel usage rate
- channel usage std/min/max

The active width histogram bins are:

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

## 6. Main Files

Implementation:

- `model.py`
  - `FreeChannelMLP`
  - free-channel config fields in `GPTConfig`
  - free-channel budget/cost loss
  - free-channel runtime statistics
- `train.py`
  - free-channel training options
  - temperature annealing
  - eval and training logs
- `config/train_shakespeare_char_free_channel.py`
  - default Shakespeare char free-channel training config
- `experiments/free_channel_analysis.py`
  - checkpoint analysis
  - exports summary, layer, and channel usage CSV files

## 7. Baseline Free-Channel Run

Command:

```bash
.venv/bin/python train.py config/train_shakespeare_char_free_channel.py \
  --device=cpu \
  --compile=False \
  --max_iters=1000
```

Default free-channel settings:

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

Result after analysis:

```text
PPL                 = 7.82
mean active width   = 151.21 / 512
active ratio        = 29.53%
P10/P50/P90 width   = 95 / 140 / 216
P90 - P10           = 121
channel usage std   = 0.0952
```

Important observation:

The model did not stay near the requested global budget of about 205 channels
when `free_channel_budget_weight = 0.01`. It converged to roughly 150 active
channels while maintaining good perplexity and clear token-level width
differentiation.

## 8. Budget Weight Sweep

The first clean ablation changed only:

```python
free_channel_budget_weight
```

All other settings were held fixed. Each run used 1000 training iterations.

| budget_weight | PPL | mean active width | P90 - P10 | channel usage std |
|---:|---:|---:|---:|---:|
| 0.0 | 7.85 | 147.22 | 119 | 0.0926 |
| 0.001 | 7.84 | 147.33 | 123 | 0.0947 |
| 0.01 | 7.82 | 151.21 | 121 | 0.0952 |
| 0.1 | 7.86 | 183.97 | 96 | 0.1046 |
| 1.0 | 7.92 | 200.67 | 89 | 0.1132 |

Summary CSV:

```text
out-free-channel-budget-summary.csv
```

Per-run analysis outputs:

```text
out-free-channel-budget-0p0/free_channel_summary.csv
out-free-channel-budget-0p001/free_channel_summary.csv
out-free-channel-budget-0p01/free_channel_summary.csv
out-free-channel-budget-0p1/free_channel_summary.csv
out-free-channel-budget-1p0/free_channel_summary.csv
```

## 9. Interpretation So Far

The sweep suggests three regimes.

Very weak or no budget pressure:

```text
budget_weight = 0.0, 0.001, 0.01
```

These runs all converge to about 147-151 active channels. PPL is best in this
region, and token-level width differentiation is strongest:

```text
P90 - P10 ~= 119-123
```

Moderate budget pressure:

```text
budget_weight = 0.1
```

The average width rises to about 184 channels. Token differentiation remains,
but the distribution is compressed:

```text
P90 - P10 = 96
```

Strong budget pressure:

```text
budget_weight = 1.0
```

The average width is close to the target:

```text
mean active width = 200.67
target width      ~= 204.8
```

But PPL is slightly worse, and token width spread is further reduced:

```text
P90 - P10 = 89
```

Overall, the free-channel mechanism appears to work: channels do not all behave
identically, active width differs across tokens, and per-channel usage rates
spread out. However, a strong global budget term can partially flatten the
token-level width distribution.

## 10. Current Working Hypotheses

1. The task loss alone induces a preferred sparse operating point around
   150 active channels for this model/data/training budget.
2. A weak global budget term does not meaningfully control the final average
   width.
3. A strong budget term can control mean width, but may reduce token-level
   adaptive differentiation.
4. Channel specialization is present: channel usage rates have nontrivial spread,
   and the strongest channels are used far more often than the weakest channels.

## 11. Suggested Next Experiments

Useful next checks:

1. Repeat the budget sweep with multiple random seeds.
2. Run longer training for `budget_weight = 0.01`, `0.1`, and `1.0`.
3. Compare soft routing vs STE routing under the same budget sweep.
4. Sweep target ratios, for example `0.25`, `0.4`, `0.6`.
5. Analyze per-layer channel usage separately, because early and late layers may
   specialize differently.
6. Compare against the prior discrete hard dynamic-width result at matched PPL
   and matched active width.

