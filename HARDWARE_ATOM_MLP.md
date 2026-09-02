# Hardware-Aligned Retrieval MLP

This prototype implements the design rule:

> Routing freedom lives between GPU-executable atoms, not individual channels.

## Execution model

For the supplied Shakespeare configuration, the normal 512-channel MLP is
split into eight contiguous 64-channel atoms. A 128-D transformer state is
projected to a 16-D normalized query and compared with three learned mode keys:

- 2 atoms / 128 channels
- 4 atoms / 256 channels
- 8 atoms / 512 channels

Modes are nested prefixes. The eval `grouped` implementation batches tokens by
mode and slices both MLP matrices before calling `F.linear`; inactive channels
are never materialized. This is real width skipping using dense GEMMs, not a
mask applied after a full-width multiplication.

## Three benchmark paths

- `dense_full`: ordinary full-width MLP without router overhead. This is the
  fair speed baseline.
- `dense_mask`: evaluates every candidate mode and mixes them. It is the
  differentiable training and numerical-correctness reference.
- `grouped`: retrieves one mode and executes only its selected prefix.

`grouped` and `dense_mask` must agree numerically in hard eval. The meaningful
hardware claim is `grouped` versus `dense_full`, not versus `dense_mask`.

## Train and benchmark

Prepare the character dataset, then train:

```powershell
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char_hardware_atom.py
```

Benchmark batched inference:

```powershell
python experiments/hardware_atom_benchmark.py `
  --out_dir=out-shakespeare-char-hardware-atom-sequence `
  --device=cuda --dtype=float16 --batch_size=16 --block_size=64
```

Benchmark token-at-a-time inference separately:

```powershell
python experiments/hardware_atom_benchmark.py `
  --out_dir=out-shakespeare-char-hardware-atom-sequence `
  --device=cuda --dtype=float16 --batch_size=1 --block_size=1 `
  --output_csv=out-shakespeare-char-hardware-atom/hardware_atom_tok1.csv
```

Use `--synthetic` only for execution smoke tests when prepared data is absent.
Synthetic input does not measure language quality.

Use `--cache_exact_routes` to run the router once on the request prefill and
cache exact GPU request-index buckets for every mode in every layer. Subsequent
steps reuse those buckets, retain each request's own route, and avoid repeated
`nonzero` dispatch. This requires `hardware_atom_route_scope = 'sequence'`.

`--cache_route` is retained as an experimental comparison: it caches one
majority mode per layer for the entire batch. It can be fast, but changes other
requests' routes and therefore may reduce quality.

## GTX 1650 result (CUDA 12.4, FP16)

The 1000-step sequence-routed model reached about 222 active channels out of
512 at its best checkpoint. Measurements on the local GTX 1650 show:

| Scenario | Dense full | Atom path | Relative speed |
|---|---:|---:|---:|
| 16 x 64 tokens, per-sequence dynamic dispatch | 11.742 ms | 14.413 ms | 0.815x |
| 16 x 64 tokens, exact route buckets | 11.386 ms | 9.593 ms | 1.187x |
| 32 x 64 tokens, exact route buckets | 17.122 ms | 13.714 ms | 1.249x |
| 64 x 64 tokens, exact route buckets | 30.026 ms | 22.053 ms | 1.362x |
| 1 x 64-token request, exact cached route | 6.246 ms | 6.769 ms | 0.923x |
| 1-token decode, exact cached route | 5.925 ms | 6.409 ms | 0.925x |

Across 20 validation batches of 16 requests, dynamic and exact-bucket routing
are quality-identical: CE 2.1160, PPL 8.298. Forcing the same batches onto one
majority pattern changes PPL to 8.738. Exact bucketing therefore recovers the
GPU speedup without overwriting any request's chosen subnetwork.

The aligned sliced GEMM is beneficial at useful batch size: exact mixed-route
buckets scale from 1.187x at batch 16 to 1.362x at batch 64 versus the ordinary
full-width MLP. Single-request latency remains launch-bound on this small GPU;
the next systems step is a persistent/fused grouped kernel or a serving queue
that continuously fills route buckets.

## CUDA Graph serving path

`HardwareAtomCUDAGraphRunner` captures a fixed-shape batch after exact routes
have been cached. Request tokens may change on every replay, while batch shape
and route assignments remain fixed. Rebuild the graph when either changes.
The graph uses `forward_inference_fast` to avoid Python-side diagnostic
reductions in the serving hot path.

Run the fair graph benchmark with:

```powershell
python experiments/hardware_atom_cuda_graph_benchmark.py `
  --out_dir=out-shakespeare-char-hardware-atom-sequence `
  --device=cuda --dtype=float16 --batch_size=16 --block_size=64
```

Local GTX 1650 results compare both implementations under the same CUDA Graph
execution model:

| Scenario | Graph dense full | Graph atom | Atom speedup |
|---|---:|---:|---:|
| 1 x 1-token decode | 0.491 ms | 0.445 ms | 1.102x |
| 1 x 64-token request | 1.047 ms | 0.782 ms | 1.338x |
| 16 x 64 tokens | 7.050 ms | 5.017 ms | 1.405x |
| 32 x 64 tokens | 12.532 ms | 8.711 ms | 1.439x |
| 64 x 64 tokens | 24.998 ms | 16.716 ms | 1.495x |

Graph replay also removes most host launch overhead: the single-token atom path
fell from 6.934 ms eager to 0.445 ms graph, and the 16 x 64 path fell from
9.489 ms to 5.017 ms. Every graph result had zero maximum output difference
from its eager reference.

## Direct associative address experiment

`hardware_atom_address_experiment.py` tests a brain-inspired engineering
analogy: familiar inputs use a learned shortcut, while unfamiliar inputs keep
the slower layer-by-layer router. It is not intended as a biological model.
A single 128-to-12 linear projection predicts all four three-way layer choices
from the mean initial embedding state.

```powershell
python experiments/hardware_atom_address_experiment.py `
  --out_dir=out-shakespeare-char-hardware-atom-sequence `
  --train_examples=4096 --val_examples=1024 `
  --hot_k=8 --confidence=0.70
```

On the local checkpoint, only 12 of 81 possible route plans appeared. The top
1, 4, and 8 plans covered 29.2%, 61.8%, and 89.7% of training requests. The
direct head reached 52.4% exact whole-plan accuracy and 86.9%, 72.0%, 86.4%,
and 100% per-layer accuracy. Confidence gating accepted 29.0% of requests as
hot shortcuts; their exact-plan accuracy was 77.1%.

The direct projection took 0.151 ms for a batch of 32, or 0.302 ms including
embedding aggregation. Running the four original router computations on
already available hidden states took 2.844 ms. The hot/confident shortcut plus
dynamic fallback produced PPL 8.481 versus 8.480 for fully dynamic routing,
with mean active MLP ratios of 39.3% and 39.2%, respectively. Predicting every
route reduced retrieval further but changed PPL to 8.580, establishing the
current quality/speed boundary.

## Joint address consolidation and automatic graph dispatch

`finetune_shakespeare_char_joint_address.py` adds the address head to the model
and jointly optimizes language quality, address imitation, and agreement that
pulls the original layer routers toward the early address. A second
address-selected language forward makes functionally equivalent routes valid
even when they do not exactly copy the old router.

After 1000 consolidation steps, the fixed 1024-example evaluation showed:

- the top 4 and 8 plans cover 90.0% and 97.7% of requests;
- at confidence 0.60, 76.6% of requests qualify for the shortcut;
- dynamic PPL is 7.416 and all-direct PPL is 7.474;
- mean active MLP ratio is 37.8% for both paths;
- direct address projection is 0.152 ms versus 2.776 ms for four layer routers.

`DirectAddressCUDAGraphDispatcher` connects the pieces at runtime. It generates
the address, groups hot requests by plan, rounds group capacity to
GPU-friendly tiles, builds CUDA Graphs on first use, and replays cached graphs
thereafter. `hardware_atom_combined_benchmark.py` measures the complete chain.

| Automatic all-hot dispatch | Dynamic layered | Direct + graphs | Speedup |
|---|---:|---:|---:|
| 16 x 64 tokens | 10.868 ms | 7.770 ms | 1.399x |
| 32 x 64 tokens | 15.250 ms | 11.910 ms | 1.280x |
| 64 x 64 tokens | 23.943 ms | 20.757 ms | 1.154x |

Strict synchronous hot/cold splitting is not yet beneficial on this WDDM GPU:
at batch 16 and confidence 0.60 it measured 0.67x because cold requests form a
separate tiny dynamic batch. A production scheduler should retain separate hot
and cold queues across arrivals instead of flushing both pieces of every batch
immediately. The all-hot mode is already end-to-end faster and has a measured
PPL delta of 0.058 on this checkpoint.

## Token-level and per-MLP familiarity

`hardware_atom_token_familiarity.py` moves the address below sequence-level
context. A learned token table plus hashed bigram table emits one mode for every
token and every MLP layer. The normal hidden-state router remains available as
the low-confidence fallback. These addresses select GPU-aligned MLP atoms, not
individual parameters.

The fine-grained training configuration adds a one-atom option, producing
64/128/256/512-channel modes:

```powershell
python train.py config/train_shakespeare_char_hardware_atom_token_fine.py
python experiments/hardware_atom_token_familiarity.py `
  --out_dir=out-shakespeare-char-hardware-atom-token-fine `
  --train_examples=4096 --val_examples=1024 `
  --confidence=0.70
```

At the final checkpoint, 52.6% of dynamically routed tokens selected the
smallest 64-channel mode. Token/bigram familiarity achieved 82.5% exact
four-layer token-plan accuracy. At confidence 0.70, 76.7% of tokens qualified
for the shortcut and 91.2% of those matched the dynamic plan exactly.

| Token/MLP result | Dynamic | Familiarity path |
|---|---:|---:|
| Validation PPL | 8.383 | 8.393 all-direct / 8.381 with fallback |
| Mean active MLP ratio | 39.9% | 40.0% |
| Route lookup | 2.606 ms | 0.202 ms |
| Full-model latency, 32 x 64 | 21.942 ms | 16.867 ms |

The full-model token-direct speedup is 1.301x and the lookup itself is about
12.9x faster. This validates bottom-up familiarity at token and individual MLP
levels while keeping the hardware atom at 64 channels.

## Current hardware boundary

This version validates retrieval, real width skipping, exact request buckets,
and CUDA Graph replay with standard dense FP16/FP32 GEMMs. It does not pretend
that fake-quantized weights are native INT4. Packed INT4/INT8 or an MLP-fused
kernel requires a CUDA/Triton toolchain; the local machine currently provides
CUDA Graph support but neither `nvcc` nor Triton.
