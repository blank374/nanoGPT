"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class FastSlowMLP(nn.Module):
    """
    Logically sparse MLP: a small dense fast path plus a larger routed slow path.

    Training keeps the slow path differentiable with a soft sigmoid gate. Eval can
    switch to hard token routing so the slow branch is only run for selected tokens.
    """

    def __init__(self, config):
        super().__init__()
        fast_hidden = max(1, int(config.dynamic_mlp_fast_ratio * config.n_embd))
        slow_hidden = max(1, int(config.dynamic_mlp_slow_ratio * config.n_embd))
        self.fast_fc = nn.Linear(config.n_embd, fast_hidden, bias=config.bias)
        self.fast_proj = nn.Linear(fast_hidden, config.n_embd, bias=config.bias)
        self.slow_fc = nn.Linear(config.n_embd, slow_hidden, bias=config.bias)
        self.slow_proj = nn.Linear(slow_hidden, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.router_hidden = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.router_features = nn.Linear(3, 1, bias=False)
        self.hard_eval = config.dynamic_mlp_hard_eval
        self.threshold = config.dynamic_mlp_threshold
        self.last_gate = None
        self.last_hard_mask = None

    def _slow_path(self, x):
        x = self.slow_fc(x)
        x = self.gelu(x)
        x = self.slow_proj(x)
        return self.dropout(x)

    def forward(self, x, residual_delta=None):
        fast = self.fast_fc(x)
        fast = self.gelu(fast)
        fast = self.fast_proj(fast)
        fast = self.dropout(fast)

        with torch.no_grad():
            hidden_norm = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            if residual_delta is None:
                delta_norm = torch.zeros_like(hidden_norm)
            else:
                delta_norm = residual_delta.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
            relative_change = delta_norm / hidden_norm.clamp_min(1e-6)
            router_features = torch.cat([hidden_norm, delta_norm, relative_change], dim=-1).to(x.dtype)
        gate_logits = self.router_hidden(x) + self.router_features(router_features)
        gate = torch.sigmoid(gate_logits)

        if self.hard_eval and not self.training:
            hard_mask = gate.squeeze(-1) >= self.threshold
            slow = torch.zeros_like(x)
            if hard_mask.any():
                slow_tokens = self._slow_path(x[hard_mask])
                slow[hard_mask] = slow_tokens
            self.last_hard_mask = hard_mask.detach()
            out = fast + slow
        else:
            slow = self._slow_path(x)
            self.last_hard_mask = None
            out = fast + gate.to(slow.dtype) * slow

        self.last_gate = gate.detach()
        return out

class WidthRouter(nn.Module):

    def __init__(self, config, num_widths):
        super().__init__()
        self.proj = nn.Linear(config.n_embd, num_widths, bias=config.bias)

    def forward(self, x):
        return self.proj(x)

class AdaptiveWidthMLP(nn.Module):
    """
    Nested-width MLP.

    The full 4*d hidden tensor is always materialized in this prototype, but a
    per-token channel mask makes smaller widths true prefixes of larger widths.
    """

    def __init__(self, config):
        super().__init__()
        self.max_hidden = 4 * config.n_embd
        width_choices = sorted(set(max(1, min(self.max_hidden, int(ratio * config.n_embd)))
                                   for ratio in config.dynamic_width_ratios))
        assert width_choices, "dynamic_width_ratios must produce at least one width"
        assert width_choices[-1] == self.max_hidden, \
            f"dynamic_width_ratios must include max width {self.max_hidden}"
        self.width_choices = width_choices
        self.c_fc = nn.Linear(config.n_embd, self.max_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.max_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.router = WidthRouter(config, len(width_choices))
        self.temperature = config.dynamic_width_temperature
        self.hard_eval = config.dynamic_width_hard_eval
        masks = torch.zeros(len(width_choices), self.max_hidden)
        for i, width in enumerate(width_choices):
            masks[i, :width] = 1.0
        costs = torch.tensor([width / self.max_hidden for width in width_choices], dtype=torch.float32)
        widths = torch.tensor(width_choices, dtype=torch.float32)
        self.register_buffer("width_masks", masks)
        self.register_buffer("width_costs", costs)
        self.register_buffer("width_values", widths)
        self.force_width_index = None
        self.last_width_probs = None
        self.last_selected_width_idx = None
        self.last_expected_cost = None
        self.last_effective_width = None
        self.last_router_entropy = None

    def forward(self, x):
        router_logits = self.router(x)
        probs = F.softmax(router_logits / self.temperature, dim=-1)

        if self.force_width_index is not None:
            selected = torch.full(x.shape[:2], int(self.force_width_index), dtype=torch.long, device=x.device)
            mask = self.width_masks[selected].to(x.dtype)
            expected_cost = self.width_costs[selected].to(x.dtype)
            effective_width = self.width_values[selected].to(x.dtype)
        elif self.hard_eval and not self.training:
            selected = probs.argmax(dim=-1)
            mask = self.width_masks[selected].to(x.dtype)
            expected_cost = self.width_costs[selected].to(x.dtype)
            effective_width = self.width_values[selected].to(x.dtype)
        else:
            selected = probs.argmax(dim=-1)
            mask = torch.matmul(probs, self.width_masks.to(probs.dtype)).to(x.dtype)
            expected_cost = torch.matmul(probs, self.width_costs.to(probs.dtype))
            effective_width = torch.matmul(probs, self.width_values.to(probs.dtype)).to(x.dtype)

        hidden = self.c_fc(x)
        hidden = self.gelu(hidden)
        hidden = hidden * mask
        x = self.c_proj(hidden)
        x = self.dropout(x)

        entropy = -(probs.float() * torch.log(probs.float().clamp_min(1e-9))).sum(dim=-1)
        self.last_width_probs = probs
        self.last_selected_width_idx = selected.detach()
        self.last_expected_cost = expected_cost
        self.last_effective_width = effective_width.detach()
        self.last_router_entropy = entropy
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert not (config.dynamic_mlp and config.dynamic_width), "dynamic_mlp and dynamic_width are mutually exclusive"
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.dynamic_mlp = config.dynamic_mlp
        self.dynamic_width = config.dynamic_width
        if config.dynamic_mlp:
            self.mlp = FastSlowMLP(config)
        elif config.dynamic_width:
            self.mlp = AdaptiveWidthMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        attn_delta = self.attn(self.ln_1(x))
        x = x + attn_delta
        if self.dynamic_mlp:
            x = x + self.mlp(self.ln_2(x), residual_delta=attn_delta)
        else:
            x = x + self.mlp(self.ln_2(x))
        return x

class EarlyExitHead(nn.Module):
    """
    Lightweight prediction head for an intermediate layer.

    The projection weight is tied to the main lm_head in GPT.__init__ when
    dynamic_exit=True. That keeps the heads cheap: each exit only adds its own
    LayerNorm parameters instead of a full vocab-sized output matrix.
    """

    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        return self.proj(self.ln(x))

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    dynamic_exit: bool = False
    exit_layers: list = None # 1-indexed transformer layers, e.g. [3, 6, 9]
    confidence_method: str = "max_prob" # "max_prob" or "entropy"
    confidence_threshold: float = 0.95
    entropy_threshold: float = 0.5
    early_exit_loss_weight: float = 0.3
    use_distillation: bool = False
    distillation_temperature: float = 2.0
    distillation_beta: float = 0.5
    dynamic_mlp: bool = False
    dynamic_mlp_fast_ratio: float = 1.0
    dynamic_mlp_slow_ratio: float = 3.0
    dynamic_mlp_cost_weight: float = 0.01
    dynamic_mlp_threshold: float = 0.5
    dynamic_mlp_hard_eval: bool = True
    dynamic_width: bool = False
    dynamic_width_ratios: list = None
    dynamic_width_cost_weight: float = 0.01
    dynamic_width_hard_eval: bool = True
    dynamic_width_temperature: float = 1.0
    dynamic_width_entropy_weight: float = 0.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert not (config.dynamic_mlp and config.dynamic_width), "dynamic_mlp and dynamic_width are mutually exclusive"
        if config.dynamic_width_ratios is None:
            config.dynamic_width_ratios = [0.5, 1.0, 2.0, 4.0]
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.exit_layers = self._normalize_exit_layers(config.exit_layers) if config.dynamic_exit else []
        self.exit_heads = nn.ModuleDict()
        if config.dynamic_exit:
            for layer_idx in self.exit_layers:
                self.exit_heads[str(layer_idx)] = EarlyExitHead(config)
        self.last_exit_stats = None
        self.last_exit_details = None
        self.last_dynamic_mlp_stats = None
        self.last_dynamic_width_stats = None
        self.last_loss_stats = None
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        for head in self.exit_heads.values():
            head.proj.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith(('c_proj.weight', 'fast_proj.weight', 'slow_proj.weight')):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        if config.dynamic_exit:
            print(f"dynamic exit enabled at layers: {self.exit_layers}")
        if config.dynamic_mlp:
            print(
                "dynamic MLP enabled: "
                f"fast={config.dynamic_mlp_fast_ratio:.2f}x, "
                f"slow={config.dynamic_mlp_slow_ratio:.2f}x, "
                f"cost_weight={config.dynamic_mlp_cost_weight}"
            )
        if config.dynamic_width:
            width_choices = sorted(set(max(1, min(4 * config.n_embd, int(ratio * config.n_embd)))
                                       for ratio in config.dynamic_width_ratios))
            print(
                "dynamic width enabled: "
                f"widths={width_choices}, "
                f"cost_weight={config.dynamic_width_cost_weight}"
            )

    def _normalize_exit_layers(self, exit_layers):
        if exit_layers is None:
            exit_layers = [3, 6, 9]
        exit_layers = sorted(set(int(layer) for layer in exit_layers))
        assert all(1 <= layer <= self.config.n_layer for layer in exit_layers), \
            f"exit_layers must be between 1 and n_layer={self.config.n_layer}"
        # The final layer already uses the normal lm_head, so no extra head is needed there.
        return [layer for layer in exit_layers if layer < self.config.n_layer]

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _confidence_from_logits(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        method = self.config.confidence_method
        if method == "max_prob":
            should_exit = max_prob >= self.config.confidence_threshold
            confidence = max_prob
        elif method == "entropy":
            should_exit = entropy <= self.config.entropy_threshold
            confidence = -entropy
        else:
            raise ValueError(f"unknown confidence_method: {method}")
        return should_exit, confidence, max_prob, entropy

    def _set_training_exit_stats(self, aux_losses):
        if aux_losses:
            self.last_exit_stats = {
                "mode": "train",
                "num_exit_heads": len(aux_losses),
                "aux_loss": torch.stack([loss.detach() for loss in aux_losses]).mean().item(),
                "early_exit_loss_weight": self.config.early_exit_loss_weight,
            }
        else:
            self.last_exit_stats = None

    def _dynamic_mlp_gates(self):
        gates = []
        hard_masks = []
        for block in self.transformer.h:
            mlp = block.mlp
            if isinstance(mlp, FastSlowMLP) and mlp.last_gate is not None:
                gates.append(mlp.last_gate)
                if mlp.last_hard_mask is not None:
                    hard_masks.append(mlp.last_hard_mask)
        return gates, hard_masks

    def _set_dynamic_mlp_stats(self, gates, hard_masks=None, valid_mask=None):
        if not gates:
            self.last_dynamic_mlp_stats = None
            return
        gate_tensor = torch.stack(gates)
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(0).unsqueeze(-1)
            gate_values = gate_tensor[valid.expand_as(gate_tensor)]
        else:
            gate_values = gate_tensor.reshape(-1)
        stats = {
            "mean_gate": gate_values.float().mean().item(),
            "slow_soft_fraction": (gate_values.float() >= self.config.dynamic_mlp_threshold).float().mean().item(),
        }
        if hard_masks:
            hard_tensor = torch.stack([mask.float() for mask in hard_masks])
            stats["slow_hard_fraction"] = hard_tensor.mean().item()
        self.last_dynamic_mlp_stats = stats

    def _dynamic_mlp_cost_loss(self, gates, valid_mask):
        if not gates:
            return None
        gate_tensor = torch.stack(gates)
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(0).unsqueeze(-1)
            return gate_tensor[valid.expand_as(gate_tensor)].float().mean()
        return gate_tensor.float().mean()

    def _dynamic_width_modules(self):
        modules = []
        for layer_idx, block in enumerate(self.transformer.h, start=1):
            if isinstance(block.mlp, AdaptiveWidthMLP) and block.mlp.last_width_probs is not None:
                modules.append((layer_idx, block.mlp))
        return modules

    def _dynamic_width_loss_terms(self, modules, valid_mask):
        if not modules:
            return None, None
        costs = torch.stack([mlp.last_expected_cost for _, mlp in modules])
        entropies = torch.stack([mlp.last_router_entropy for _, mlp in modules])
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(0)
            costs = costs[valid.expand_as(costs)]
            entropies = entropies[valid.expand_as(entropies)]
        return costs.float().mean(), entropies.float().mean()

    def _set_dynamic_width_stats(self, modules, valid_mask=None):
        if not modules:
            self.last_dynamic_width_stats = None
            return
        width_values = modules[0][1].width_values.float()
        max_width = float(modules[0][1].max_hidden)
        layer_stats = []
        all_probs = []
        all_selected = []
        all_expected_width = []
        all_entropy = []

        for layer_idx, mlp in modules:
            probs = mlp.last_width_probs.detach().float()
            selected = mlp.last_selected_width_idx.detach()
            entropy = mlp.last_router_entropy.detach().float()
            effective_width = mlp.last_effective_width.detach().float()
            if valid_mask is not None:
                probs_values = probs[valid_mask]
                selected_values = selected[valid_mask]
                entropy_values = entropy[valid_mask]
                effective_width_values = effective_width[valid_mask]
            else:
                probs_values = probs.reshape(-1, probs.size(-1))
                selected_values = selected.reshape(-1)
                entropy_values = entropy.reshape(-1)
                effective_width_values = effective_width.reshape(-1)
            fractions = torch.bincount(selected_values, minlength=len(mlp.width_choices)).float()
            fractions = fractions / fractions.sum().clamp_min(1.0)
            prob_means = probs_values.mean(dim=0)
            layer_stats.append({
                "layer": layer_idx,
                "mean_effective_width": effective_width_values.mean().item(),
                "mean_width_ratio": (effective_width_values.mean() / max_width).item(),
                "router_entropy": entropy_values.mean().item(),
                "width_fractions": {
                    str(width): fractions[i].item()
                    for i, width in enumerate(mlp.width_choices)
                },
                "width_prob_means": {
                    str(width): prob_means[i].item()
                    for i, width in enumerate(mlp.width_choices)
                },
            })
            all_probs.append(probs_values)
            all_selected.append(selected_values)
            all_expected_width.append(effective_width_values)
            all_entropy.append(entropy_values)

        probs_cat = torch.cat(all_probs)
        selected_cat = torch.cat(all_selected)
        expected_width_cat = torch.cat(all_expected_width)
        entropy_cat = torch.cat(all_entropy)
        fractions = torch.bincount(selected_cat, minlength=len(width_values)).float()
        fractions = fractions / fractions.sum().clamp_min(1.0)
        prob_means = probs_cat.mean(dim=0)
        self.last_dynamic_width_stats = {
            "width_choices": [int(v.item()) for v in width_values],
            "max_width": max_width,
            "mean_effective_width": expected_width_cat.mean().item(),
            "mean_width_ratio": (expected_width_cat.mean() / max_width).item(),
            "router_entropy": entropy_cat.mean().item(),
            "width_fractions": {
                str(int(width_values[i].item())): fractions[i].item()
                for i in range(len(width_values))
            },
            "width_prob_means": {
                str(int(width_values[i].item())): prob_means[i].item()
                for i in range(len(width_values))
            },
            "layers": layer_stats,
        }

    def _forward_dynamic_inference(self, idx, pos):
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        B, T, _ = x.size()
        final_logits = None
        active = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        exit_layer = torch.full((B, T), self.config.n_layer, dtype=torch.long, device=idx.device)
        exit_confidence = torch.zeros(B, T, dtype=torch.float32, device=idx.device)
        exit_max_prob = torch.zeros(B, T, dtype=torch.float32, device=idx.device)
        exit_entropy = torch.zeros(B, T, dtype=torch.float32, device=idx.device)

        for layer_idx, block in enumerate(self.transformer.h, start=1):
            x_next = block(x)
            # Tokens that already exited keep their old hidden state; active tokens
            # continue to deepen. This is a research approximation inside dense
            # PyTorch tensors, not a low-level compute-skipping kernel.
            x = torch.where(active.unsqueeze(-1), x_next, x)

            if layer_idx in self.exit_layers:
                logits = self.exit_heads[str(layer_idx)](x)
                should_exit, confidence, max_prob, entropy = self._confidence_from_logits(logits)
                exiting = active & should_exit
                if exiting.any():
                    if final_logits is None:
                        final_logits = torch.empty(B, T, self.config.vocab_size, device=idx.device, dtype=logits.dtype)
                    final_logits[exiting] = logits[exiting]
                    exit_layer[exiting] = layer_idx
                    exit_confidence[exiting] = confidence[exiting].float()
                    exit_max_prob[exiting] = max_prob[exiting].float()
                    exit_entropy[exiting] = entropy[exiting].float()
                    active = active & ~exiting
                if not active.any():
                    break

        if active.any():
            logits = self.lm_head(self.transformer.ln_f(x))
            if final_logits is None:
                final_logits = logits
            else:
                final_logits[active] = logits[active]
            should_exit, confidence, max_prob, entropy = self._confidence_from_logits(logits)
            exit_confidence[active] = confidence[active].float()
            exit_max_prob[active] = max_prob[active].float()
            exit_entropy[active] = entropy[active].float()

        with torch.no_grad():
            flat_exit_layer = exit_layer.view(-1)
            counts = {
                int(layer): int((flat_exit_layer == layer).sum().item())
                for layer in self.exit_layers + [self.config.n_layer]
            }
            self.last_exit_stats = {
                "mode": "eval",
                "counts": counts,
                "mean_exit_layer": exit_layer.float().mean().item(),
                "early_exit_fraction": (exit_layer < self.config.n_layer).float().mean().item(),
                "mean_max_prob": exit_max_prob.mean().item(),
                "mean_entropy": exit_entropy.mean().item(),
            }
            self.last_exit_details = {
                "exit_layer": exit_layer.detach(),
                "confidence": exit_confidence.detach(),
                "max_prob": exit_max_prob.detach(),
                "entropy": exit_entropy.detach(),
            }
            gates, hard_masks = self._dynamic_mlp_gates()
            self._set_dynamic_mlp_stats(gates, hard_masks)
            self._set_dynamic_width_stats(self._dynamic_width_modules())

        return final_logits, None

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        if self.config.dynamic_exit and targets is None and not self.training:
            return self._forward_dynamic_inference(idx, pos)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        exit_outputs = []
        for layer_idx, block in enumerate(self.transformer.h, start=1):
            x = block(x)
            if self.config.dynamic_exit and targets is not None and layer_idx in self.exit_layers:
                exit_logits = self.exit_heads[str(layer_idx)](x)
                exit_outputs.append((layer_idx, exit_logits))
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = task_loss
            aux_losses = []
            if self.config.dynamic_exit and exit_outputs:
                for _, exit_logits in exit_outputs:
                    ce_loss = F.cross_entropy(exit_logits.view(-1, exit_logits.size(-1)), targets.view(-1), ignore_index=-1)
                    if self.config.use_distillation:
                        T = self.config.distillation_temperature
                        per_token_kl = F.kl_div(
                            F.log_softmax(exit_logits / T, dim=-1),
                            F.softmax(logits.detach() / T, dim=-1),
                            reduction='none',
                        ).sum(dim=-1)
                        valid = targets != -1
                        distill_loss = per_token_kl[valid].mean() * (T * T)
                        beta = self.config.distillation_beta
                        aux_losses.append(beta * ce_loss + (1.0 - beta) * distill_loss)
                    else:
                        aux_losses.append(ce_loss)
            if self.config.dynamic_exit and aux_losses:
                loss = loss + self.config.early_exit_loss_weight * torch.stack(aux_losses).mean()
            self._set_training_exit_stats(aux_losses)
            gates, hard_masks = self._dynamic_mlp_gates()
            dynamic_mlp_cost = self._dynamic_mlp_cost_loss(gates, targets != -1)
            if dynamic_mlp_cost is not None:
                loss = loss + self.config.dynamic_mlp_cost_weight * dynamic_mlp_cost
            self._set_dynamic_mlp_stats(gates, hard_masks, valid_mask=targets != -1)
            width_modules = self._dynamic_width_modules()
            dynamic_width_cost, dynamic_width_entropy = self._dynamic_width_loss_terms(width_modules, targets != -1)
            if dynamic_width_cost is not None:
                loss = loss + self.config.dynamic_width_cost_weight * dynamic_width_cost
            if dynamic_width_entropy is not None and self.config.dynamic_width_entropy_weight != 0.0:
                loss = loss - self.config.dynamic_width_entropy_weight * dynamic_width_entropy
            self._set_dynamic_width_stats(width_modules, valid_mask=targets != -1)
            self.last_loss_stats = {
                "task_loss": task_loss.detach().item(),
                "total_loss": loss.detach().item(),
            }
            if dynamic_mlp_cost is not None:
                self.last_loss_stats["dynamic_mlp_cost"] = dynamic_mlp_cost.detach().item()
            if dynamic_width_cost is not None:
                self.last_loss_stats["dynamic_width_cost"] = dynamic_width_cost.detach().item()
            if dynamic_width_entropy is not None:
                self.last_loss_stats["dynamic_width_entropy"] = dynamic_width_entropy.detach().item()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            self.last_exit_stats = None
            self.last_exit_details = None
            self.last_loss_stats = None
            gates, hard_masks = self._dynamic_mlp_gates()
            self._set_dynamic_mlp_stats(gates, hard_masks)
            self._set_dynamic_width_stats(self._dynamic_width_modules())

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, return_exit_stats=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        exit_stats = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            if return_exit_stats and self.config.dynamic_exit and self.last_exit_stats is not None:
                exit_stats.append(self.last_exit_stats)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        if return_exit_stats:
            return idx, exit_stats
        return idx

    @torch.no_grad()
    def generate_dynamic(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Dynamic Fast/Slow Path generation.

        Unlike generate(), this always returns per-generated-token routing stats.
        The normal full-depth path remains the fallback whenever confidence is
        below the configured threshold at all early exits.
        """
        assert self.config.dynamic_exit, "generate_dynamic() requires GPTConfig(dynamic_exit=True)"
        was_training = self.training
        self.eval()
        generation_stats = []

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            last_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(last_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            details = self.last_exit_details
            # Keep stats for the generated token decision, i.e. the final context
            # position whose logits produced idx_next. Batched generation records
            # one entry per batch item per generated step.
            for batch_idx in range(idx.size(0)):
                exit_layer = int(details["exit_layer"][batch_idx, -1].item())
                generation_stats.append({
                    "token_id": int(idx_next[batch_idx, 0].item()),
                    "exit_layer": exit_layer,
                    "confidence": float(details["confidence"][batch_idx, -1].item()),
                    "max_prob": float(details["max_prob"][batch_idx, -1].item()),
                    "entropy": float(details["entropy"][batch_idx, -1].item()),
                    "used_full_path": exit_layer == self.config.n_layer,
                })

            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        if generation_stats:
            avg_layers = sum(s["exit_layer"] for s in generation_stats) / len(generation_stats)
            early_exit_rate = sum(not s["used_full_path"] for s in generation_stats) / len(generation_stats)
        else:
            avg_layers = 0.0
            early_exit_rate = 0.0
        summary = {
            "avg_layers_per_token": avg_layers,
            "layer_saving_ratio": 1.0 - avg_layers / self.config.n_layer if self.config.n_layer else 0.0,
            "early_exit_rate": early_exit_rate,
            "full_path_rate": 1.0 - early_exit_rate,
            "exit_distribution": {
                layer: sum(s["exit_layer"] == layer for s in generation_stats) / len(generation_stats)
                for layer in self.exit_layers + [self.config.n_layer]
            } if generation_stats else {},
        }
        return idx, generation_stats, summary
