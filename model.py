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
        self.sliced_eval = config.dynamic_width_sliced_eval
        self.routing = config.dynamic_width_routing
        self.force_hard = False
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

    def _forward_dense_mask(self, x, mask):
        hidden = self.c_fc(x)
        hidden = self.gelu(hidden)
        hidden = hidden * mask
        x = self.c_proj(hidden)
        return self.dropout(x)

    def _forward_sliced(self, x, selected):
        B, T, C = x.size()
        x_flat = x.reshape(B * T, C)
        selected_flat = selected.reshape(B * T)

        if x_flat.size(0) == 1:
            width = self.width_choices[int(selected_flat.item())]
            fc_bias = self.c_fc.bias[:width] if self.c_fc.bias is not None else None
            hidden = F.linear(x_flat, self.c_fc.weight[:width], fc_bias)
            hidden = self.gelu(hidden)
            out = F.linear(hidden, self.c_proj.weight[:, :width], self.c_proj.bias)
            return self.dropout(out.view(B, T, C))

        out_flat = torch.zeros_like(x_flat)

        for width_idx, width in enumerate(self.width_choices):
            token_idx = (selected_flat == width_idx).nonzero(as_tuple=False).flatten()
            if token_idx.numel() == 0:
                continue
            fc_bias = self.c_fc.bias[:width] if self.c_fc.bias is not None else None
            hidden = F.linear(
                x_flat.index_select(0, token_idx),
                self.c_fc.weight[:width],
                fc_bias,
            )
            hidden = self.gelu(hidden)
            out = F.linear(hidden, self.c_proj.weight[:, :width], None)
            out_flat.index_add_(0, token_idx, out)

        if self.c_proj.bias is not None:
            out_flat = out_flat + self.c_proj.bias
        return self.dropout(out_flat.view(B, T, C))

    def forward(self, x):
        router_logits = self.router(x)
        probs = F.softmax(router_logits / self.temperature, dim=-1)

        use_hard = self.force_width_index is not None or self.force_hard or (self.hard_eval and not self.training)
        if self.force_width_index is not None:
            selected = torch.full(x.shape[:2], int(self.force_width_index), dtype=torch.long, device=x.device)
            mask = self.width_masks[selected].to(x.dtype)
            expected_cost = self.width_costs[selected].to(x.dtype)
            effective_width = self.width_values[selected].to(x.dtype)
        elif use_hard:
            selected = probs.argmax(dim=-1)
            mask = self.width_masks[selected].to(x.dtype)
            expected_cost = self.width_costs[selected].to(x.dtype)
            effective_width = self.width_values[selected].to(x.dtype)
        else:
            selected = probs.argmax(dim=-1)
            soft_mask = torch.matmul(probs, self.width_masks.to(probs.dtype))
            if self.training and self.routing == "ste":
                hard_mask = self.width_masks[selected].to(probs.dtype)
                mask = hard_mask + soft_mask - soft_mask.detach()
            else:
                mask = soft_mask
            mask = mask.to(x.dtype)
            expected_cost = torch.matmul(probs, self.width_costs.to(probs.dtype))
            effective_width = torch.matmul(probs, self.width_values.to(probs.dtype)).to(x.dtype)

        entropy = -(probs.float() * torch.log(probs.float().clamp_min(1e-9))).sum(dim=-1)
        self.last_width_probs = probs
        self.last_selected_width_idx = selected.detach()
        self.last_expected_cost = expected_cost
        self.last_effective_width = effective_width.detach()
        self.last_router_entropy = entropy
        if use_hard and not self.training and self.sliced_eval:
            return self._forward_sliced(x, selected)
        x = self._forward_dense_mask(x, mask)
        return x

class FreeChannelMLP(nn.Module):
    """
    Per-token, per-channel gated MLP.

    All hidden channels are structurally symmetric: the router emits one sigmoid
    decision per channel instead of choosing among fixed width buckets.
    """

    def __init__(self, config):
        super().__init__()
        self.max_hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, self.max_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.max_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gate_network = nn.Linear(config.n_embd, self.max_hidden, bias=config.bias)
        self.temperature = config.free_channel_temperature
        self.routing = config.free_channel_routing
        self.threshold = config.free_channel_threshold
        self.eval_impl = config.free_channel_eval_impl
        self.prefix_granularity = config.free_channel_prefix_granularity
        self.last_gate_prob = None
        self.last_gate = None
        self.last_active_channels = None
        self.last_gate_entropy = None

    def _forward_dense_mask(self, x, gate):
        hidden = self.c_fc(x)
        hidden = self.gelu(hidden)
        hidden = hidden * gate.to(hidden.dtype)
        x = self.c_proj(hidden)
        return self.dropout(x)

    def _forward_prefix_sliced(self, x, hard_gate, active_channels):
        B, T, C = x.size()
        x_flat = x.reshape(B * T, C)
        gate_flat = hard_gate.reshape(B * T, self.max_hidden).to(x.dtype)
        if self.eval_impl == "prefix_cover_sliced":
            ranks = torch.arange(1, self.max_hidden + 1, device=x.device, dtype=torch.long)
            width = (hard_gate.to(torch.long) * ranks).amax(dim=-1)
            width_flat = width.reshape(B * T)
        else:
            width_flat = active_channels.reshape(B * T).to(torch.long)
        if self.prefix_granularity > 1:
            width_flat = (
                (width_flat + self.prefix_granularity - 1)
                // self.prefix_granularity
                * self.prefix_granularity
            )
            width_flat = width_flat.clamp(max=self.max_hidden)
        self.last_active_channels = width_flat.view(B, T).float()

        if x_flat.size(0) == 1:
            width = int(width_flat.item())
            if width == 0:
                out = x_flat.new_zeros(x_flat.shape)
                if self.c_proj.bias is not None:
                    out = out + self.c_proj.bias
                return self.dropout(out.view(B, T, C))
            fc_bias = self.c_fc.bias[:width] if self.c_fc.bias is not None else None
            hidden = F.linear(x_flat, self.c_fc.weight[:width], fc_bias)
            hidden = self.gelu(hidden)
            hidden = hidden * gate_flat[:, :width]
            out = F.linear(hidden, self.c_proj.weight[:, :width], self.c_proj.bias)
            return self.dropout(out.view(B, T, C))

        out_flat = torch.zeros_like(x_flat)
        for width in torch.unique(width_flat).tolist():
            width = int(width)
            token_idx = (width_flat == width).nonzero(as_tuple=False).flatten()
            if width == 0:
                continue
            fc_bias = self.c_fc.bias[:width] if self.c_fc.bias is not None else None
            hidden = F.linear(
                x_flat.index_select(0, token_idx),
                self.c_fc.weight[:width],
                fc_bias,
            )
            hidden = self.gelu(hidden)
            hidden = hidden * gate_flat.index_select(0, token_idx)[:, :width]
            out = F.linear(hidden, self.c_proj.weight[:, :width], None)
            out_flat.index_add_(0, token_idx, out)

        if self.c_proj.bias is not None:
            out_flat = out_flat + self.c_proj.bias
        return self.dropout(out_flat.view(B, T, C))

    def forward(self, x):
        gate_logits = self.gate_network(x)
        gate_prob = torch.sigmoid(gate_logits / self.temperature)
        hard_gate = (gate_prob > self.threshold).to(gate_prob.dtype)

        if self.routing == "soft":
            gate = gate_prob
        elif self.training:
            gate = hard_gate + gate_prob - gate_prob.detach()
        else:
            gate = hard_gate

        gate_prob_f = gate_prob.float()
        active_channels = gate.detach().float().sum(dim=-1)
        entropy = -(
            gate_prob_f * torch.log(gate_prob_f.clamp_min(1e-9))
            + (1.0 - gate_prob_f) * torch.log((1.0 - gate_prob_f).clamp_min(1e-9))
        ).sum(dim=-1)
        self.last_gate_prob = gate_prob
        self.last_gate = gate
        self.last_active_channels = active_channels
        self.last_gate_entropy = entropy
        if not self.training and self.eval_impl in ("prefix_sliced", "prefix_cover_sliced") and self.routing == "ste":
            return self._forward_prefix_sliced(x, hard_gate, active_channels)
        x = self._forward_dense_mask(x, gate)
        return x

class BlockSparseMLP(nn.Module):
    """
    Block-wise gated MLP with an eval-time sliced Linear path.

    Training uses the same dense-mask approximation as FreeChannelMLP. During
    eval, sliced_eval=True skips inactive hidden-channel blocks before c_fc and
    c_proj, so it is a real compute-skipping prototype rather than mask-only.
    """

    def __init__(self, config):
        super().__init__()
        self.max_hidden = 4 * config.n_embd
        self.block_size = config.block_sparse_block_size
        assert self.max_hidden % self.block_size == 0, \
            f"max hidden {self.max_hidden} must be divisible by block size {self.block_size}"
        self.num_blocks = self.max_hidden // self.block_size
        self.c_fc = nn.Linear(config.n_embd, self.max_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.max_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gate_network = nn.Linear(config.n_embd, self.num_blocks, bias=config.bias)
        self.temperature = config.block_sparse_temperature
        self.routing = config.block_sparse_routing
        self.threshold = config.block_sparse_threshold
        self.sliced_eval = config.block_sparse_sliced_eval
        self.eval_impl = config.block_sparse_eval_impl
        self.last_gate_prob = None
        self.last_gate = None
        self.last_active_channels = None
        self.last_gate_entropy = None

    def _channel_gate(self, block_gate):
        return block_gate.repeat_interleave(self.block_size, dim=-1)

    def _record_stats(self, gate_prob, gate):
        gate_prob_f = gate_prob.float()
        entropy = -(
            gate_prob_f * torch.log(gate_prob_f.clamp_min(1e-9))
            + (1.0 - gate_prob_f) * torch.log((1.0 - gate_prob_f).clamp_min(1e-9))
        ).sum(dim=-1) * self.block_size
        self.last_gate_prob = self._channel_gate(gate_prob)
        self.last_gate = self._channel_gate(gate)
        self.last_active_channels = gate.detach().float().sum(dim=-1) * self.block_size
        self.last_gate_entropy = entropy

    def _forward_dense_mask(self, x, gate):
        hidden = self.c_fc(x)
        hidden = self.gelu(hidden)
        hidden = hidden * self._channel_gate(gate).to(hidden.dtype)
        x = self.c_proj(hidden)
        return self.dropout(x)

    def _forward_sliced(self, x, hard_gate):
        B, T, C = x.size()
        x_flat = x.reshape(B * T, C)
        hard_flat = hard_gate.reshape(B * T, self.num_blocks).bool()
        out_flat = torch.zeros_like(x_flat)

        for block_idx in range(self.num_blocks):
            token_mask = hard_flat[:, block_idx]
            if not token_mask.any():
                continue
            start = block_idx * self.block_size
            end = start + self.block_size
            fc_bias = self.c_fc.bias[start:end] if self.c_fc.bias is not None else None
            hidden = F.linear(x_flat[token_mask], self.c_fc.weight[start:end], fc_bias)
            hidden = self.gelu(hidden)
            out_flat[token_mask] += F.linear(hidden, self.c_proj.weight[:, start:end], None)

        if self.c_proj.bias is not None:
            out_flat = out_flat + self.c_proj.bias
        return self.dropout(out_flat.view(B, T, C))

    def _forward_grouped_sliced(self, x, hard_gate):
        B, T, C = x.size()
        x_flat = x.reshape(B * T, C)
        hard_flat = hard_gate.reshape(B * T, self.num_blocks).bool()
        out_flat = torch.zeros_like(x_flat)

        patterns, inverse = torch.unique(hard_flat, dim=0, return_inverse=True)
        channel_template = torch.arange(
            self.block_size, device=x.device, dtype=torch.long
        ).unsqueeze(0)
        for pattern_idx, pattern in enumerate(patterns):
            active_blocks = pattern.nonzero(as_tuple=False).flatten()
            if active_blocks.numel() == 0:
                continue
            token_idx = (inverse == pattern_idx).nonzero(as_tuple=False).flatten()
            channels = (
                active_blocks.unsqueeze(1) * self.block_size + channel_template
            ).reshape(-1)
            fc_bias = self.c_fc.bias.index_select(0, channels) if self.c_fc.bias is not None else None
            hidden = F.linear(
                x_flat.index_select(0, token_idx),
                self.c_fc.weight.index_select(0, channels),
                fc_bias,
            )
            hidden = self.gelu(hidden)
            out = F.linear(hidden, self.c_proj.weight.index_select(1, channels), None)
            out_flat.index_add_(0, token_idx, out)

        if self.c_proj.bias is not None:
            out_flat = out_flat + self.c_proj.bias
        return self.dropout(out_flat.view(B, T, C))

    def forward(self, x):
        gate_logits = self.gate_network(x)
        gate_prob = torch.sigmoid(gate_logits / self.temperature)
        hard_gate = (gate_prob > self.threshold).to(gate_prob.dtype)

        if self.routing == "soft":
            gate = gate_prob
        elif self.training:
            gate = hard_gate + gate_prob - gate_prob.detach()
        else:
            gate = hard_gate

        self._record_stats(gate_prob, gate)
        if not self.training and self.sliced_eval and self.routing == "ste":
            if self.eval_impl == "grouped":
                return self._forward_grouped_sliced(x, hard_gate)
            return self._forward_sliced(x, hard_gate)
        return self._forward_dense_mask(x, gate)

class BlockPrecisionMLP(nn.Module):
    """
    Hardware-aligned Width x Bit MLP prototype.

    The controller emits a continuous prefix-block width demand plus per-block
    precision logits. Forward uses hard block/bit choices with STE during
    training, while the resource term tracks MLP weight bits fetched per token.
    """

    def __init__(self, config):
        super().__init__()
        self.max_hidden = 4 * config.n_embd
        self.block_size = config.block_precision_block_size
        assert self.max_hidden % self.block_size == 0, \
            f"max hidden {self.max_hidden} must be divisible by block size {self.block_size}"
        self.num_blocks = self.max_hidden // self.block_size
        bit_choices = [int(bit) for bit in config.block_precision_bit_choices]
        assert bit_choices, "block_precision_bit_choices must not be empty"
        assert all(bit >= 2 for bit in bit_choices), "fake quantization needs bit choices >= 2"
        self.bit_choices = bit_choices
        self.c_fc = nn.Linear(config.n_embd, self.max_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.max_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.width_network = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.bit_network = nn.Linear(config.n_embd, self.num_blocks * len(bit_choices), bias=config.bias)
        self.temperature = config.block_precision_temperature
        self.width_temperature = config.block_precision_width_temperature
        self.routing = config.block_precision_routing
        self.sliced_eval = config.block_precision_sliced_eval
        self.eval_impl = config.block_precision_eval_impl
        self.register_buffer("bit_values", torch.tensor(bit_choices, dtype=torch.float32))
        self.force_blocks = None
        self.force_bit = None
        self.last_width_demand = None
        self.last_hard_blocks = None
        self.last_block_gate = None
        self.last_bit_probs = None
        self.last_bit_onehot = None
        self.last_selected_bits = None
        self.last_weight_bits_per_token = None
        self.last_bit_entropy = None

    def _channel_gate(self, block_gate):
        return block_gate.repeat_interleave(self.block_size, dim=-1)

    def _fake_quant_weight(self, weight, bits):
        if bits >= 16:
            return weight
        qmax = (1 << (bits - 1)) - 1
        scale = weight.detach().abs().amax().clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
        quantized = q * scale
        return weight + (quantized - weight).detach()

    def _route(self, x):
        B, T, _ = x.size()
        width_raw = self.width_network(x).squeeze(-1)
        width_demand = torch.sigmoid(width_raw) * self.num_blocks
        if self.force_blocks is not None:
            blocks = max(1, min(int(self.force_blocks), self.num_blocks))
            width_demand = torch.full_like(width_demand, float(blocks))
        block_ids = torch.arange(self.num_blocks, device=x.device, dtype=width_demand.dtype)
        soft_gate = torch.sigmoid((width_demand.unsqueeze(-1) - block_ids) / self.width_temperature)
        hard_blocks = torch.ceil(width_demand).clamp(1, self.num_blocks).to(torch.long)
        hard_gate = (block_ids.unsqueeze(0).unsqueeze(0) < hard_blocks.unsqueeze(-1)).to(soft_gate.dtype)

        bit_logits = self.bit_network(x).view(B, T, self.num_blocks, len(self.bit_choices))
        bit_probs = F.softmax(bit_logits / self.temperature, dim=-1)
        if self.force_bit is not None:
            bit_idx = self.bit_choices.index(int(self.force_bit))
            bit_probs = F.one_hot(
                torch.full((B, T, self.num_blocks), bit_idx, dtype=torch.long, device=x.device),
                num_classes=len(self.bit_choices),
            ).to(bit_logits.dtype)
        selected = bit_probs.argmax(dim=-1)
        hard_bit = F.one_hot(selected, num_classes=len(self.bit_choices)).to(bit_probs.dtype)

        if self.routing == "soft":
            block_gate = soft_gate
            bit_onehot = bit_probs
        elif self.training:
            block_gate = hard_gate + soft_gate - soft_gate.detach()
            bit_onehot = hard_bit + bit_probs - bit_probs.detach()
        else:
            block_gate = hard_gate
            bit_onehot = hard_bit

        entropy = -(bit_probs.float() * torch.log(bit_probs.float().clamp_min(1e-9))).sum(dim=-1)
        bits = torch.matmul(bit_onehot, self.bit_values.to(bit_onehot.dtype))
        weight_bits = (
            block_gate.float()
            * bits.float()
            * self.block_size
            * (self.c_fc.in_features + self.c_proj.out_features)
        ).sum(dim=-1)

        self.last_width_demand = width_demand
        self.last_hard_blocks = hard_blocks.detach()
        self.last_block_gate = block_gate
        self.last_bit_probs = bit_probs
        self.last_bit_onehot = bit_onehot
        self.last_selected_bits = self.bit_values.to(x.device)[selected].detach()
        self.last_weight_bits_per_token = weight_bits
        self.last_bit_entropy = entropy
        return block_gate, bit_onehot

    def _forward_dense_mask(self, x, block_gate, bit_onehot):
        out = torch.zeros_like(x)
        for bit_idx, bits in enumerate(self.bit_choices):
            bit_block_gate = block_gate * bit_onehot[..., bit_idx]
            channel_gate = self._channel_gate(bit_block_gate).to(x.dtype)
            fc_weight = self._fake_quant_weight(self.c_fc.weight, bits)
            proj_weight = self._fake_quant_weight(self.c_proj.weight, bits)
            hidden = F.linear(x, fc_weight, self.c_fc.bias)
            hidden = self.gelu(hidden)
            hidden = hidden * channel_gate
            out = out + F.linear(hidden, proj_weight, None)
        if self.c_proj.bias is not None:
            out = out + self.c_proj.bias
        return self.dropout(out)

    def _forward_grouped_sliced(self, x, block_gate, bit_onehot):
        hard_gate = (block_gate > 0.5)
        selected_bit_idx = bit_onehot.argmax(dim=-1)
        B, T, C = x.size()
        x_flat = x.reshape(B * T, C)
        gate_flat = hard_gate.reshape(B * T, self.num_blocks)
        bit_flat = selected_bit_idx.reshape(B * T, self.num_blocks)
        signature = torch.where(gate_flat, bit_flat + 1, torch.zeros_like(bit_flat))
        patterns, inverse = torch.unique(signature, dim=0, return_inverse=True)
        out_flat = torch.zeros_like(x_flat)
        channel_template = torch.arange(self.block_size, device=x.device, dtype=torch.long).unsqueeze(0)

        for pattern_idx, pattern in enumerate(patterns):
            token_idx = (inverse == pattern_idx).nonzero(as_tuple=False).flatten()
            for bit_idx, bits in enumerate(self.bit_choices):
                active_blocks = (pattern == bit_idx + 1).nonzero(as_tuple=False).flatten()
                if active_blocks.numel() == 0:
                    continue
                channels = (
                    active_blocks.unsqueeze(1) * self.block_size + channel_template
                ).reshape(-1)
                fc_weight = self._fake_quant_weight(self.c_fc.weight, bits).index_select(0, channels)
                proj_weight = self._fake_quant_weight(self.c_proj.weight, bits).index_select(1, channels)
                fc_bias = self.c_fc.bias.index_select(0, channels) if self.c_fc.bias is not None else None
                hidden = F.linear(x_flat.index_select(0, token_idx), fc_weight, fc_bias)
                hidden = self.gelu(hidden)
                out = F.linear(hidden, proj_weight, None)
                out_flat.index_add_(0, token_idx, out)

        if self.c_proj.bias is not None:
            out_flat = out_flat + self.c_proj.bias
        return self.dropout(out_flat.view(B, T, C))

    def forward(self, x):
        block_gate, bit_onehot = self._route(x)
        if not self.training and self.sliced_eval and self.routing == "ste" and self.eval_impl == "grouped":
            return self._forward_grouped_sliced(x, block_gate, bit_onehot)
        return self._forward_dense_mask(x, block_gate, bit_onehot)

class ResourceModeMLP(nn.Module):
    """
    Batch-friendly Width x Bit MLP.

    Each token selects one of a small number of executable resource modes such
    as (4 blocks, 2 bit) or (16 blocks, 8 bit). This keeps the model dynamic
    while limiting runtime shapes to a few batchable grouped matmuls.
    """

    def __init__(self, config):
        super().__init__()
        self.max_hidden = 4 * config.n_embd
        self.block_size = config.resource_mode_block_size
        assert self.max_hidden % self.block_size == 0, \
            f"max hidden {self.max_hidden} must be divisible by block size {self.block_size}"
        self.num_blocks = self.max_hidden // self.block_size
        modes = [(int(blocks), int(bits)) for blocks, bits in config.resource_mode_choices]
        assert modes, "resource_mode_choices must not be empty"
        for blocks, bits in modes:
            assert 1 <= blocks <= self.num_blocks, f"mode blocks must be in [1, {self.num_blocks}]"
            assert bits >= 2, "fake quantization needs mode bit choices >= 2"
        self.modes = modes
        self.c_fc = nn.Linear(config.n_embd, self.max_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.max_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.router = nn.Linear(config.n_embd, len(modes), bias=config.bias)
        self.temperature = config.resource_mode_temperature
        self.routing = config.resource_mode_routing
        self.sliced_eval = config.resource_mode_sliced_eval
        self.eval_impl = config.resource_mode_eval_impl
        mode_blocks = torch.tensor([blocks for blocks, _ in modes], dtype=torch.float32)
        mode_bits = torch.tensor([bits for _, bits in modes], dtype=torch.float32)
        mode_costs = mode_blocks * self.block_size * mode_bits * (config.n_embd + config.n_embd)
        self.register_buffer("mode_blocks", mode_blocks)
        self.register_buffer("mode_bits", mode_bits)
        self.register_buffer("mode_costs", mode_costs)
        self.force_mode_index = None
        self.last_mode_probs = None
        self.last_mode_onehot = None
        self.last_selected_mode = None
        self.last_weight_bits_per_token = None
        self.last_router_entropy = None

    def _fake_quant_weight(self, weight, bits):
        if bits >= 16:
            return weight
        qmax = (1 << (bits - 1)) - 1
        scale = weight.detach().abs().amax().clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
        quantized = q * scale
        return weight + (quantized - weight).detach()

    def _route(self, x):
        logits = self.router(x)
        probs = F.softmax(logits / self.temperature, dim=-1)
        if self.force_mode_index is not None:
            selected = torch.full(x.shape[:2], int(self.force_mode_index), dtype=torch.long, device=x.device)
            hard = F.one_hot(selected, num_classes=len(self.modes)).to(probs.dtype)
            probs = hard
        else:
            selected = probs.argmax(dim=-1)
            hard = F.one_hot(selected, num_classes=len(self.modes)).to(probs.dtype)

        if self.routing == "soft":
            mode_onehot = probs
        elif self.training:
            mode_onehot = hard + probs - probs.detach()
        else:
            mode_onehot = hard

        costs = torch.matmul(mode_onehot.float(), self.mode_costs.to(mode_onehot.device))
        entropy = -(probs.float() * torch.log(probs.float().clamp_min(1e-9))).sum(dim=-1)
        self.last_mode_probs = probs
        self.last_mode_onehot = mode_onehot
        self.last_selected_mode = selected.detach()
        self.last_weight_bits_per_token = costs
        self.last_router_entropy = entropy
        return mode_onehot

    def _forward_dense_mask(self, x, mode_onehot):
        out = torch.zeros_like(x)
        for mode_idx, (blocks, bits) in enumerate(self.modes):
            width = blocks * self.block_size
            fc_bias = self.c_fc.bias[:width] if self.c_fc.bias is not None else None
            fc_weight = self._fake_quant_weight(self.c_fc.weight[:width], bits)
            proj_weight = self._fake_quant_weight(self.c_proj.weight[:, :width], bits)
            hidden = F.linear(x, fc_weight, fc_bias)
            hidden = self.gelu(hidden)
            mode_out = F.linear(hidden, proj_weight, None)
            out = out + mode_onehot[..., mode_idx].unsqueeze(-1).to(mode_out.dtype) * mode_out
        if self.c_proj.bias is not None:
            out = out + self.c_proj.bias
        return self.dropout(out)

    def _forward_grouped_sliced(self, x, selected):
        B, T, C = x.size()
        x_flat = x.reshape(B * T, C)
        selected_flat = selected.reshape(B * T)
        out_flat = torch.zeros_like(x_flat)
        for mode_idx, (blocks, bits) in enumerate(self.modes):
            token_idx = (selected_flat == mode_idx).nonzero(as_tuple=False).flatten()
            if token_idx.numel() == 0:
                continue
            width = blocks * self.block_size
            fc_bias = self.c_fc.bias[:width] if self.c_fc.bias is not None else None
            fc_weight = self._fake_quant_weight(self.c_fc.weight[:width], bits)
            proj_weight = self._fake_quant_weight(self.c_proj.weight[:, :width], bits)
            hidden = F.linear(x_flat.index_select(0, token_idx), fc_weight, fc_bias)
            hidden = self.gelu(hidden)
            out = F.linear(hidden, proj_weight, None)
            out_flat.index_add_(0, token_idx, out)
        if self.c_proj.bias is not None:
            out_flat = out_flat + self.c_proj.bias
        return self.dropout(out_flat.view(B, T, C))

    def forward(self, x):
        mode_onehot = self._route(x)
        if not self.training and self.sliced_eval and self.routing == "ste" and self.eval_impl == "grouped":
            return self._forward_grouped_sliced(x, self.last_selected_mode)
        return self._forward_dense_mask(x, mode_onehot)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert sum([config.dynamic_mlp, config.dynamic_width, config.free_channel_mlp, config.block_sparse_mlp, config.block_precision_mlp, config.resource_mode_mlp]) <= 1, \
            "dynamic_mlp, dynamic_width, free_channel_mlp, block_sparse_mlp, block_precision_mlp, and resource_mode_mlp are mutually exclusive"
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.dynamic_mlp = config.dynamic_mlp
        self.dynamic_width = config.dynamic_width
        self.free_channel_mlp = config.free_channel_mlp
        self.block_sparse_mlp = config.block_sparse_mlp
        self.block_precision_mlp = config.block_precision_mlp
        self.resource_mode_mlp = config.resource_mode_mlp
        if config.dynamic_mlp:
            self.mlp = FastSlowMLP(config)
        elif config.dynamic_width:
            self.mlp = AdaptiveWidthMLP(config)
        elif config.free_channel_mlp:
            self.mlp = FreeChannelMLP(config)
        elif config.block_sparse_mlp:
            self.mlp = BlockSparseMLP(config)
        elif config.block_precision_mlp:
            self.mlp = BlockPrecisionMLP(config)
        elif config.resource_mode_mlp:
            self.mlp = ResourceModeMLP(config)
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
    dynamic_width_temperature_final: float = 1.0
    dynamic_width_temperature_anneal_iters: int = 0
    dynamic_width_routing: str = "soft" # "soft" or "ste"
    dynamic_width_hard_loss_weight: float = 0.0
    dynamic_width_entropy_weight: float = 0.0
    dynamic_width_sliced_eval: bool = True
    free_channel_mlp: bool = False
    free_channel_routing: str = "soft" # "soft" or "ste"
    free_channel_threshold: float = 0.5
    free_channel_target_ratio: float = 0.4
    free_channel_budget_weight: float = 0.01
    free_channel_cost_weight: float = 0.0
    free_channel_temperature: float = 2.0
    free_channel_temperature_final: float = 0.5
    free_channel_temperature_anneal_iters: int = 1000
    free_channel_eval_impl: str = "dense_mask" # "dense_mask", "prefix_sliced", or "prefix_cover_sliced"
    free_channel_prefix_granularity: int = 64
    block_sparse_mlp: bool = False
    block_sparse_block_size: int = 16
    block_sparse_routing: str = "ste" # "soft" or "ste"
    block_sparse_threshold: float = 0.5
    block_sparse_target_ratio: float = 0.4
    block_sparse_budget_weight: float = 0.01
    block_sparse_cost_weight: float = 0.0
    block_sparse_temperature: float = 2.0
    block_sparse_temperature_final: float = 0.5
    block_sparse_temperature_anneal_iters: int = 1000
    block_sparse_sliced_eval: bool = True
    block_sparse_eval_impl: str = "grouped" # "block" or "grouped"
    block_precision_mlp: bool = False
    block_precision_block_size: int = 16
    block_precision_bit_choices: list = None
    block_precision_routing: str = "ste" # "soft" or "ste"
    block_precision_cost_weight: float = 0.01
    block_precision_temperature: float = 1.0
    block_precision_temperature_final: float = 1.0
    block_precision_temperature_anneal_iters: int = 0
    block_precision_width_temperature: float = 1.0
    block_precision_sliced_eval: bool = True
    block_precision_eval_impl: str = "grouped" # "dense_mask" or "grouped"
    resource_mode_mlp: bool = False
    resource_mode_block_size: int = 16
    resource_mode_choices: list = None
    resource_mode_routing: str = "ste" # "soft" or "ste"
    resource_mode_cost_weight: float = 0.01
    resource_mode_temperature: float = 1.0
    resource_mode_temperature_final: float = 1.0
    resource_mode_temperature_anneal_iters: int = 0
    resource_mode_sliced_eval: bool = True
    resource_mode_eval_impl: str = "grouped" # "dense_mask" or "grouped"

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert sum([config.dynamic_mlp, config.dynamic_width, config.free_channel_mlp, config.block_sparse_mlp, config.block_precision_mlp, config.resource_mode_mlp]) <= 1, \
            "dynamic_mlp, dynamic_width, free_channel_mlp, block_sparse_mlp, block_precision_mlp, and resource_mode_mlp are mutually exclusive"
        if config.dynamic_width_ratios is None:
            config.dynamic_width_ratios = [0.5, 1.0, 2.0, 4.0]
        if config.block_precision_bit_choices is None:
            config.block_precision_bit_choices = [2, 4, 8, 16]
        if config.resource_mode_choices is None:
            config.resource_mode_choices = [(4, 2), (8, 4), (12, 4), (16, 8), (32, 16)]
        assert config.dynamic_width_routing in ("soft", "ste"), "dynamic_width_routing must be 'soft' or 'ste'"
        assert config.free_channel_routing in ("soft", "ste"), "free_channel_routing must be 'soft' or 'ste'"
        assert config.free_channel_eval_impl in ("dense_mask", "prefix_sliced", "prefix_cover_sliced"), \
            "free_channel_eval_impl must be 'dense_mask', 'prefix_sliced', or 'prefix_cover_sliced'"
        assert config.block_sparse_routing in ("soft", "ste"), "block_sparse_routing must be 'soft' or 'ste'"
        assert config.block_sparse_eval_impl in ("block", "grouped"), "block_sparse_eval_impl must be 'block' or 'grouped'"
        assert config.block_precision_routing in ("soft", "ste"), "block_precision_routing must be 'soft' or 'ste'"
        assert config.block_precision_eval_impl in ("dense_mask", "grouped"), \
            "block_precision_eval_impl must be 'dense_mask' or 'grouped'"
        assert config.resource_mode_routing in ("soft", "ste"), "resource_mode_routing must be 'soft' or 'ste'"
        assert config.resource_mode_eval_impl in ("dense_mask", "grouped"), \
            "resource_mode_eval_impl must be 'dense_mask' or 'grouped'"
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
        self.last_free_channel_stats = None
        self.last_block_precision_stats = None
        self.last_resource_mode_stats = None
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
        if config.free_channel_mlp:
            print(
                "free-channel MLP enabled: "
                f"max_hidden={4 * config.n_embd}, "
                f"routing={config.free_channel_routing}, "
                f"target_ratio={config.free_channel_target_ratio}, "
                f"budget_weight={config.free_channel_budget_weight}"
            )
        if config.block_sparse_mlp:
            print(
                "block-sparse MLP enabled: "
                f"max_hidden={4 * config.n_embd}, "
                f"block_size={config.block_sparse_block_size}, "
                f"routing={config.block_sparse_routing}, "
                f"target_ratio={config.block_sparse_target_ratio}, "
                f"budget_weight={config.block_sparse_budget_weight}, "
                f"sliced_eval={config.block_sparse_sliced_eval}"
            )
        if config.block_precision_mlp:
            print(
                "block-precision MLP enabled: "
                f"max_hidden={4 * config.n_embd}, "
                f"block_size={config.block_precision_block_size}, "
                f"bits={config.block_precision_bit_choices}, "
                f"routing={config.block_precision_routing}, "
                f"cost_weight={config.block_precision_cost_weight}, "
                f"sliced_eval={config.block_precision_sliced_eval}"
            )
        if config.resource_mode_mlp:
            print(
                "resource-mode MLP enabled: "
                f"max_hidden={4 * config.n_embd}, "
                f"block_size={config.resource_mode_block_size}, "
                f"modes={config.resource_mode_choices}, "
                f"routing={config.resource_mode_routing}, "
                f"cost_weight={config.resource_mode_cost_weight}, "
                f"sliced_eval={config.resource_mode_sliced_eval}"
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

    def set_dynamic_width_temperature(self, temperature):
        if not self.config.dynamic_width:
            return
        self.config.dynamic_width_temperature = temperature
        for block in self.transformer.h:
            if isinstance(block.mlp, AdaptiveWidthMLP):
                block.mlp.temperature = temperature

    def _set_dynamic_width_force_hard(self, force_hard):
        for block in self.transformer.h:
            if isinstance(block.mlp, AdaptiveWidthMLP):
                block.mlp.force_hard = force_hard

    def _forward_logits(self, idx, targets=None, force_hard_width=False):
        old_force = []
        if force_hard_width:
            for block in self.transformer.h:
                if isinstance(block.mlp, AdaptiveWidthMLP):
                    old_force.append((block.mlp, block.mlp.force_hard))
                    block.mlp.force_hard = True
        try:
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            exit_outputs = []
            for layer_idx, block in enumerate(self.transformer.h, start=1):
                x = block(x)
                if self.config.dynamic_exit and targets is not None and layer_idx in self.exit_layers:
                    exit_logits = self.exit_heads[str(layer_idx)](x)
                    exit_outputs.append((layer_idx, exit_logits))
            x = self.transformer.ln_f(x)
            if targets is not None:
                logits = self.lm_head(x)
            else:
                logits = self.lm_head(x[:, [-1], :])
            return logits, exit_outputs
        finally:
            for mlp, force in old_force:
                mlp.force_hard = force

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

    def _free_channel_modules(self):
        modules = []
        for layer_idx, block in enumerate(self.transformer.h, start=1):
            if isinstance(block.mlp, (FreeChannelMLP, BlockSparseMLP)) and block.mlp.last_gate is not None:
                modules.append((layer_idx, block.mlp))
        return modules

    def set_free_channel_temperature(self, temperature):
        if not self.config.free_channel_mlp:
            return
        self.config.free_channel_temperature = temperature
        for block in self.transformer.h:
            if isinstance(block.mlp, FreeChannelMLP):
                block.mlp.temperature = temperature

    def set_block_sparse_temperature(self, temperature):
        if not self.config.block_sparse_mlp:
            return
        self.config.block_sparse_temperature = temperature
        for block in self.transformer.h:
            if isinstance(block.mlp, BlockSparseMLP):
                block.mlp.temperature = temperature

    def _free_channel_loss_terms(self, modules, valid_mask):
        if not modules:
            return None, None
        gates = torch.stack([mlp.last_gate for _, mlp in modules])
        gate_probs = torch.stack([mlp.last_gate_prob for _, mlp in modules])
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(0).unsqueeze(-1)
            gates = gates[valid.expand_as(gates)]
            gate_probs = gate_probs[valid.expand_as(gate_probs)]
        mean_gate_ratio = gates.float().mean()
        target_ratio = self.config.block_sparse_target_ratio if self.config.block_sparse_mlp else self.config.free_channel_target_ratio
        budget_loss = (mean_gate_ratio - target_ratio) ** 2
        channel_cost = gate_probs.float().mean()
        return budget_loss, channel_cost

    def _set_free_channel_stats(self, modules, valid_mask=None):
        if not modules:
            self.last_free_channel_stats = None
            return
        max_hidden = float(modules[0][1].max_hidden)
        bins = [(0, 64), (65, 128), (129, 192), (193, 256),
                (257, 320), (321, 384), (385, 448), (449, int(max_hidden))]
        layer_stats = []
        all_active = []
        all_entropy = []
        all_gate_prob = []
        all_gate = []

        for layer_idx, mlp in modules:
            active = mlp.last_active_channels.detach().float()
            entropy = mlp.last_gate_entropy.detach().float()
            gate_prob = mlp.last_gate_prob.detach().float()
            gate = mlp.last_gate.detach().float()
            if valid_mask is not None:
                active_values = active[valid_mask]
                entropy_values = entropy[valid_mask]
                gate_prob_values = gate_prob[valid_mask]
                gate_values = gate[valid_mask]
            else:
                active_values = active.reshape(-1)
                entropy_values = entropy.reshape(-1)
                gate_prob_values = gate_prob.reshape(-1, gate_prob.size(-1))
                gate_values = gate.reshape(-1, gate.size(-1))
            hist = {
                f"{lo}-{hi}": ((active_values >= lo) & (active_values <= hi)).float().mean().item()
                for lo, hi in bins
            }
            quantiles = torch.quantile(
                active_values.float(),
                torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=active_values.device),
            )
            layer_stats.append({
                "layer": layer_idx,
                "mean_active_channels": active_values.mean().item(),
                "median_active_channels": active_values.median().item(),
                "std_active_channels": active_values.std(unbiased=False).item(),
                "min_active_channels": active_values.min().item(),
                "max_active_channels": active_values.max().item(),
                "gate_entropy": entropy_values.mean().item(),
                "fraction_gate_gt_0_5": (gate_prob_values > 0.5).float().mean().item(),
                "fraction_gate_gt_0_9": (gate_prob_values > 0.9).float().mean().item(),
                "fraction_gate_lt_0_1": (gate_prob_values < 0.1).float().mean().item(),
                "active_width_histogram": hist,
                "active_width_quantiles": {
                    "p10": quantiles[0].item(),
                    "p25": quantiles[1].item(),
                    "p50": quantiles[2].item(),
                    "p75": quantiles[3].item(),
                    "p90": quantiles[4].item(),
                },
                "mean_channel_usage_rate": gate_values.mean(dim=0).mean().item(),
                "std_channel_usage_rate": gate_values.mean(dim=0).std(unbiased=False).item(),
                "min_channel_usage_rate": gate_values.mean(dim=0).min().item(),
                "max_channel_usage_rate": gate_values.mean(dim=0).max().item(),
            })
            all_active.append(active_values)
            all_entropy.append(entropy_values)
            all_gate_prob.append(gate_prob_values)
            all_gate.append(gate_values)

        active_cat = torch.cat(all_active)
        entropy_cat = torch.cat(all_entropy)
        gate_prob_cat = torch.cat(all_gate_prob)
        gate_cat = torch.cat(all_gate)
        hist = {
            f"{lo}-{hi}": ((active_cat >= lo) & (active_cat <= hi)).float().mean().item()
            for lo, hi in bins
        }
        quantiles = torch.quantile(
            active_cat.float(),
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=active_cat.device),
        )
        channel_usage = gate_cat.mean(dim=0)
        self.last_free_channel_stats = {
            "max_width": max_hidden,
            "mean_active_channels": active_cat.mean().item(),
            "mean_active_ratio": (active_cat.mean() / max_hidden).item(),
            "median_active_channels": active_cat.median().item(),
            "std_active_channels": active_cat.std(unbiased=False).item(),
            "min_active_channels": active_cat.min().item(),
            "max_active_channels": active_cat.max().item(),
            "gate_entropy": entropy_cat.mean().item(),
            "fraction_gate_gt_0_5": (gate_prob_cat > 0.5).float().mean().item(),
            "fraction_gate_gt_0_9": (gate_prob_cat > 0.9).float().mean().item(),
            "fraction_gate_lt_0_1": (gate_prob_cat < 0.1).float().mean().item(),
            "active_width_histogram": hist,
            "active_width_quantiles": {
                "p10": quantiles[0].item(),
                "p25": quantiles[1].item(),
                "p50": quantiles[2].item(),
                "p75": quantiles[3].item(),
                "p90": quantiles[4].item(),
            },
            "channel_usage_rate": channel_usage.detach().cpu(),
            "mean_channel_usage_rate": channel_usage.mean().item(),
            "std_channel_usage_rate": channel_usage.std(unbiased=False).item(),
            "min_channel_usage_rate": channel_usage.min().item(),
            "max_channel_usage_rate": channel_usage.max().item(),
            "layers": layer_stats,
        }

    def _block_precision_modules(self):
        modules = []
        for layer_idx, block in enumerate(self.transformer.h, start=1):
            if isinstance(block.mlp, BlockPrecisionMLP) and block.mlp.last_block_gate is not None:
                modules.append((layer_idx, block.mlp))
        return modules

    def set_block_precision_temperature(self, temperature):
        if not self.config.block_precision_mlp:
            return
        self.config.block_precision_temperature = temperature
        for block in self.transformer.h:
            if isinstance(block.mlp, BlockPrecisionMLP):
                block.mlp.temperature = temperature

    def _block_precision_loss_terms(self, modules, valid_mask):
        if not modules:
            return None
        costs = torch.stack([mlp.last_weight_bits_per_token for _, mlp in modules])
        max_costs = torch.tensor(
            [
                mlp.num_blocks
                * mlp.block_size
                * (mlp.c_fc.in_features + mlp.c_proj.out_features)
                * max(mlp.bit_choices)
                for _, mlp in modules
            ],
            device=costs.device,
            dtype=costs.dtype,
        ).view(-1, 1, 1)
        normalized = costs / max_costs.clamp_min(1.0)
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(0)
            normalized = normalized[valid.expand_as(normalized)]
        return normalized.float().mean()

    def _set_block_precision_stats(self, modules, valid_mask=None):
        if not modules:
            self.last_block_precision_stats = None
            return
        bit_choices = modules[0][1].bit_choices
        max_hidden = float(modules[0][1].max_hidden)
        layer_stats = []
        all_active_blocks = []
        all_selected_bits = []
        all_active_bits = []
        all_costs = []
        all_entropy = []

        for layer_idx, mlp in modules:
            block_gate = mlp.last_block_gate.detach().float()
            hard_blocks = mlp.last_hard_blocks.detach().float()
            selected_bits = mlp.last_selected_bits.detach().float()
            costs = mlp.last_weight_bits_per_token.detach().float()
            entropy = mlp.last_bit_entropy.detach().float()
            active_mask = block_gate > 0.5
            if valid_mask is not None:
                hard_values = hard_blocks[valid_mask]
                selected_values = selected_bits[valid_mask]
                active_values = active_mask[valid_mask]
                costs_values = costs[valid_mask]
                entropy_values = entropy[valid_mask]
            else:
                hard_values = hard_blocks.reshape(-1)
                selected_values = selected_bits.reshape(-1, selected_bits.size(-1))
                active_values = active_mask.reshape(-1, active_mask.size(-1))
                costs_values = costs.reshape(-1)
                entropy_values = entropy.reshape(-1, entropy.size(-1))

            active_selected = selected_values[active_values]
            if active_selected.numel() == 0:
                active_selected = selected_values.reshape(-1)
            bit_fractions = {
                str(bit): (active_selected == bit).float().mean().item()
                for bit in bit_choices
            }
            layer_stats.append({
                "layer": layer_idx,
                "mean_active_blocks": hard_values.mean().item(),
                "mean_active_channels": (hard_values.mean() * mlp.block_size).item(),
                "mean_active_bit": active_selected.mean().item(),
                "mean_weight_bits_per_token": costs_values.mean().item(),
                "bit_entropy": entropy_values.mean().item(),
                "bit_fractions": bit_fractions,
            })
            all_active_blocks.append(hard_values)
            all_selected_bits.append(selected_values)
            all_active_bits.append(active_values)
            all_costs.append(costs_values)
            all_entropy.append(entropy_values)

        active_blocks_cat = torch.cat(all_active_blocks)
        selected_cat = torch.cat(all_selected_bits)
        active_cat = torch.cat(all_active_bits)
        costs_cat = torch.cat(all_costs)
        entropy_cat = torch.cat(all_entropy)
        active_selected = selected_cat[active_cat]
        if active_selected.numel() == 0:
            active_selected = selected_cat.reshape(-1)
        max_bits_per_token = (
            max_hidden
            * (modules[0][1].c_fc.in_features + modules[0][1].c_proj.out_features)
            * max(bit_choices)
        )
        self.last_block_precision_stats = {
            "bit_choices": bit_choices,
            "max_width": max_hidden,
            "mean_active_blocks": active_blocks_cat.mean().item(),
            "mean_active_channels": (active_blocks_cat.mean() * modules[0][1].block_size).item(),
            "mean_active_ratio": (active_blocks_cat.mean() / modules[0][1].num_blocks).item(),
            "mean_active_bit": active_selected.mean().item(),
            "mean_weight_bits_per_token": costs_cat.mean().item(),
            "mean_weight_bit_fraction": (costs_cat.mean() / max_bits_per_token).item(),
            "bit_entropy": entropy_cat.mean().item(),
            "bit_fractions": {
                str(bit): (active_selected == bit).float().mean().item()
                for bit in bit_choices
            },
            "layers": layer_stats,
        }

    def _resource_mode_modules(self):
        modules = []
        for layer_idx, block in enumerate(self.transformer.h, start=1):
            if isinstance(block.mlp, ResourceModeMLP) and block.mlp.last_mode_onehot is not None:
                modules.append((layer_idx, block.mlp))
        return modules

    def set_resource_mode_temperature(self, temperature):
        if not self.config.resource_mode_mlp:
            return
        self.config.resource_mode_temperature = temperature
        for block in self.transformer.h:
            if isinstance(block.mlp, ResourceModeMLP):
                block.mlp.temperature = temperature

    def _resource_mode_loss_terms(self, modules, valid_mask):
        if not modules:
            return None
        costs = torch.stack([mlp.last_weight_bits_per_token for _, mlp in modules])
        max_costs = torch.tensor(
            [
                mlp.num_blocks
                * mlp.block_size
                * (mlp.c_fc.in_features + mlp.c_proj.out_features)
                * max(int(bits) for _, bits in mlp.modes)
                for _, mlp in modules
            ],
            device=costs.device,
            dtype=costs.dtype,
        ).view(-1, 1, 1)
        normalized = costs / max_costs.clamp_min(1.0)
        if valid_mask is not None:
            valid = valid_mask.unsqueeze(0)
            normalized = normalized[valid.expand_as(normalized)]
        return normalized.float().mean()

    def _set_resource_mode_stats(self, modules, valid_mask=None):
        if not modules:
            self.last_resource_mode_stats = None
            return
        modes = modules[0][1].modes
        layer_stats = []
        all_selected = []
        all_probs = []
        all_costs = []
        all_entropy = []

        for layer_idx, mlp in modules:
            selected = mlp.last_selected_mode.detach()
            probs = mlp.last_mode_probs.detach().float()
            costs = mlp.last_weight_bits_per_token.detach().float()
            entropy = mlp.last_router_entropy.detach().float()
            if valid_mask is not None:
                selected_values = selected[valid_mask]
                probs_values = probs[valid_mask]
                costs_values = costs[valid_mask]
                entropy_values = entropy[valid_mask]
            else:
                selected_values = selected.reshape(-1)
                probs_values = probs.reshape(-1, probs.size(-1))
                costs_values = costs.reshape(-1)
                entropy_values = entropy.reshape(-1)
            fractions = torch.bincount(selected_values, minlength=len(modes)).float()
            fractions = fractions / fractions.sum().clamp_min(1.0)
            prob_means = probs_values.mean(dim=0)
            mode_blocks = mlp.mode_blocks.to(selected_values.device)[selected_values].float()
            mode_bits = mlp.mode_bits.to(selected_values.device)[selected_values].float()
            layer_stats.append({
                "layer": layer_idx,
                "mean_active_blocks": mode_blocks.mean().item(),
                "mean_active_channels": (mode_blocks.mean() * mlp.block_size).item(),
                "mean_active_bit": mode_bits.mean().item(),
                "mean_weight_bits_per_token": costs_values.mean().item(),
                "router_entropy": entropy_values.mean().item(),
                "mode_fractions": {
                    f"{blocks}x{bits}": fractions[i].item()
                    for i, (blocks, bits) in enumerate(modes)
                },
                "mode_prob_means": {
                    f"{blocks}x{bits}": prob_means[i].item()
                    for i, (blocks, bits) in enumerate(modes)
                },
            })
            all_selected.append(selected_values)
            all_probs.append(probs_values)
            all_costs.append(costs_values)
            all_entropy.append(entropy_values)

        selected_cat = torch.cat(all_selected)
        probs_cat = torch.cat(all_probs)
        costs_cat = torch.cat(all_costs)
        entropy_cat = torch.cat(all_entropy)
        fractions = torch.bincount(selected_cat, minlength=len(modes)).float()
        fractions = fractions / fractions.sum().clamp_min(1.0)
        prob_means = probs_cat.mean(dim=0)
        first = modules[0][1]
        selected_blocks = first.mode_blocks.to(selected_cat.device)[selected_cat].float()
        selected_bits = first.mode_bits.to(selected_cat.device)[selected_cat].float()
        max_bits_per_token = (
            first.num_blocks
            * first.block_size
            * (first.c_fc.in_features + first.c_proj.out_features)
            * max(int(bits) for _, bits in modes)
        )
        self.last_resource_mode_stats = {
            "modes": [f"{blocks}x{bits}" for blocks, bits in modes],
            "mean_active_blocks": selected_blocks.mean().item(),
            "mean_active_channels": (selected_blocks.mean() * first.block_size).item(),
            "mean_active_ratio": (selected_blocks.mean() / first.num_blocks).item(),
            "mean_active_bit": selected_bits.mean().item(),
            "mean_weight_bits_per_token": costs_cat.mean().item(),
            "mean_weight_bit_fraction": (costs_cat.mean() / max_bits_per_token).item(),
            "router_entropy": entropy_cat.mean().item(),
            "mode_fractions": {
                f"{blocks}x{bits}": fractions[i].item()
                for i, (blocks, bits) in enumerate(modes)
            },
            "mode_prob_means": {
                f"{blocks}x{bits}": prob_means[i].item()
                for i, (blocks, bits) in enumerate(modes)
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
            self._set_free_channel_stats(self._free_channel_modules())
            self._set_block_precision_stats(self._block_precision_modules())
            self._set_resource_mode_stats(self._resource_mode_modules())

        return final_logits, None

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        if self.config.dynamic_exit and targets is None and not self.training:
            return self._forward_dynamic_inference(idx, pos)

        logits, exit_outputs = self._forward_logits(idx, targets)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
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
            dynamic_width_hard_loss = None
            if width_modules and self.config.dynamic_width_hard_loss_weight != 0.0:
                hard_logits, _ = self._forward_logits(idx, targets, force_hard_width=True)
                dynamic_width_hard_loss = F.cross_entropy(
                    hard_logits.view(-1, hard_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )
                loss = loss + self.config.dynamic_width_hard_loss_weight * dynamic_width_hard_loss
            self._set_dynamic_width_stats(width_modules, valid_mask=targets != -1)
            free_channel_modules = self._free_channel_modules()
            free_channel_budget_loss, free_channel_cost = self._free_channel_loss_terms(free_channel_modules, targets != -1)
            budget_weight = self.config.block_sparse_budget_weight if self.config.block_sparse_mlp else self.config.free_channel_budget_weight
            cost_weight = self.config.block_sparse_cost_weight if self.config.block_sparse_mlp else self.config.free_channel_cost_weight
            if free_channel_budget_loss is not None:
                loss = loss + budget_weight * free_channel_budget_loss
            if free_channel_cost is not None and cost_weight != 0.0:
                loss = loss + cost_weight * free_channel_cost
            self._set_free_channel_stats(free_channel_modules, valid_mask=targets != -1)
            block_precision_modules = self._block_precision_modules()
            block_precision_cost = self._block_precision_loss_terms(block_precision_modules, targets != -1)
            if block_precision_cost is not None:
                loss = loss + self.config.block_precision_cost_weight * block_precision_cost
            self._set_block_precision_stats(block_precision_modules, valid_mask=targets != -1)
            resource_mode_modules = self._resource_mode_modules()
            resource_mode_cost = self._resource_mode_loss_terms(resource_mode_modules, targets != -1)
            if resource_mode_cost is not None:
                loss = loss + self.config.resource_mode_cost_weight * resource_mode_cost
            self._set_resource_mode_stats(resource_mode_modules, valid_mask=targets != -1)
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
            if dynamic_width_hard_loss is not None:
                self.last_loss_stats["dynamic_width_hard_loss"] = dynamic_width_hard_loss.detach().item()
            if free_channel_budget_loss is not None:
                self.last_loss_stats["free_channel_budget_loss"] = free_channel_budget_loss.detach().item()
            if free_channel_cost is not None:
                self.last_loss_stats["free_channel_cost"] = free_channel_cost.detach().item()
            if block_precision_cost is not None:
                self.last_loss_stats["block_precision_cost"] = block_precision_cost.detach().item()
            if resource_mode_cost is not None:
                self.last_loss_stats["resource_mode_cost"] = resource_mode_cost.detach().item()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            loss = None
            self.last_exit_stats = None
            self.last_exit_details = None
            self.last_loss_stats = None
            gates, hard_masks = self._dynamic_mlp_gates()
            self._set_dynamic_mlp_stats(gates, hard_masks)
            self._set_dynamic_width_stats(self._dynamic_width_modules())
            self._set_free_channel_stats(self._free_channel_modules())
            self._set_block_precision_stats(self._block_precision_modules())
            self._set_resource_mode_stats(self._resource_mode_modules())

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
