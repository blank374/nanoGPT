"""Low-overhead serving helpers for hardware-atom nanoGPT models."""

from contextlib import nullcontext

import torch

from model import HardwareAtomMLP


def hardware_atom_modules(model):
    return [
        block.mlp for block in model.transformer.h
        if isinstance(block.mlp, HardwareAtomMLP)
    ]


class HardwareAtomCUDAGraphRunner:
    """Replay a fixed-shape, fixed-route atom batch with one CPU launch.

    Exact sequence routes must be cached before construction. Input values may
    change between calls, but batch and sequence shapes and route assignments
    must remain fixed. Rebuild the runner when the serving bucket changes.
    """

    def __init__(self, model, example_idx, dtype=torch.float16, warmup=3):
        if example_idx.device.type != 'cuda':
            raise ValueError('CUDA Graph runner requires a CUDA input tensor')
        if model.training:
            raise ValueError('CUDA Graph runner requires model.eval()')
        modules = hardware_atom_modules(model)
        if not modules or any(mlp.cached_sequence_indices is None for mlp in modules):
            raise ValueError('cache exact sequence routes before graph capture')

        # CUDA Graph stores raw addresses, so each runner must own strong
        # references to the exact route tensors used during its capture. The
        # model module may later install another plan while building more graphs.
        self.route_modes = [mlp.cached_sequence_modes for mlp in modules]
        self.route_indices = [tuple(mlp.cached_sequence_indices) for mlp in modules]

        self.model = model
        self.shape = tuple(example_idx.shape)
        self.device = example_idx.device
        self.input_dtype = example_idx.dtype
        self.static_input = torch.empty_like(example_idx)
        self.static_input.copy_(example_idx)
        self.graph = torch.cuda.CUDAGraph()
        self.autocast = (torch.amp.autocast(device_type='cuda', dtype=dtype)
                         if dtype != torch.float32 else nullcontext())

        # Diagnostics contain Python-side reductions and are unnecessary on a
        # replay path. The prefill pass has already recorded route statistics.
        previous_stats_flags = [mlp.runtime_stats_enabled for mlp in modules]
        try:
            for mlp in modules:
                mlp.runtime_stats_enabled = False

            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(capture_stream):
                for _ in range(warmup):
                    with self.autocast:
                        model.forward_inference_fast(
                            self.static_input, compute_address=False
                        )
            torch.cuda.current_stream(self.device).wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)

            with torch.cuda.graph(self.graph):
                with self.autocast:
                    self.static_logits = model.forward_inference_fast(
                        self.static_input, compute_address=False
                    )
        finally:
            for mlp, enabled in zip(modules, previous_stats_flags):
                mlp.runtime_stats_enabled = enabled

    def __call__(self, idx):
        if tuple(idx.shape) != self.shape or idx.dtype != self.input_dtype or idx.device != self.device:
            raise ValueError(
                f'expected input shape={self.shape}, dtype={self.input_dtype}, device={self.device}; '
                f'got shape={tuple(idx.shape)}, dtype={idx.dtype}, device={idx.device}'
            )
        self.static_input.copy_(idx)
        self.graph.replay()
        return self.static_logits


class DirectAddressCUDAGraphDispatcher:
    """Route hot requests directly into reusable per-plan CUDA Graph buckets."""

    def __init__(self, model, hot_plan_ids, confidence=0.6, dtype=torch.float16,
                 graph_warmup=3):
        if model.hardware_atom_address is None:
            raise ValueError('model has no jointly trained direct address head')
        if model.training:
            raise ValueError('dispatcher requires model.eval()')
        self.model = model
        self.modules = hardware_atom_modules(model)
        self.num_modes = len(self.modules[0].width_choices)
        self.hot_plan_ids = torch.as_tensor(
            hot_plan_ids, dtype=torch.long, device=next(model.parameters()).device
        )
        self.confidence = float(confidence)
        self.dtype = dtype
        self.graph_warmup = graph_warmup
        self.graph_cache = {}
        self.last_stats = None

    def _autocast(self):
        return (torch.amp.autocast(device_type='cuda', dtype=self.dtype)
                if self.dtype != torch.float32 else nullcontext())

    def _address(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        hidden = self.model.transformer.wte(idx) + self.model.transformer.wpe(positions)
        logits = self.model.hardware_atom_address(hidden)
        probabilities = logits.softmax(dim=-1)
        routes = logits.argmax(dim=-1)
        powers = self.num_modes ** torch.arange(routes.size(1), device=idx.device)
        plan_ids = (routes * powers).sum(dim=1)
        confidence = probabilities.max(dim=-1).values.min(dim=1).values
        accepted = torch.isin(plan_ids, self.hot_plan_ids) & (confidence >= self.confidence)
        return routes, plan_ids, accepted

    @staticmethod
    def _capacity(size):
        size = max(1, size)
        if size <= 8:
            return 1 << (size - 1).bit_length()
        # Large powers of two waste too much compute for a skewed hot path
        # distribution (for example 41 requests padded to 64). Eight-request
        # tiles keep graph reuse practical while bounding padding below eight.
        return ((size + 7) // 8) * 8

    def _runner(self, plan_id, sequence_length, capacity, example):
        key = (int(plan_id), int(sequence_length), int(capacity))
        if key in self.graph_cache:
            return self.graph_cache[key]
        modes = []
        value = int(plan_id)
        for _ in self.modules:
            modes.append(value % self.num_modes)
            value //= self.num_modes
        for mlp, mode in zip(self.modules, modes):
            selected = torch.full(
                (capacity, 1), mode, dtype=torch.long, device=example.device
            )
            mlp.cache_sequence_routes(selected)
            mlp.eval_impl = 'grouped'
        runner = HardwareAtomCUDAGraphRunner(
            self.model, example, dtype=self.dtype, warmup=self.graph_warmup
        )
        self.graph_cache[key] = runner
        return runner

    def clear_graph_cache(self):
        self.graph_cache.clear()

    @torch.no_grad()
    def __call__(self, idx):
        if idx.device.type != 'cuda':
            raise ValueError('direct address CUDA Graph dispatch requires CUDA input')
        with self._autocast():
            routes, plan_ids, accepted = self._address(idx)

        # One compact device-to-host transfer is cheaper than synchronizing on
        # every plan id and every statistic separately, especially under WDDM.
        plan_ids_cpu = plan_ids.detach().cpu()
        accepted_cpu = accepted.detach().cpu()

        chunks = []
        accepted_count = int(accepted_cpu.sum())
        if accepted_count:
            accepted_plans_cpu = plan_ids_cpu[accepted_cpu]
            for plan_id in accepted_plans_cpu.unique().tolist():
                indices_cpu = ((plan_ids_cpu == plan_id) & accepted_cpu).nonzero().flatten()
                indices = indices_cpu.to(idx.device)
                group = idx.index_select(0, indices)
                count = group.size(0)
                capacity = self._capacity(count)
                if capacity != count:
                    padding = group[:1].expand(capacity - count, -1)
                    graph_input = torch.cat((group, padding), dim=0)
                else:
                    graph_input = group
                runner = self._runner(plan_id, idx.size(1), capacity, graph_input)
                chunks.append((indices, runner(graph_input)[:count]))

        cold_indices_cpu = (~accepted_cpu).nonzero().flatten()
        cold_count = int(cold_indices_cpu.numel())
        if cold_count:
            cold_indices = cold_indices_cpu.to(idx.device)
            for mlp in self.modules:
                mlp.clear_cached_sequence_routes()
                mlp.eval_impl = 'grouped'
            cold = idx.index_select(0, cold_indices)
            with self._autocast():
                cold_logits = self.model.forward_inference_fast(cold, compute_address=False)
            chunks.append((cold_indices, cold_logits))

        output_dtype = chunks[0][1].dtype
        output = torch.empty(
            idx.size(0), 1, self.model.config.vocab_size,
            dtype=output_dtype, device=idx.device
        )
        for indices, logits in chunks:
            output.index_copy_(0, indices, logits.to(output_dtype))
        self.last_stats = {
            'accepted_fraction': accepted_count / idx.size(0),
            'accepted_requests': accepted_count,
            'cold_requests': cold_count,
            'hot_groups': len(chunks) - int(cold_count > 0),
            'graph_cache_entries': len(self.graph_cache),
        }
        return output
