"""Attention-free-depth extension for the frozen Full-Free v1 graph.

This module deliberately lives outside ``model.py`` so the v1 architecture and
its recorded hashes stay frozen.  It reuses the v1 router, Compute Cells and
standard causal Attention weights, but makes each step's Attention conditional
on that step being present in the same Cell graph.

There is no second Attention router.  The sparsemax "skip" mass already emitted
by the v1 step router defines whether a step exists:

    step active = any Compute Cell in this step is active

For training, a straight-through gate gives hard forward semantics and a soft
gradient.  During evaluation a completely empty step does not call Attention at
all.  Mixed token masks still use dense Attention followed by masking; a future
query-queue executor can replace that path without changing model semantics.

``cell_graph_attention_cells`` is repurposed only by v2 as the number of evenly
spaced mandatory Attention anchors.  With 8 steps and a value of 3 the anchors
are [0, 4, 7].  The value is saved in ordinary nanoGPT checkpoints, making the
experimental architecture reproducible without changing GPTConfig.
"""

import math

import torch
import torch.nn.functional as F

from model import FullFreeDynamicCellGraph, GPT


def evenly_spaced_anchors(num_steps, count):
    """Return deterministic endpoints-inclusive anchor indices."""
    count = max(0, min(int(count), int(num_steps)))
    if count == 0:
        return []
    if count == 1:
        return [0]
    anchors = {
        int(round(index * (num_steps - 1) / (count - 1)))
        for index in range(count)
    }
    # Rounding can only collide for invalid count > num_steps, clamped above.
    return sorted(anchors)


class FullFreeAttentionDynamicCellGraphV2(FullFreeDynamicCellGraph):
    """One graph controls Cell input, width, depth, path and Attention depth."""

    architecture_name = "full-free-attention-v2"

    def __init__(self, config):
        super().__init__(config)
        anchors = evenly_spaced_anchors(
            self.num_steps, config.cell_graph_attention_cells
        )
        anchor_mask = torch.zeros(self.num_steps, dtype=torch.bool)
        if anchors:
            anchor_mask[anchors] = True
        self.register_buffer("attention_anchor_mask", anchor_mask, persistent=True)
        self.register_buffer(
            "v2_dual_updates", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.last_attention_probs = None
        self.last_attention_cost_probs = None
        self.last_attention_mask = None
        self.last_attention_executed_steps = None
        self.force_dense_attention_execution = False
        # Optional learned-plan condensation used only for evaluation/export.
        # None keeps the fully dynamic graph; a bool [num_steps] tensor forces
        # every token onto one shared, hardware-friendly Attention plan.
        self.attention_plan_override = None
        self.physical_cell_execution = False

    @staticmethod
    def _attention_gate(cell_weights, node_masks, anchor, training):
        # Sparsemax distributes unit mass over Cells plus an implicit skip
        # choice.  Therefore sum(Cell weights) is exactly 1 - skip probability.
        soft = cell_weights.sum(dim=-1).clamp(0.0, 1.0)
        hard = node_masks.any(dim=-1).to(cell_weights.dtype)
        if anchor:
            soft = torch.ones_like(soft)
            hard = torch.ones_like(hard)
        gate = hard + soft - soft.detach() if training else hard
        return gate, soft, hard.bool()

    def attention_stats(self):
        if self.last_attention_mask is None:
            return None
        mask = self.last_attention_mask.float()
        return {
            "architecture": self.architecture_name,
            "mandatory_steps": self.attention_anchor_mask.nonzero().flatten().tolist(),
            "mean_active_attention_steps": mask.sum(dim=-1).mean().item(),
            "mean_active_attention_ratio": mask.mean().item(),
            "empty_attention_step_fraction": (~self.last_attention_mask).float().mean().item(),
            "physically_executed_attention_steps": int(
                self.last_attention_executed_steps.sum().item()
            ),
        }

    def forward(self, anchor):
        B, T, _ = anchor.shape
        temperature = max(float(self.temperature), 1e-4)
        context = self.router.graph_context(anchor)
        current = anchor
        cell_outputs = [None] * self.num_cells
        cell_source_features = [None] * self.num_cells
        cell_available = [None] * self.num_cells
        node_weights_records = []
        node_mask_records = []
        edge_weights_records = []
        edge_mask_records = []
        node_score_records = []
        edge_score_records = []
        halt_records = []
        attention_prob_records = []
        attention_mask_records = []
        attention_executed_records = []
        alive = torch.ones(B, T, device=anchor.device, dtype=anchor.dtype)

        for step in range(self.num_steps):
            # Route before Attention so the same graph decision can physically
            # skip Attention.  This also avoids paying Attention merely to learn
            # that the step should not exist.
            local_context = context + self.router.state_down(current)
            routed, node_scores = self.router.node_weights(
                local_context, step, temperature,
                self.config.cell_graph_node_selector,
            )
            cell_weights = routed[..., :self.cells_per_step] * alive.unsqueeze(-1)
            halt_weight = (
                routed[..., -1]
                if self.config.cell_graph_halt else torch.zeros_like(alive)
            )
            if self.training and self.exploration > 0:
                explore = torch.rand(B, T, 1, device=anchor.device) < self.exploration
                random_cells = F.one_hot(
                    torch.randint(self.cells_per_step, (B, T), device=anchor.device),
                    self.cells_per_step,
                ).to(cell_weights.dtype)
                cell_weights = torch.where(
                    explore, random_cells * alive.unsqueeze(-1), cell_weights
                )
            cell_weights = self._override_nodes(
                cell_weights, step, B, T
            ) * alive.unsqueeze(-1)
            node_masks = cell_weights > 0

            attention_gate, attention_soft, attention_hard = self._attention_gate(
                cell_weights,
                node_masks,
                bool(self.attention_anchor_mask[step]),
                self.training,
            )
            if self.attention_plan_override is not None:
                plan_active = bool(self.attention_plan_override[step])
                fill = 1.0 if plan_active else 0.0
                attention_gate = torch.full_like(attention_gate, fill)
                attention_soft = torch.full_like(attention_soft, fill)
                attention_hard = torch.full_like(attention_hard, plan_active)
            execute_attention = (
                self.training
                or self.force_dense_attention_execution
                or bool(attention_hard.any())
            )
            if execute_attention:
                attention_delta = self.fixed_attentions[step](
                    self.fixed_attention_norms[step](current)
                )
                current = current + attention_gate.unsqueeze(-1) * attention_delta
            attention_prob_records.append(attention_soft)
            attention_mask_records.append(attention_hard)
            attention_executed_records.append(execute_attention)

            # Attention changes the current state, so source/edge features use
            # the post-Attention state while the node decision remains the one
            # made before the optional computation.
            local_context = context + self.router.state_down(current)
            step_deltas = []
            candidate_indices = self._candidate_indices(step)
            source_tensors = [current]
            source_features = [self.router.source_down(current)]
            source_available = [
                torch.ones(B, T, dtype=torch.bool, device=anchor.device)
            ]
            for source_index in candidate_indices[1:]:
                prior_node = source_index - 1
                source_tensors.append(cell_outputs[prior_node])
                source_features.append(cell_source_features[prior_node])
                source_available.append(cell_available[prior_node])
            sources = torch.stack(source_tensors, dim=2)
            routed_sources = torch.stack(source_features, dim=2)
            available = torch.stack(source_available, dim=-1)
            step_edge_weights, step_edge_scores = self.router.edge_weights_step(
                local_context, routed_sources, available,
                step * self.cells_per_step, temperature,
                self.config.cell_graph_edge_selector,
            )

            for offset in range(self.cells_per_step):
                node = step * self.cells_per_step + offset
                edge_weights = step_edge_weights[..., offset, :]
                edge_scores = step_edge_scores[..., offset, :]
                edge_weights = self._override_edges(
                    edge_weights, node, candidate_indices, B, T
                )
                edge_weights = edge_weights * available.to(edge_weights.dtype)
                empty = edge_weights.sum(dim=-1, keepdim=True) == 0
                fallback = torch.zeros_like(edge_weights)
                fallback[..., 0] = 1.0
                edge_weights = torch.where(empty, fallback, edge_weights)
                edge_weights = edge_weights / edge_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                projected = self._project_sources(node, sources)
                fused = (projected * edge_weights.unsqueeze(-1)).sum(dim=2)
                if self.physical_cell_execution and not self.training:
                    active_flat = node_masks[..., offset].reshape(-1)
                    fused_flat = fused.reshape(B * T, -1)
                    delta_flat = torch.zeros_like(fused_flat)
                    if bool(active_flat.any()):
                        queued = self.input_norms[node](fused_flat[active_flat])
                        delta_flat[active_flat] = self.cells[node](queued).to(
                            delta_flat.dtype
                        )
                    delta = delta_flat.view_as(fused)
                else:
                    fused = self.input_norms[node](fused)
                    delta = self.cells[node](fused)
                step_deltas.append(delta)
                cell_outputs[node] = delta
                cell_source_features[node] = self.router.source_down(delta)
                cell_available[node] = node_masks[..., offset]

                padded_weights = torch.zeros(
                    B, T, 1 + self.num_cells,
                    device=anchor.device, dtype=edge_weights.dtype,
                )
                padded_scores = torch.zeros(
                    B, T, 1 + self.num_cells,
                    device=anchor.device, dtype=edge_scores.dtype,
                )
                padded_weights[..., candidate_indices] = edge_weights
                padded_scores[..., candidate_indices] = edge_scores
                edge_weights_records.append(padded_weights)
                edge_mask_records.append(
                    (padded_weights > 0) & node_masks[..., offset, None]
                )
                edge_score_records.append(padded_scores)

            deltas = torch.stack(step_deltas, dim=2)
            current = current + (deltas * cell_weights.unsqueeze(-1)).sum(dim=2)
            node_weights_records.append(cell_weights)
            node_mask_records.append(node_masks)
            node_score_records.append(node_scores[..., :self.cells_per_step])
            halt_records.append(halt_weight * alive)
            if self.config.cell_graph_halt:
                alive = alive * (1.0 - halt_weight)

        node_weights = torch.cat(node_weights_records, dim=-1)
        node_mask = torch.cat(node_mask_records, dim=-1)
        edge_weights = torch.stack(edge_weights_records, dim=2)
        edge_mask = torch.stack(edge_mask_records, dim=2)
        self.last_node_probs = node_weights
        self.last_node_mask = node_mask.detach()
        self.last_edge_probs = edge_weights
        self.last_edge_mask = edge_mask.detach()
        self.last_node_scores = torch.cat(node_score_records, dim=-1)
        self.last_edge_scores = torch.stack(edge_score_records, dim=2)
        self.last_halt_weights = torch.stack(halt_records, dim=-1)
        self.last_depth = self._hard_depth(node_mask.detach(), edge_mask.detach())
        self.last_attention_probs = torch.stack(attention_prob_records, dim=-1).detach()
        # Keep the differentiable copy until GPT builds the joint compute loss.
        self.last_attention_cost_probs = torch.stack(attention_prob_records, dim=-1)
        self.last_attention_mask = torch.stack(attention_mask_records, dim=-1).detach()
        self.last_attention_executed_steps = torch.tensor(
            attention_executed_records, device=anchor.device, dtype=torch.bool
        )
        return current


class FullFreeAttentionV2GPT(GPT):
    """GPT loss/stat hooks for a single Cell+Attention compute budget.

    ``cell_graph_active_cell_budget`` is measured in Cell-equivalent MAC units
    for v2.  One standard Attention step costs approximately

        (4*C*C + 2*T*C) / (2*C*A) = (2*C + T) / A

    Compute Cells, where C is embedding width, T block length and A atom width.
    For the pilot (C=128, T=128, A=64), one Attention is six Cell units.
    """

    def _attention_cell_equivalent(self):
        return (
            2 * self.config.n_embd + self.config.block_size
        ) / self.config.cell_graph_atom_size

    def _cell_graph_loss_terms(self, valid_mask):
        budget, edge_cost, balance, expected_node_ratio = super()._cell_graph_loss_terms(
            valid_mask
        )
        graph = self.cell_graph
        attention = graph.last_attention_cost_probs
        selected_attention = attention[valid_mask]
        expected_attention_steps = selected_attention.float().sum(dim=-1).mean()
        expected_cell_units = expected_node_ratio * graph.num_cells
        effective_units = (
            expected_cell_units
            + expected_attention_steps * self._attention_cell_equivalent()
        )
        # The frozen GPT forward multiplies this value by num_cells before the
        # dual penalty. Return a normalized joint compute cost in that slot.
        return budget, edge_cost, balance, effective_units / graph.num_cells

    def _set_cell_graph_stats(self, valid_mask=None):
        super()._set_cell_graph_stats(valid_mask)
        graph = self.cell_graph
        if self.last_cell_graph_stats is None or graph.last_attention_mask is None:
            return
        attention_mask = graph.last_attention_mask
        if valid_mask is not None:
            attention_mask = attention_mask[valid_mask]
        else:
            attention_mask = attention_mask.reshape(-1, graph.num_steps)
        mean_attention = attention_mask.float().sum(dim=-1).mean()
        mean_cells = self.last_cell_graph_stats["mean_active_cells"]
        equivalent = self._attention_cell_equivalent()
        effective_units = mean_cells + mean_attention.item() * equivalent
        self.last_cell_graph_stats.update({
            "mean_active_attention_steps": mean_attention.item(),
            "mean_active_attention_ratio": mean_attention.item() / graph.num_steps,
            "attention_cell_equivalent": equivalent,
            "mean_effective_compute_units": effective_units,
            "physically_executed_attention_steps": int(
                graph.last_attention_executed_steps.sum().item()
            ),
        })

    def update_cell_graph_dual(self):
        if (
            self.config.cell_graph_budget_mode != "dual_active_cells"
            or self.last_cell_graph_stats is None
        ):
            return self.config.cell_graph_dual_value
        self.cell_graph.v2_dual_updates.add_(1)
        updates = int(self.cell_graph.v2_dual_updates.item())
        # v2 reuses this otherwise inert field (exploration is 0 -> 0 in the
        # pilot) as a checkpointed dual warmup.  A linear ramp of the same
        # length prevents early irreversible sparsemax collapse.
        warmup = int(self.config.cell_graph_exploration_anneal_iters)
        if updates <= warmup:
            self.config.cell_graph_dual_value = 0.0
            return 0.0
        ramp = 1.0 if warmup <= 0 else min((updates - warmup) / warmup, 1.0)
        actual = self.last_cell_graph_stats["mean_effective_compute_units"]
        error = (
            actual - self.config.cell_graph_active_cell_budget
        ) / self.cell_graph.num_cells
        self.config.cell_graph_dual_value = max(
            0.0,
            float(
                self.config.cell_graph_dual_value
                + self.config.cell_graph_dual_lr * ramp * error
            ),
        )
        self.last_cell_graph_stats["dual_schedule_scale"] = ramp
        return self.config.cell_graph_dual_value


def install_into_frozen_model_module(model_module):
    """Patch only the in-process constructor used by the dedicated v2 runner."""
    model_module.FullFreeDynamicCellGraph = FullFreeAttentionDynamicCellGraphV2
    model_module.GPT = FullFreeAttentionV2GPT
