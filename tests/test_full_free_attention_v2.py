import torch

from model import GPTConfig
from experiments.full_free_attention_v2 import (
    FullFreeAttentionDynamicCellGraphV2,
    FullFreeAttentionV2GPT,
    evenly_spaced_anchors,
)


def tiny_config(anchor_count=3):
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=8,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        cell_graph=True,
        cell_graph_mode="full_free",
        cell_graph_cells_per_step=2,
        cell_graph_attention_cells=anchor_count,
        cell_graph_fixed_attention=True,
        cell_graph_atom_size=8,
        cell_graph_router_hidden=8,
        cell_graph_lookback_steps=2,
    )


def test_evenly_spaced_attention_anchors():
    assert evenly_spaced_anchors(8, 3) == [0, 4, 7]
    assert evenly_spaced_anchors(8, 0) == []
    assert evenly_spaced_anchors(8, 1) == [0]


def test_attention_gate_is_tied_to_same_cell_graph_step():
    weights = torch.tensor([[[0.0, 0.0], [0.25, 0.75]]])
    masks = weights > 0
    gate, soft, hard = FullFreeAttentionDynamicCellGraphV2._attention_gate(
        weights, masks, anchor=False, training=False
    )
    assert torch.equal(hard, torch.tensor([[False, True]]))
    assert torch.equal(gate, hard.float())
    assert torch.allclose(soft, torch.tensor([[0.0, 1.0]]))


def test_anchor_attention_cannot_be_skipped():
    weights = torch.zeros(1, 2, 2)
    masks = weights.bool()
    gate, soft, hard = FullFreeAttentionDynamicCellGraphV2._attention_gate(
        weights, masks, anchor=True, training=False
    )
    assert hard.all()
    assert gate.eq(1).all()
    assert soft.eq(1).all()


def test_eval_physically_skips_empty_optional_attention_steps():
    graph = FullFreeAttentionDynamicCellGraphV2(tiny_config(anchor_count=3)).eval()
    # No Compute Cells are active anywhere.  Only anchors [0,4,7] may execute.
    graph.node_override = torch.zeros(graph.num_cells)
    output = graph(torch.randn(2, 8, 16))
    assert output.shape == (2, 8, 16)
    assert graph.last_attention_mask[:, :, [0, 4, 7]].all()
    assert not graph.last_attention_mask[:, :, [1, 2, 3, 5, 6]].any()
    assert graph.last_attention_executed_steps.tolist() == [
        True, False, False, False, True, False, False, True
    ]
    stats = graph.attention_stats()
    assert stats["mean_active_attention_steps"] == 3.0
    assert stats["physically_executed_attention_steps"] == 3


def test_training_gate_preserves_gradient_to_router_mass():
    weights = torch.tensor([[[0.2, 0.0]]], requires_grad=True)
    masks = weights > 0
    gate, _, _ = FullFreeAttentionDynamicCellGraphV2._attention_gate(
        weights, masks, anchor=False, training=True
    )
    gate.sum().backward()
    assert torch.equal(weights.grad, torch.ones_like(weights))


def test_attention_cost_is_expressed_in_cell_equivalents():
    model = FullFreeAttentionV2GPT(tiny_config(anchor_count=2))
    # (2*C + T) / atom = (32 + 8) / 8 = 5 Cell-equivalent units.
    assert model._attention_cell_equivalent() == 5.0


def test_physical_cell_queue_preserves_eval_output():
    torch.manual_seed(7)
    graph = FullFreeAttentionDynamicCellGraphV2(tiny_config(anchor_count=2)).eval()
    # A deterministic sparse graph exercises gather/scatter without depending
    # on random router support patterns.
    override = torch.zeros(graph.num_cells)
    override[::2] = 1.0
    graph.node_override = override
    anchor = torch.randn(2, 8, 16)
    dense = graph(anchor)
    graph.physical_cell_execution = True
    queued = graph(anchor)
    assert torch.allclose(dense, queued, atol=1e-6, rtol=1e-6)
