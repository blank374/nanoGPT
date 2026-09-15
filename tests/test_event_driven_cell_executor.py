import torch
import torch.nn as nn

from experiments.event_driven_cell_executor import (
    CellEventBatch,
    EventDrivenCellExecutor,
    PersistentCellEventScheduler,
)
from model import GPTConfig
from experiments.full_free_attention_v2 import FullFreeAttentionDynamicCellGraphV2


def tiny_config():
    return GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        cell_graph=True,
        cell_graph_mode="full_free",
        cell_graph_cells_per_step=2,
        cell_graph_attention_cells=2,
        cell_graph_fixed_attention=True,
        cell_graph_atom_size=8,
        cell_graph_router_hidden=8,
        cell_graph_lookback_steps=2,
    )


def test_same_cell_events_from_request_shards_are_coalesced():
    executor = EventDrivenCellExecutor()
    linear = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(4))
    jobs = [
        CellEventBatch(
            0,
            torch.arange(8, dtype=torch.float32).view(1, 2, 4),
            torch.tensor([[True, False]]),
            torch.tensor([[1, 1]]),
        ),
        CellEventBatch(
            0,
            torch.arange(8, 16, dtype=torch.float32).view(1, 2, 4),
            torch.tensor([[True, True]]),
            torch.tensor([[2, 1]]),
        ),
    ]
    outputs = executor.execute(jobs, nn.ModuleList([nn.Identity()]), nn.ModuleList([linear]))
    assert torch.equal(outputs[0][0, 0], jobs[0].fused[0, 0])
    assert outputs[0][0, 1].eq(0).all()
    assert torch.equal(outputs[1], jobs[1].fused)
    assert executor.last_stats["ready_events"] == 3
    assert executor.last_stats["cell_launches"] == 1
    assert executor.last_stats["coalesced_jobs"] == 1


def test_dependency_counter_builds_ready_waves():
    nodes = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    edges = torch.zeros(4, 5, dtype=torch.bool)
    edges[0, 0] = True
    edges[1, 0] = True
    edges[2, 1] = True  # C2 depends on C0.
    edges[3, 2] = True  # C3 depends on C1.
    assert EventDrivenCellExecutor.dependency_waves(nodes, edges) == [[0, 1], [2, 3]]


def test_event_executor_preserves_dynamic_graph_output():
    torch.manual_seed(11)
    graph = FullFreeAttentionDynamicCellGraphV2(tiny_config()).eval()
    override = torch.zeros(graph.num_cells)
    override[::2] = 1.0
    graph.node_override = override
    anchor = torch.randn(3, 8, 16)
    dense = graph(anchor)
    graph.event_driven_execution = True
    event_driven = graph(anchor)
    assert torch.allclose(dense, event_driven, atol=1e-6, rtol=1e-6)
    assert graph.last_event_executor_stats["ready_events"] > 0


def test_persistent_scheduler_batches_separate_submissions():
    scheduler = PersistentCellEventScheduler()
    modules = nn.ModuleList([nn.Identity()])
    first = CellEventBatch(0, torch.ones(1, 2, 4), torch.tensor([[True, False]]))
    second = CellEventBatch(0, torch.full((1, 2, 4), 2.0), torch.tensor([[True, True]]))
    first_ticket = scheduler.submit(first)
    assert scheduler.flush_ready(modules, modules, min_events=3, max_wait_seconds=10) == {}
    second_ticket = scheduler.submit(second)
    completed = scheduler.flush_ready(
        modules, modules, min_events=3, max_wait_seconds=10
    )
    assert set(completed) == {first_ticket, second_ticket}
    assert completed[first_ticket][0, 0].eq(1).all()
    assert completed[first_ticket][0, 1].eq(0).all()
    assert completed[second_ticket].eq(2).all()
    assert scheduler.pending_stats()["pending_events"] == 0
