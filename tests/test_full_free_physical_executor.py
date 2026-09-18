import unittest

import torch

from model import GPT, GPTConfig
from experiments.full_free_physical_executor import physical_executor


class FullFreePhysicalExecutorTest(unittest.TestCase):
    def test_token_queue_is_semantics_preserving_and_executes_only_active_pairs(self):
        torch.manual_seed(17)
        config = GPTConfig(
            vocab_size=32,
            block_size=8,
            n_layer=3,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            cell_graph=True,
            cell_graph_mode="full_free",
            cell_graph_cells_per_step=2,
            cell_graph_fixed_attention=True,
            cell_graph_attention_cells=0,
            cell_graph_atom_size=8,
            cell_graph_router_hidden=8,
            cell_graph_lookback_steps=2,
            cell_graph_halt=False,
        )
        model = GPT(config).eval()
        # Freeze a deterministic sparse router without using graph overrides;
        # the production executor intentionally rejects replay/override paths.
        with torch.no_grad():
            router = model.cell_graph.router
            router.node_keys.zero_()
            router.node_bias[:, :2].fill_(-3.0)
            router.node_bias[:, 2].fill_(3.0)
        idx = torch.randint(0, config.vocab_size, (2, config.block_size))
        targets = torch.randint(0, config.vocab_size, idx.shape)

        with torch.no_grad():
            dense_logits, _ = model(idx, targets)
            dense_mask = model.cell_graph.last_node_mask.clone()
            with physical_executor(model, mode="token_queue", min_token_positions=0):
                sparse_logits, _ = model(idx, targets)
                stats = dict(model.cell_graph.last_physical_executor_stats)

        torch.testing.assert_close(dense_logits, sparse_logits, atol=1e-6, rtol=1e-6)
        self.assertEqual(stats["active_token_cell_pairs"], int(dense_mask.sum()))
        self.assertLess(stats["active_pair_ratio"], 1.0)
        self.assertGreater(stats["empty_cell_queues"], 0)


if __name__ == "__main__":
    unittest.main()
