import unittest

import torch

from model import GPT, GPTConfig


class DynamicCellGraphTest(unittest.TestCase):
    def config(self, **kwargs):
        values = dict(
            vocab_size=32,
            block_size=8,
            n_layer=3,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            cell_graph=True,
            cell_graph_cells_per_step=2,
            cell_graph_attention_cells=1,
            cell_graph_atom_size=8,
        )
        values.update(kwargs)
        return GPTConfig(**values)

    def test_forward_backward_reaches_unified_router(self):
        torch.manual_seed(7)
        model = GPT(self.config())
        idx = torch.randint(0, 32, (2, 8))
        targets = torch.randint(0, 32, (2, 8))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.cell_graph.router.node_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.cell_graph.router.edge_proj.weight.grad.abs().sum().item(), 0.0)

    def test_edges_only_point_to_strictly_earlier_steps(self):
        model = GPT(self.config()).eval()
        model(torch.randint(0, 32, (2, 8)))
        records = model.cell_graph_route_records()
        edges = records["edge_mask"]
        cells_per_step = model.config.cell_graph_cells_per_step
        for node in range(model.cell_graph.num_cells):
            step = node // cells_per_step
            first_illegal_source = 1 + step * cells_per_step
            self.assertFalse(edges[:, :, node, first_illegal_source:].any())
            self.assertTrue((edges[:, :, node].sum(dim=-1) >= 1).all())

    def test_router_and_attention_preserve_causality(self):
        torch.manual_seed(11)
        model = GPT(self.config()).eval()
        first = torch.randint(0, 32, (1, 8))
        second = first.clone()
        second[:, 5:] = torch.randint(0, 32, (1, 3))
        logits_a, _ = model(first, first)
        logits_b, _ = model(second, second)
        torch.testing.assert_close(logits_a[:, :5], logits_b[:, :5])

    def test_all_nodes_off_is_well_defined_skip_graph(self):
        model = GPT(self.config()).eval()
        with torch.no_grad():
            model.cell_graph.router.node_proj.weight.zero_()
            model.cell_graph.router.node_proj.bias.fill_(-20.0)
        idx = torch.randint(0, 32, (2, 8))
        actual, _ = model(idx, idx)
        pos = torch.arange(idx.size(1))
        anchor = model.transformer.drop(
            model.transformer.wte(idx) + model.transformer.wpe(pos)
        )
        expected = model.lm_head(model.transformer.ln_f(anchor))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.last_cell_graph_stats["mean_active_cells"], 0.0)
        self.assertEqual(model.last_cell_graph_stats["mean_depth"], 0.0)

    def test_graph_stats_report_width_depth_and_fanin(self):
        model = GPT(self.config()).eval()
        with torch.no_grad():
            model.cell_graph.router.node_proj.weight.zero_()
            model.cell_graph.router.node_proj.bias.fill_(20.0)
            model.cell_graph.router.edge_proj.weight.zero_()
            model.cell_graph.router.edge_proj.bias.fill_(20.0)
        idx = torch.randint(0, 32, (1, 8))
        model(idx, idx)
        stats = model.last_cell_graph_stats
        self.assertEqual(stats["mean_active_cells"], 6.0)
        self.assertEqual(stats["mean_width"], 2.0)
        self.assertEqual(stats["mean_depth"], 3.0)
        self.assertGreater(stats["mean_fanin"], 1.0)

    def test_cell_graph_rejects_legacy_routers(self):
        with self.assertRaisesRegex(AssertionError, "standalone architecture"):
            GPT(self.config(dynamic_width=True))

    def test_free_edge_protocol_fixes_nodes_and_attention(self):
        model = GPT(self.config(
            cell_graph_fixed_attention=True,
            cell_graph_fixed_active_cells=1,
            cell_graph_attention_cells=0,
        )).eval()
        model(torch.randint(0, 32, (2, 8)))
        mask = model.cell_graph_route_records()["node_mask"]
        expected = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected.view(1, 1, -1).expand_as(mask)))
        self.assertEqual(len(model.cell_graph.fixed_attentions), 3)
        self.assertTrue(all(
            isinstance(cell, type(model.cell_graph.cells[0]))
            for cell in model.cell_graph.cells
        ))

    def test_fixed_and_replayed_edge_overrides(self):
        model = GPT(self.config(
            cell_graph_fixed_attention=True,
            cell_graph_fixed_active_cells=1,
            cell_graph_attention_cells=0,
        )).eval()
        idx = torch.randint(0, 32, (2, 8))
        model(idx)
        learned = model.cell_graph_route_records()["edge_mask"].clone()
        fixed = torch.zeros(model.cell_graph.num_cells, 1 + model.cell_graph.num_cells,
                            dtype=torch.bool)
        fixed[:, 0] = True
        model.set_cell_graph_edge_override(fixed)
        model(idx)
        self.assertTrue(model.cell_graph_route_records()["edge_mask"][..., 0].all())
        model.set_cell_graph_edge_override(learned)
        model(idx)
        torch.testing.assert_close(model.cell_graph_route_records()["edge_mask"], learned)
        model.set_cell_graph_edge_override(None)

    def full_free_config(self, **kwargs):
        values = dict(
            cell_graph_mode="full_free",
            cell_graph_fixed_attention=True,
            cell_graph_attention_cells=0,
            cell_graph_fixed_active_cells=0,
            cell_graph_edge_mode="learned",
            cell_graph_router_hidden=8,
            cell_graph_lookback_steps=2,
            cell_graph_budget_weight=0.0,
            cell_graph_edge_cost_weight=0.0,
            cell_graph_balance_weight=0.0,
        )
        values.update(kwargs)
        return self.config(**values)

    def test_full_free_sparsemax_has_exact_zeros(self):
        from model import FullFreeGraphRouter
        logits = torch.tensor([[1.4, 0.0, -1.2, 1.1]])
        weights = FullFreeGraphRouter.sparsemax(logits)
        self.assertEqual(int((weights > 0).sum()), 2)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        entmax = FullFreeGraphRouter.entmax15(torch.tensor([[3.0, 0.0, -3.0]]))
        self.assertAlmostEqual(entmax.sum().item(), 1.0, places=6)
        self.assertGreater(int((entmax == 0).sum()), 0)

    def test_full_free_variable_width_empty_step_and_true_depth(self):
        model = GPT(self.full_free_config()).eval()
        # S0: one Cell, S1: two Cells, S2: empty. The only long path is C0->C2.
        nodes = torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.float32)
        edges = torch.zeros(6, 7)
        edges[:, 0] = 1
        edges[2].zero_()
        edges[2, 1] = 1
        model.set_cell_graph_overrides(nodes, edges)
        idx = torch.randint(0, 32, (2, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(loss))
        stats = model.last_cell_graph_stats
        self.assertEqual(stats["step_widths"], [1.0, 2.0, 0.0])
        self.assertEqual(stats["mean_active_cells"], 3.0)
        self.assertEqual(stats["mean_depth"], 2.0)
        self.assertEqual(stats["longest_path"], 2.0)

    def test_full_free_skip_competitor_can_naturally_zero_every_cell(self):
        model = GPT(self.full_free_config()).eval()
        with torch.no_grad():
            model.cell_graph.router.node_keys.zero_()
            model.cell_graph.router.node_bias[:, :2].fill_(-3.0)
            model.cell_graph.router.node_bias[:, 2].fill_(3.0)
        model(torch.randint(0, 32, (1, 8)))
        stats = model.last_cell_graph_stats
        self.assertEqual(stats["mean_active_cells"], 0.0)
        self.assertEqual(stats["mean_depth"], 0.0)
        self.assertEqual(stats["empty_step_fraction"], 1.0)

    def test_full_free_uses_bounded_lookback(self):
        model = GPT(self.full_free_config(cell_graph_lookback_steps=1)).eval()
        model(torch.randint(0, 32, (1, 8)))
        valid = model.cell_graph._valid_source_mask(torch.device("cpu"))
        # Step 2 may see current + Step 1, but not Step 0.
        self.assertFalse(valid[4, 1:3].any())
        self.assertTrue(valid[4, 3:5].all())
        self.assertEqual(int(valid.sum()), 14)

    def test_full_free_router_gradients_and_causality(self):
        torch.manual_seed(19)
        model = GPT(self.full_free_config())
        idx = torch.randint(0, 32, (2, 8))
        _, loss = model(idx, idx)
        loss.backward()
        self.assertGreater(model.cell_graph.router.context_down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.cell_graph.router.node_keys.grad.abs().sum().item(), 0)
        self.assertGreater(model.cell_graph.router.edge_local_keys.grad.abs().sum().item(), 0)
        model.eval()
        first = torch.randint(0, 32, (1, 8))
        second = first.clone()
        second[:, 5:] = torch.randint(0, 32, (1, 3))
        logits_a, _ = model(first, first)
        logits_b, _ = model(second, second)
        torch.testing.assert_close(logits_a[:, :5], logits_b[:, :5])

    def test_full_free_dual_budget_updates_without_balance_loss(self):
        model = GPT(self.full_free_config(
            cell_graph_budget_mode="dual_active_cells",
            cell_graph_active_cell_budget=0.0,
            cell_graph_dual_lr=0.1,
        )).eval()
        model(torch.randint(0, 32, (1, 8)))
        before = model.config.cell_graph_dual_value
        after = model.update_cell_graph_dual()
        self.assertGreater(after, before)
        self.assertEqual(model.config.cell_graph_balance_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
