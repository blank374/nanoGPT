import unittest
from unittest.mock import patch

import torch

from model import CausalSelfAttention, GPT, GPTConfig


class FreeQTest(unittest.TestCase):
    def config(self, selector="binary", projection="identity", **kwargs):
        values = dict(
            vocab_size=32, block_size=8, n_layer=4, n_head=4, n_embd=16,
            dropout=0.0, bias=True, free_q=True,
            free_q_selector=selector,
            free_q_source_projection=projection,
        )
        values.update(kwargs)
        return GPTConfig(**values)

    def test_forward_backward_and_head_shapes(self):
        torch.manual_seed(7)
        model = GPT(self.config())
        idx = torch.randint(0, 32, (2, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        loss.backward()
        router_grad = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.named_parameters()
            if "q_source_router" in name and parameter.grad is not None
        )
        self.assertGreater(router_grad, 0.0)
        for record in model.free_q_route_records():
            self.assertEqual(tuple(record["mask"].shape), (2, 8, 4, 4))
            self.assertTrue((record["mask"].sum(dim=-1) >= 1).all())

    def test_forced_current_matches_standard_q(self):
        torch.manual_seed(5)
        standard = GPT(GPTConfig(
            vocab_size=32, block_size=8, n_layer=4, n_head=4, n_embd=16,
            dropout=0.0, bias=True,
        )).eval()
        free_q = GPT(self.config()).eval()
        common = {name: value for name, value in standard.state_dict().items()
                  if name in free_q.state_dict()}
        free_q.load_state_dict(common, strict=False)
        current_only = []
        for _ in range(4):
            mask = torch.zeros(4, 4, dtype=torch.bool)
            mask[:, 0] = True
            current_only.append(mask)
        free_q.set_free_q_overrides(current_only)
        idx = torch.randint(0, 32, (2, 8))
        expected, _ = standard(idx, idx)
        actual, _ = free_q(idx, idx)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_distinct_source_availability_by_layer(self):
        model = GPT(self.config()).eval()
        model(torch.randint(0, 32, (1, 8)))
        available = [record["available"].tolist()
                     for record in model.free_q_route_records()]
        self.assertEqual(available, [
            [True, False, False, True],
            [True, False, False, True],
            [True, True, False, True],
            [True, True, True, True],
        ])

    def test_sparsemax_is_sparse_normalized_and_nonempty(self):
        torch.manual_seed(11)
        model = GPT(self.config(selector="sparsemax")).eval()
        model(torch.randint(0, 32, (3, 8)))
        saw_zero = False
        for record in model.free_q_route_records():
            weights = record["weights"]
            torch.testing.assert_close(
                weights.sum(dim=-1), torch.ones_like(weights[..., 0])
            )
            self.assertTrue((record["mask"].sum(dim=-1) >= 1).all())
            saw_zero = saw_zero or bool((weights == 0).any().item())
        self.assertTrue(saw_zero)

    def test_independent_linear_source_projection(self):
        model = GPT(self.config(projection="linear"))
        logits, loss = model(
            torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8))
        )
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        loss.backward()
        projected = [parameter for name, parameter in model.named_parameters()
                     if "q_source_projections" in name]
        self.assertTrue(projected)
        self.assertTrue(any(parameter.grad is not None for parameter in projected))

    def test_fixed_and_sequence_shuffle_overrides(self):
        model = GPT(self.config()).eval()
        idx = torch.randint(0, 32, (2, 8))
        model(idx)
        learned = [record["mask"].clone() for record in model.free_q_route_records()]
        fixed = []
        for layer, record in enumerate(model.free_q_route_records()):
            mask = torch.zeros(4, 4, dtype=torch.bool)
            mask[:, 0] = True
            if not record["available"][0]:
                self.fail(f"current source unavailable at layer {layer}")
            fixed.append(mask)
        model.set_free_q_overrides(fixed)
        model(idx)
        for record in model.free_q_route_records():
            self.assertTrue(record["mask"][..., 0].all())
            self.assertEqual(record["mask"].sum().item(), 2 * 8 * 4)

        shuffled = [mask.flip(0) for mask in learned]
        model.set_free_q_overrides(shuffled)
        model(idx)
        for actual, expected in zip(model.free_q_route_records(), shuffled):
            self.assertTrue(torch.equal(actual["mask"], expected))
        model.set_free_q_overrides(None)

    def test_kv_do_not_depend_on_q_sources(self):
        config = self.config()
        attention = CausalSelfAttention(config).eval()
        x = torch.randn(2, 8, 16)
        current = torch.randn_like(x)
        history_a = torch.randn_like(x)
        history_b = torch.randn_like(x)
        anchor = torch.randn_like(x)
        captured = []

        def fake_attention(q, k, v, **kwargs):
            captured.append((q.detach().clone(), k.detach().clone(), v.detach().clone()))
            return torch.zeros_like(q)

        override = torch.zeros(4, 4, dtype=torch.bool)
        override[:, 1] = True
        attention.free_q_override = override
        with patch("torch.nn.functional.scaled_dot_product_attention", fake_attention):
            attention(x, [current, history_a, None, anchor])
            attention(x, [current, history_b, None, anchor])
        self.assertFalse(torch.equal(captured[0][0], captured[1][0]))
        torch.testing.assert_close(captured[0][1], captured[1][1])
        torch.testing.assert_close(captured[0][2], captured[1][2])

    def test_token_routing_preserves_causality(self):
        torch.manual_seed(17)
        model = GPT(self.config(selector="sparsemax")).eval()
        first = torch.randint(0, 32, (1, 8))
        second = first.clone()
        second[:, 5:] = torch.randint(0, 32, (1, 3))
        logits_a, _ = model(first, first)
        logits_b, _ = model(second, second)
        torch.testing.assert_close(logits_a[:, :5], logits_b[:, :5])

    def test_free_q_rejects_other_dynamic_variables(self):
        with self.assertRaisesRegex(AssertionError, "Free-Q must be isolated"):
            GPT(self.config(dynamic_width=True))


if __name__ == "__main__":
    unittest.main()
