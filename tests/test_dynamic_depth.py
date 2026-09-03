import unittest

import torch

from model import GPT, GPTConfig


class DynamicDepthTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(GPTConfig(
            vocab_size=32, block_size=12, n_layer=4, n_head=2, n_embd=16,
            dropout=0.0, enable_dynamic_depth=True,
            dynamic_depth_choices=[2, 3, 4],
        ))
        self.x = torch.randint(0, 32, (6, 12))

    @torch.no_grad()
    def manual_prefix(self, x, depth):
        pos = torch.arange(x.size(1))
        hidden = self.model.transformer.drop(
            self.model.transformer.wte(x) + self.model.transformer.wpe(pos)
        )
        for block in self.model.transformer.h[:depth]:
            hidden = block(hidden)
        return self.model.lm_head(self.model.transformer.ln_f(hidden[:, -1:, :]))

    def test_forced_prefix_matches_manual_prefix(self):
        self.model.eval()
        for depth in (2, 3, 4):
            actual, _ = self.model.forward_dynamic_depth(
                self.x, forced_depth=depth, record_stats=False
            )
            torch.testing.assert_close(actual, self.manual_prefix(self.x, depth))

    def test_cached_mixed_plan_matches_individual_requests(self):
        self.model.eval()
        depths = torch.tensor([2, 3, 4, 2, 4, 3])
        plan = self.model.build_dynamic_depth_plan(depths)
        grouped, _ = self.model.forward_dynamic_depth(
            self.x, route_plan=plan, record_stats=False
        )
        individual = torch.cat([
            self.model.forward_dynamic_depth(
                self.x[i:i + 1], forced_depth=int(depths[i]), record_stats=False
            )[0]
            for i in range(self.x.size(0))
        ], dim=0)
        torch.testing.assert_close(grouped, individual)

    def test_training_loss_backpropagates_to_router(self):
        self.model.train()
        targets = torch.randint(0, 32, self.x.shape)
        _, loss = self.model(self.x, targets)
        loss.backward()
        self.assertIsNotNone(self.model.depth_router.proj.weight.grad)
        self.assertGreater(self.model.depth_router.proj.weight.grad.abs().sum().item(), 0.0)
        self.assertEqual(set(self.model.last_dynamic_depth_logits), {2, 3, 4})


if __name__ == "__main__":
    unittest.main()

