"""Forward/backward and physical-skip checks for dynamic resource routing."""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig


def make_model(skip_mode="mlp"):
    return GPT(GPTConfig(
        block_size=16, vocab_size=65, n_layer=4, n_head=4, n_embd=128,
        dropout=0.0, bias=False, dynamic_resource=True,
        dynamic_resource_widths=[0, 64, 128, 256, 512],
        dynamic_resource_skip_mode=skip_mode,
        dynamic_resource_routing="ste",
    ))


def main():
    torch.manual_seed(7)
    x = torch.randint(0, 65, (3, 12))
    y = torch.randint(0, 65, (3, 12))
    paths = [
        [64, 64, 0, 0],
        [128, 128, 256, 0],
        [512, 512, 512, 512],
    ]

    model = make_model("mlp").train()
    for path in paths:
        model.zero_grad(set_to_none=True)
        logits = model._forward_dynamic_resource_logits(x, forced_path=path, all_logits=True)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 65), y.reshape(-1))
        loss.backward()
        assert torch.isfinite(loss)
        assert model.transformer.h[0].mlp.c_fc.weight.grad is not None
        print(f"PASS forward/backward path={path}, loss={loss.item():.4f}")

    model.zero_grad(set_to_none=True)
    _, total_loss = model(x, y)
    total_loss.backward()
    router_grad = model.resource_router.proj.weight.grad
    assert router_grad is not None and torch.isfinite(router_grad).all()
    stats = model.last_dynamic_resource_stats
    assert stats["theoretical_paths"] == 625
    assert 1 <= stats["observed_paths"] <= x.size(0)
    print(f"PASS automatic router gradient and 625-path statistics, loss={total_loss.item():.4f}")

    model.eval()
    path = paths[1]
    model.config.dynamic_resource_eval_impl = "research"
    dense = model._forward_dynamic_resource_logits(x, forced_path=path, all_logits=True)
    model.config.dynamic_resource_eval_impl = "physical"
    physical = model._forward_dynamic_resource_logits(x, forced_path=path, all_logits=True)
    torch.testing.assert_close(dense, physical, rtol=1e-5, atol=1e-6)
    print("PASS research/physical equivalence for MLP Skip")

    block_model = make_model("block").eval()
    logits = block_model._forward_dynamic_resource_logits(
        x, forced_path=[0, 0, 0, 0], all_logits=True
    )
    assert logits.shape == (3, 12, 65)
    print("PASS whole-Block Skip physical forward")


if __name__ == "__main__":
    main()
