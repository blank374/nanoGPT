"""Quality-constrained Full-Free pilot.

This is the first experiment for the "minimum necessary computation graph"
objective. The final LM loss remains active, while a dual active-Cell budget
pushes the router toward a smaller graph and an auxiliary local free-energy
loss makes every intermediate graph state predictive instead of rewarding a
router that simply turns everything off.

The budget is deliberately a pilot value, not a claim about the final
speedup. Real latency must still be measured with
``experiments/full_free_physical_executor.py``.
"""

exec(open('config/train_shakespeare_char_full_free_common.py').read())

out_dir = 'out-full-free-quality-seed1337'
seed = 1337

# 32 Cells are available per [batch, token] at the 8x4 envelope. Start at
# roughly 40% active Cell support and sweep this value in follow-up runs.
cell_graph_budget_mode = 'dual_active_cells'
cell_graph_active_cell_budget = 12.8
cell_graph_dual_lr = 0.05
cell_graph_dual_init = 0.0
cell_graph_target_node_ratio = 0.40
cell_graph_budget_weight = 0.05

# Local predictive free energy: intermediate states must reduce next-token
# error, with a small state-complexity term. The final task CE is still the
# primary objective in GPT.forward.
cell_graph_free_energy_enabled = True
cell_graph_free_energy_prediction_weight = 1.0
cell_graph_free_energy_complexity_weight = 0.01
cell_graph_free_energy_loss_weight = 0.05

# Keep routing exploratory during the early quality-recovery phase.
cell_graph_exploration = 0.05
cell_graph_exploration_final = 0.0
cell_graph_exploration_anneal_iters = 350
