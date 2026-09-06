# Quality-first Attention v2 Pareto protocol.
exec(open('config/train_shakespeare_char_full_free_common.py').read())

cell_graph_attention_cells = 3  # mandatory anchors [0, 4, 7]
cell_graph_budget_mode = 'dual_active_cells'
cell_graph_dual_lr = 0.001
cell_graph_dual_init = 0.0

# In the dedicated v2 runner this is also the joint-cost dual warmup and ramp
# length. Exploration itself stays disabled (0 -> 0). The first 250 steps learn
# language modeling without compute pressure; steps 250-500 ramp the pressure.
cell_graph_exploration = 0.0
cell_graph_exploration_final = 0.0
cell_graph_exploration_anneal_iters = 250

seed = 1337

