# Full-Free Budget: one dual-controlled average active-Cell constraint.
exec(open('config/train_shakespeare_char_full_free_common.py').read())
out_dir = 'out-full-free-budget16-seed1337'
cell_graph_budget_mode = 'dual_active_cells'
cell_graph_active_cell_budget = 16.0
cell_graph_dual_lr = 0.001
cell_graph_dual_init = 0.0

