# Full-Free Natural: the only optimized objective is language-model CE.
exec(open('config/train_shakespeare_char_full_free_common.py').read())
out_dir = 'out-full-free-natural-seed1337'
cell_graph_budget_mode = 'none'
cell_graph_dual_init = 0.0

