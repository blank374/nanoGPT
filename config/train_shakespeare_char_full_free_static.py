# Train a static graph with the same Cell/Attention parameter envelope.
# Run full_free_cell_graph_analysis.py first to create matched_static_graph.npz.
exec(open('config/train_shakespeare_char_full_free_common.py').read())
out_dir = 'out-full-free-static-seed1337'
cell_graph_budget_mode = 'none'
cell_graph_static_graph_path = 'out-full-free-natural-seed1337/matched_static_graph.npz'

