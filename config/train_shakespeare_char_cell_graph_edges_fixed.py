# Run A: canonical fixed graph. Every active Cell reads the current backbone state.
exec(open('config/train_shakespeare_char_cell_graph_free_edges_common.py').read())
out_dir = 'out-cell-graph-edges-fixed-seed1337'
cell_graph_edge_mode = 'fixed'
cell_graph_edge_cost_weight = 0.0
