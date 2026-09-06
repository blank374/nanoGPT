# Run B: dynamic learned edges, with no structural cost.
exec(open('config/train_shakespeare_char_cell_graph_free_edges_common.py').read())
out_dir = 'out-cell-graph-edges-learned-seed1337'
cell_graph_edge_mode = 'learned'
cell_graph_edge_cost_weight = 0.0
