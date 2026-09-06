# Run C: dynamic learned edges with an edge budget.
exec(open('config/train_shakespeare_char_cell_graph_free_edges_common.py').read())
out_dir = 'out-cell-graph-edges-budget-seed1337'
cell_graph_edge_mode = 'learned'
cell_graph_edge_cost_weight = 0.01
