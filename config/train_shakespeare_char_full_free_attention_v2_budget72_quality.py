# Quality-recovery point: mild joint cost, four Attention anchors and a softer
# final sparsemax temperature to avoid losing useful paths too early.
exec(open('config/train_shakespeare_char_full_free_attention_v2_pareto_common.py').read())

out_dir = 'out-full-free-attention-v2-budget72-quality-seed1337'
cell_graph_attention_cells = 4  # evenly spaced anchors [0, 2, 5, 7]
cell_graph_active_cell_budget = 72.0
cell_graph_temperature_final = 0.5

# Longer full-capacity learning; pressure then ramps from steps 350 to 700.
cell_graph_exploration_anneal_iters = 350
