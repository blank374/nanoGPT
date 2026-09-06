# Full-Free Attention v2 pilot.  Run only through
# experiments/train_full_free_attention_v2.py so frozen v1 remains untouched.
exec(open('config/train_shakespeare_char_full_free_common.py').read())

out_dir = 'out-full-free-attention-v2-budget8-seed1337'
cell_graph_attention_cells = 3  # mandatory Attention anchors: [0, 4, 7]

cell_graph_budget_mode = 'dual_active_cells'
# Joint compute target in Cell-equivalent MAC units.  At block_size=128 one
# Attention costs about six Cells, so 32 units roughly means 3 mandatory
# Attention anchors plus 14 Cell atoms, or 4 Attention plus 8 Cell atoms.
cell_graph_active_cell_budget = 32.0
cell_graph_dual_lr = 0.001
cell_graph_dual_init = 0.0

seed = 1337
