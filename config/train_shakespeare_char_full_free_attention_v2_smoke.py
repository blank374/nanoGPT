# Full-Free Attention v2 smoke protocol.  The dedicated v2 runner interprets
# cell_graph_attention_cells=3 as mandatory Attention anchors [0, 4, 7].
exec(open('config/train_shakespeare_char_full_free_common.py').read())

out_dir = 'out-full-free-attention-v2-smoke'
cell_graph_attention_cells = 3

# A constrained Cell budget makes optional steps observable during a short run.
# Attention has no independent router: a non-anchor Attention is active exactly
# when the corresponding graph step contains at least one active Compute Cell.
cell_graph_budget_mode = 'dual_active_cells'
# Joint Cell+Attention target, expressed in Cell-equivalent MAC units.
cell_graph_active_cell_budget = 32.0
cell_graph_dual_lr = 0.001
cell_graph_dual_init = 0.0

max_iters = 50
lr_decay_iters = 50
warmup_iters = 5
eval_interval = 25
eval_iters = 5
log_interval = 5
always_save_checkpoint = True
save_checkpoint_every_eval = False
