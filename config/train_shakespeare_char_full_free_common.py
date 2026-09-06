# Shared Full-Free v1 envelope. Width/depth/fan-in are graph-derived.
eval_interval = 100
eval_iters = 10
log_interval = 25
always_save_checkpoint = True
save_checkpoint_every_eval = True

wandb_log = False
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 128

n_layer = 8
n_head = 4
n_embd = 128
dropout = 0.1
bias = False

cell_graph = True
cell_graph_mode = 'full_free'
cell_graph_cells_per_step = 4
cell_graph_attention_cells = 0
cell_graph_fixed_attention = True
cell_graph_fixed_active_cells = 0
cell_graph_edge_mode = 'learned'
cell_graph_atom_size = 64
cell_graph_router_hidden = 32
cell_graph_lookback_steps = 3
cell_graph_input_projection = 'identity'
cell_graph_node_selector = 'sparsemax'
cell_graph_edge_selector = 'sparsemax'
cell_graph_halt = False

cell_graph_temperature = 1.5
cell_graph_temperature_final = 0.25
cell_graph_temperature_anneal_iters = 1000
cell_graph_exploration = 0.0
cell_graph_exploration_final = 0.0
cell_graph_exploration_anneal_iters = 0

# No independent width/depth/edge/balance objectives in Full-Free.
cell_graph_target_node_ratio = 0.5
cell_graph_budget_weight = 0.0
cell_graph_edge_cost_weight = 0.0
cell_graph_balance_weight = 0.0

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'float16'
compile = False

