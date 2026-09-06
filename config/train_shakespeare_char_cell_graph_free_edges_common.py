# Shared protocol: fixed compute and fixed attention; only connectivity may vary.
eval_interval = 100
eval_iters = 10
log_interval = 50
always_save_checkpoint = True
save_checkpoint_every_eval = False

wandb_log = False
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
bias = False

cell_graph = True
cell_graph_cells_per_step = 4
cell_graph_attention_cells = 0
cell_graph_fixed_attention = True
cell_graph_fixed_active_cells = 2
cell_graph_atom_size = 64
cell_graph_temperature = 1.5
cell_graph_temperature_final = 0.5
cell_graph_temperature_anneal_iters = 2000
cell_graph_node_threshold = 0.5
cell_graph_edge_threshold = 0.5
cell_graph_target_node_ratio = 0.5

# Stage one is about connectivity, not compute savings or enforced utilization.
cell_graph_budget_weight = 0.0
cell_graph_balance_weight = 0.0

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'float16'
compile = False
