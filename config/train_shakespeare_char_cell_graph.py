# Small end-to-end Dynamic Cell Graph experiment.
out_dir = 'out-shakespeare-char-cell-graph'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False

wandb_log = False
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4                 # graph steps
n_head = 4
n_embd = 128
dropout = 0.1
bias = False

cell_graph = True
cell_graph_cells_per_step = 4
cell_graph_attention_cells = 1
cell_graph_atom_size = 64
cell_graph_temperature = 1.5
cell_graph_temperature_final = 0.5
cell_graph_temperature_anneal_iters = 2000
cell_graph_node_threshold = 0.5
cell_graph_edge_threshold = 0.5
cell_graph_target_node_ratio = 0.5
cell_graph_budget_weight = 0.1
cell_graph_edge_cost_weight = 0.01
cell_graph_balance_weight = 0.01

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'bfloat16'
compile = False
