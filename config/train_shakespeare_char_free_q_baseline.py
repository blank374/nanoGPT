# Matched standard-Q baseline for the Free-Fan-In Query experiment.

out_dir = 'out-free-q-baseline'
eval_interval = 100
eval_iters = 40
log_interval = 10
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

free_q = False
free_q_selector = 'binary'
free_q_source_projection = 'identity'
free_q_threshold = 0.5
free_q_temperature = 1.0

dynamic_exit = False
enable_dynamic_depth = False
enable_dynamic_width = False
dynamic_resource = False
dynamic_mlp = False
dynamic_width = False
free_channel_mlp = False
block_sparse_mlp = False
block_precision_mlp = False
resource_mode_mlp = False
hardware_atom_mlp = False

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'float16'
compile = False
