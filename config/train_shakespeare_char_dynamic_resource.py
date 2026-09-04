# Unrestricted 5^4 request-level resource-routing experiment.

out_dir = 'out-shakespeare-char-dynamic-resource'
eval_interval = 100
eval_iters = 40
log_interval = 10
always_save_checkpoint = True
save_checkpoint_every_eval = True

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

dynamic_resource = True
dynamic_resource_widths = [0, 64, 128, 256, 512]
dynamic_resource_skip_mode = 'mlp'
dynamic_resource_routing = 'gumbel'
dynamic_resource_exploration = 0.05
dynamic_resource_temperature = 1.5
dynamic_resource_temperature_final = 0.5
dynamic_resource_temperature_anneal_iters = 1000
dynamic_resource_compute_penalty_max = 0.05
dynamic_resource_compute_penalty_warmup_steps = 200
dynamic_resource_compute_penalty_anneal_steps = 800
dynamic_resource_distill_weight = 0.5
dynamic_resource_distill_temperature = 2.0
dynamic_resource_full_ce_weight = 0.5
dynamic_resource_collapse_threshold = 0.95
dynamic_resource_eval_impl = 'physical'

# All legacy routing mechanisms stay off, preserving a clean baseline.
dynamic_exit = False
enable_dynamic_depth = False
enable_dynamic_width = False
dynamic_width = False
dynamic_mlp = False
free_channel_mlp = False
block_sparse_mlp = False
block_precision_mlp = False
resource_mode_mlp = False
hardware_atom_mlp = False

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'float16'
compile = False
