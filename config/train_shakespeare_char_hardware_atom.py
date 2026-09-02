# Train a retrieval-routed MLP whose executable units are GPU-aligned prefixes.
# The high-dimensional hidden state is compressed to a 16-D address, then each
# request/sequence selects a 128/256/512-channel MLP per layer. Training may be
# expensive; eval only computes the selected prefix and can cache the route.

out_dir = 'out-shakespeare-char-hardware-atom-sequence'
eval_interval = 100
eval_iters = 20
log_interval = 50
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-hardware-atom-sequence'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 64

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

dynamic_exit = False
dynamic_mlp = False
dynamic_width = False
free_channel_mlp = False
block_sparse_mlp = False
block_precision_mlp = False
resource_mode_mlp = False

hardware_atom_mlp = True
hardware_atom_size = 64
hardware_atom_choices = [2, 4, 8]
hardware_atom_search_dim = 16
hardware_atom_routing = 'ste'
hardware_atom_route_scope = 'sequence'
hardware_atom_cost_weight = 0.03
hardware_atom_temperature = 2.0
hardware_atom_temperature_final = 0.5
hardware_atom_temperature_anneal_iters = 1000
hardware_atom_eval_impl = 'grouped'

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# GTX 1650 supports CUDA execution but not a native PyTorch packed-INT4 path.
# This experiment first validates real width skipping with aligned FP16 GEMMs.
compile = False
dtype = 'float16'
