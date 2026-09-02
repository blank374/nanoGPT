# Fine token-routed model whose smallest executable MLP bundle is one 64-wide atom.

out_dir = 'out-shakespeare-char-hardware-atom-token-fine'
eval_interval = 100
eval_iters = 20
log_interval = 50
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
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
hardware_atom_choices = [1, 2, 4, 8]
hardware_atom_search_dim = 16
hardware_atom_routing = 'ste'
hardware_atom_route_scope = 'token'
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

compile = False
dtype = 'float16'
