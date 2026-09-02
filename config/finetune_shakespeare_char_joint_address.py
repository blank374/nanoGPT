# Jointly consolidate layer-wise atom routes into one early associative address.
# Start by copying the best sequence-routed checkpoint into this output folder.

out_dir = 'out-shakespeare-char-hardware-atom-joint-address'
init_from = 'resume'
resume_optimizer = False
eval_interval = 100
eval_iters = 20
log_interval = 25
always_save_checkpoint = True
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
hardware_atom_choices = [2, 4, 8]
hardware_atom_search_dim = 16
hardware_atom_routing = 'ste'
hardware_atom_route_scope = 'sequence'
hardware_atom_cost_weight = 0.03
hardware_atom_temperature = 0.5
hardware_atom_temperature_final = 0.5
hardware_atom_temperature_anneal_iters = 0
hardware_atom_eval_impl = 'grouped'

hardware_atom_direct_address = True
hardware_atom_address_loss_weight = 0.20
hardware_atom_address_agreement_weight = 0.05
hardware_atom_address_task_weight = 0.30

# The first consolidation stage reached iteration 1400. Continue for another
# 500 steps with the address-selected language task in the objective.
learning_rate = 3e-4
decay_lr = False
max_iters = 1900
warmup_iters = 0
beta2 = 0.99

compile = False
dtype = 'float16'
