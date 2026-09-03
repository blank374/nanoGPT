# Stage 1: request-level dynamic depth only. MLP width is fixed at 4*128=512.

out_dir = 'out-shakespeare-char-dynamic-depth-oracle'
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

# Independent switches: this experiment intentionally keeps width fixed.
enable_dynamic_depth = True
enable_dynamic_width = False
dynamic_width = False
dynamic_mlp = False
free_channel_mlp = False
block_sparse_mlp = False
block_precision_mlp = False
resource_mode_mlp = False
hardware_atom_mlp = False

dynamic_depth_choices = [2, 3, 4]
dynamic_depth_temperature = 1.5
dynamic_depth_temperature_final = 0.6
dynamic_depth_temperature_anneal_iters = 800
dynamic_depth_early_ce_weight = 0.30
dynamic_depth_distill_weight = 0.30
dynamic_depth_distill_temperature = 2.0
dynamic_depth_router_quality_weight = 1.0
dynamic_depth_compute_weight = 0.08
dynamic_depth_compute_warmup_iters = 200
dynamic_depth_compute_anneal_iters = 600
dynamic_depth_entropy_weight = 0.01
dynamic_depth_full_exploration = 0.05
dynamic_depth_oracle_margin = 0.02
dynamic_depth_inference_cost_bias = 0.0

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cuda'
dtype = 'float16'
compile = False
