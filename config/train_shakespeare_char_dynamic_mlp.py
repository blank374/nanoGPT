# train a miniature character-level Shakespeare Fast/Slow MLP model
# isolates intra-block dynamic parameter capacity from early exit

out_dir = 'out-shakespeare-char-dynamic-mlp'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-dynamic-mlp'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

dynamic_exit = False

dynamic_mlp = True
dynamic_mlp_fast_ratio = 1.0
dynamic_mlp_slow_ratio = 3.0
dynamic_mlp_cost_weight = 0.01
dynamic_mlp_threshold = 0.5
dynamic_mlp_hard_eval = True

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
