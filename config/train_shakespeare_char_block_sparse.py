# train a tiny character-level Shakespeare Block-Sparse MLP model
# routes 16-channel blocks and can use true sliced Linear at eval/benchmark time

out_dir = 'out-shakespeare-char-block-sparse'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-block-sparse'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2

dynamic_exit = False
dynamic_mlp = False
dynamic_width = False
free_channel_mlp = False

block_sparse_mlp = True
block_sparse_block_size = 16
block_sparse_routing = "ste"
block_sparse_threshold = 0.5
block_sparse_target_ratio = 0.4
block_sparse_budget_weight = 1.0
block_sparse_cost_weight = 0.0
block_sparse_temperature = 2.0
block_sparse_temperature_final = 0.5
block_sparse_temperature_anneal_iters = 1000
block_sparse_sliced_eval = True
block_sparse_eval_impl = "grouped"

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
