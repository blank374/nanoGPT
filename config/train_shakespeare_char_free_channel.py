# train a tiny character-level Shakespeare Free Channel MLP model
# each MLP hidden channel independently gates itself per token

out_dir = 'out-shakespeare-char-free-channel'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-free-channel'

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

free_channel_mlp = True
free_channel_routing = "ste"
free_channel_threshold = 0.5
free_channel_target_ratio = 0.4
free_channel_budget_weight = 0.01
free_channel_cost_weight = 0.0
free_channel_temperature = 2.0
free_channel_temperature_final = 0.5
free_channel_temperature_anneal_iters = 1000
free_channel_eval_impl = "dense_mask"
free_channel_prefix_granularity = 64

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
