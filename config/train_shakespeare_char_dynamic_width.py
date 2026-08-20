# train a tiny character-level Shakespeare Adaptive Width MLP model
# isolates nested per-token MLP width routing from fast/slow MLP and early exit

out_dir = 'out-shakespeare-char-dynamic-width'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-dynamic-width'

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

dynamic_width = True
dynamic_width_ratios = [0.5, 1.0, 2.0, 4.0]
dynamic_width_cost_weight = 0.01
dynamic_width_hard_eval = True
dynamic_width_temperature = 1.0
dynamic_width_temperature_final = 1.0
dynamic_width_temperature_anneal_iters = 0
dynamic_width_routing = "soft"
dynamic_width_hard_loss_weight = 0.0
dynamic_width_entropy_weight = 0.0
dynamic_width_sliced_eval = True

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
