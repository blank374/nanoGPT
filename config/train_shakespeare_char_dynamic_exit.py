# train a miniature character-level Shakespeare dynamic-exit model
# good for a first Fast/Slow Path research smoke test on CPU/MPS/CUDA

out_dir = 'out-shakespeare-char-dynamic-exit'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-dynamic-exit'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

dynamic_exit = True
exit_layers = [2, 4]
confidence_method = "max_prob"
confidence_threshold = 0.95
entropy_threshold = 0.5
early_exit_loss_weight = 0.3

use_distillation = False
distillation_temperature = 2.0
distillation_beta = 0.5

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
