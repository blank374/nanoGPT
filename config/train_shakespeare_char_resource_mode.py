# train a tiny character-level Shakespeare batch-friendly resource-mode MLP model
# each token chooses one executable mode: (contiguous blocks, shared bit)

out_dir = 'out-shakespeare-char-resource-mode'
eval_interval = 100
eval_iters = 20
log_interval = 50
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-resource-mode'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 64

n_layer = 2
n_head = 4
n_embd = 64
dropout = 0.1

dynamic_exit = False
dynamic_mlp = False
dynamic_width = False
free_channel_mlp = False
block_sparse_mlp = False
block_precision_mlp = False

resource_mode_mlp = True
resource_mode_block_size = 16
resource_mode_choices = [(2, 2), (4, 2), (8, 4), (12, 8), (16, 16)]
resource_mode_routing = "ste"
resource_mode_cost_weight = 0.03
resource_mode_temperature = 2.0
resource_mode_temperature_final = 0.5
resource_mode_temperature_anneal_iters = 1000
resource_mode_sliced_eval = True
resource_mode_eval_impl = "grouped"

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
