# train a tiny character-level Shakespeare Width x Bit block-precision MLP model
# routes contiguous 16-channel blocks and fake-quantizes active block weights to 2/4/8/16 bit

out_dir = 'out-shakespeare-char-block-precision'
eval_interval = 100
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
save_checkpoint_every_eval = True

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-block-precision'

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
block_sparse_mlp = False

block_precision_mlp = True
block_precision_block_size = 16
block_precision_bit_choices = [2, 4, 8, 16]
block_precision_routing = "ste"
block_precision_cost_weight = 0.01
block_precision_temperature = 2.0
block_precision_temperature_final = 0.5
block_precision_temperature_anneal_iters = 1000
block_precision_width_temperature = 1.0
block_precision_sliced_eval = True
block_precision_eval_impl = "grouped"

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on macbook also add
# device = 'cpu'
# compile = False
