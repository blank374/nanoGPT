"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
save_checkpoint_every_eval = False # if True, also save ckpt_iter_{iter_num}.pt for training dynamics analysis
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# dynamic fast/slow path transformer
dynamic_exit = False
exit_layers = [3, 6, 9]
confidence_method = "max_prob" # "max_prob" or "entropy"
confidence_threshold = 0.95
entropy_threshold = 0.5
early_exit_loss_weight = 0.3
use_distillation = False
distillation_temperature = 2.0
distillation_beta = 0.5
# dynamic MLP capacity routing
dynamic_mlp = False
dynamic_mlp_fast_ratio = 1.0
dynamic_mlp_slow_ratio = 3.0
dynamic_mlp_cost_weight = 0.01
dynamic_mlp_threshold = 0.5
dynamic_mlp_hard_eval = True
# nested dynamic MLP width routing
dynamic_width = False
dynamic_width_ratios = [0.5, 1.0, 2.0, 4.0]
dynamic_width_cost_weight = 0.01
dynamic_width_hard_eval = True
dynamic_width_temperature = 1.0
dynamic_width_temperature_final = 1.0
dynamic_width_temperature_anneal_iters = 0
dynamic_width_routing = "soft" # "soft" or "ste"
dynamic_width_hard_loss_weight = 0.0
dynamic_width_entropy_weight = 0.0
dynamic_width_sliced_eval = True
# free per-channel MLP routing
free_channel_mlp = False
free_channel_routing = "soft" # "soft" or "ste"
free_channel_threshold = 0.5
free_channel_target_ratio = 0.4
free_channel_budget_weight = 0.01
free_channel_cost_weight = 0.0
free_channel_temperature = 2.0
free_channel_temperature_final = 0.5
free_channel_temperature_anneal_iters = 1000
free_channel_eval_impl = "dense_mask"
free_channel_prefix_granularity = 64
# block-wise routed MLP with true sliced eval path
block_sparse_mlp = False
block_sparse_block_size = 16
block_sparse_routing = "ste" # "soft" or "ste"
block_sparse_threshold = 0.5
block_sparse_target_ratio = 0.4
block_sparse_budget_weight = 0.01
block_sparse_cost_weight = 0.0
block_sparse_temperature = 2.0
block_sparse_temperature_final = 0.5
block_sparse_temperature_anneal_iters = 1000
block_sparse_sliced_eval = True
block_sparse_eval_impl = "grouped"
# block-wise Width x Bit MLP with fake quantized weight paths
block_precision_mlp = False
block_precision_block_size = 16
block_precision_bit_choices = [2, 4, 8, 16]
block_precision_routing = "ste"
block_precision_cost_weight = 0.01
block_precision_temperature = 1.0
block_precision_temperature_final = 1.0
block_precision_temperature_anneal_iters = 0
block_precision_width_temperature = 1.0
block_precision_sliced_eval = True
block_precision_eval_impl = "grouped"
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  dynamic_exit=dynamic_exit, exit_layers=exit_layers,
                  confidence_method=confidence_method,
                  confidence_threshold=confidence_threshold,
                  entropy_threshold=entropy_threshold,
                  early_exit_loss_weight=early_exit_loss_weight,
                  use_distillation=use_distillation,
                  distillation_temperature=distillation_temperature,
                  distillation_beta=distillation_beta,
                  dynamic_mlp=dynamic_mlp,
                  dynamic_mlp_fast_ratio=dynamic_mlp_fast_ratio,
                  dynamic_mlp_slow_ratio=dynamic_mlp_slow_ratio,
                  dynamic_mlp_cost_weight=dynamic_mlp_cost_weight,
                  dynamic_mlp_threshold=dynamic_mlp_threshold,
                  dynamic_mlp_hard_eval=dynamic_mlp_hard_eval,
                  dynamic_width=dynamic_width,
                  dynamic_width_ratios=dynamic_width_ratios,
                  dynamic_width_cost_weight=dynamic_width_cost_weight,
                  dynamic_width_hard_eval=dynamic_width_hard_eval,
                  dynamic_width_temperature=dynamic_width_temperature,
                  dynamic_width_temperature_final=dynamic_width_temperature_final,
                  dynamic_width_temperature_anneal_iters=dynamic_width_temperature_anneal_iters,
                  dynamic_width_routing=dynamic_width_routing,
                  dynamic_width_hard_loss_weight=dynamic_width_hard_loss_weight,
                  dynamic_width_entropy_weight=dynamic_width_entropy_weight,
                  dynamic_width_sliced_eval=dynamic_width_sliced_eval,
                  free_channel_mlp=free_channel_mlp,
                  free_channel_routing=free_channel_routing,
                  free_channel_threshold=free_channel_threshold,
                  free_channel_target_ratio=free_channel_target_ratio,
                  free_channel_budget_weight=free_channel_budget_weight,
                  free_channel_cost_weight=free_channel_cost_weight,
                  free_channel_temperature=free_channel_temperature,
                  free_channel_temperature_final=free_channel_temperature_final,
                  free_channel_temperature_anneal_iters=free_channel_temperature_anneal_iters,
                  free_channel_eval_impl=free_channel_eval_impl,
                  free_channel_prefix_granularity=free_channel_prefix_granularity,
                  block_sparse_mlp=block_sparse_mlp,
                  block_sparse_block_size=block_sparse_block_size,
                  block_sparse_routing=block_sparse_routing,
                  block_sparse_threshold=block_sparse_threshold,
                  block_sparse_target_ratio=block_sparse_target_ratio,
                  block_sparse_budget_weight=block_sparse_budget_weight,
                  block_sparse_cost_weight=block_sparse_cost_weight,
                  block_sparse_temperature=block_sparse_temperature,
                  block_sparse_temperature_final=block_sparse_temperature_final,
                  block_sparse_temperature_anneal_iters=block_sparse_temperature_anneal_iters,
                  block_sparse_sliced_eval=block_sparse_sliced_eval,
                  block_sparse_eval_impl=block_sparse_eval_impl,
                  block_precision_mlp=block_precision_mlp,
                  block_precision_block_size=block_precision_block_size,
                  block_precision_bit_choices=block_precision_bit_choices,
                  block_precision_routing=block_precision_routing,
                  block_precision_cost_weight=block_precision_cost_weight,
                  block_precision_temperature=block_precision_temperature,
                  block_precision_temperature_final=block_precision_temperature_final,
                  block_precision_temperature_anneal_iters=block_precision_temperature_anneal_iters,
                  block_precision_width_temperature=block_precision_width_temperature,
                  block_precision_sliced_eval=block_precision_sliced_eval,
                  block_precision_eval_impl=block_precision_eval_impl) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Older baseline checkpoints will not have these fields; keep the current
    # command-line/default values so they can still be resumed unchanged.
    for k in ['dynamic_exit', 'exit_layers', 'confidence_method', 'confidence_threshold',
              'entropy_threshold', 'early_exit_loss_weight', 'use_distillation',
              'distillation_temperature', 'distillation_beta', 'dynamic_mlp',
              'dynamic_mlp_fast_ratio', 'dynamic_mlp_slow_ratio',
              'dynamic_mlp_cost_weight', 'dynamic_mlp_threshold',
              'dynamic_mlp_hard_eval', 'dynamic_width', 'dynamic_width_ratios',
              'dynamic_width_cost_weight', 'dynamic_width_hard_eval',
              'dynamic_width_temperature', 'dynamic_width_temperature_final',
              'dynamic_width_temperature_anneal_iters', 'dynamic_width_routing',
              'dynamic_width_hard_loss_weight', 'dynamic_width_entropy_weight',
              'dynamic_width_sliced_eval',
              'free_channel_mlp', 'free_channel_routing',
              'free_channel_threshold', 'free_channel_target_ratio',
              'free_channel_budget_weight', 'free_channel_cost_weight',
              'free_channel_temperature', 'free_channel_temperature_final',
              'free_channel_temperature_anneal_iters', 'free_channel_eval_impl',
              'free_channel_prefix_granularity', 'block_sparse_mlp',
              'block_sparse_block_size', 'block_sparse_routing',
              'block_sparse_threshold', 'block_sparse_target_ratio',
              'block_sparse_budget_weight', 'block_sparse_cost_weight',
              'block_sparse_temperature', 'block_sparse_temperature_final',
              'block_sparse_temperature_anneal_iters', 'block_sparse_sliced_eval',
              'block_sparse_eval_impl', 'block_precision_mlp',
              'block_precision_block_size', 'block_precision_bit_choices',
              'block_precision_routing', 'block_precision_cost_weight', 'block_precision_temperature',
              'block_precision_temperature_final',
              'block_precision_temperature_anneal_iters',
              'block_precision_width_temperature', 'block_precision_sliced_eval',
              'block_precision_eval_impl']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"unexpected keys when loading checkpoint: {unexpected_keys}")
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    if dynamic_exit:
        print("WARNING: dynamic_exit is disabled for init_from='gpt2*' because pretrained GPT-2 checkpoints do not contain early-exit head LayerNorm weights.")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    model_args['dynamic_exit'] = False
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_dynamic_width_temperature(it):
    if not dynamic_width or dynamic_width_temperature_anneal_iters <= 0:
        return dynamic_width_temperature
    ratio = min(max(it / dynamic_width_temperature_anneal_iters, 0.0), 1.0)
    return dynamic_width_temperature + ratio * (dynamic_width_temperature_final - dynamic_width_temperature)

def get_free_channel_temperature(it):
    if not free_channel_mlp or free_channel_temperature_anneal_iters <= 0:
        return free_channel_temperature
    ratio = min(max(it / free_channel_temperature_anneal_iters, 0.0), 1.0)
    return free_channel_temperature + ratio * (free_channel_temperature_final - free_channel_temperature)

def get_block_sparse_temperature(it):
    if not block_sparse_mlp or block_sparse_temperature_anneal_iters <= 0:
        return block_sparse_temperature
    ratio = min(max(it / block_sparse_temperature_anneal_iters, 0.0), 1.0)
    return block_sparse_temperature + ratio * (block_sparse_temperature_final - block_sparse_temperature)

def get_block_precision_temperature(it):
    if not block_precision_mlp or block_precision_temperature_anneal_iters <= 0:
        return block_precision_temperature
    ratio = min(max(it / block_precision_temperature_anneal_iters, 0.0), 1.0)
    return block_precision_temperature + ratio * (block_precision_temperature_final - block_precision_temperature)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if dynamic_width:
        raw_model.set_dynamic_width_temperature(get_dynamic_width_temperature(iter_num))
    if free_channel_mlp:
        raw_model.set_free_channel_temperature(get_free_channel_temperature(iter_num))
    if block_sparse_mlp:
        raw_model.set_block_sparse_temperature(get_block_sparse_temperature(iter_num))
    if block_precision_mlp:
        raw_model.set_block_precision_temperature(get_block_precision_temperature(iter_num))

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        width_stats = raw_model.last_dynamic_width_stats
        if width_stats is not None:
            dist = ", ".join(
                f"w{width}={width_stats['width_fractions'][str(width)]:.3f}"
                for width in width_stats["width_choices"]
            )
            print(
                f"  eval_dynamic_width: mean_width {width_stats['mean_effective_width']:.2f} "
                f"({100 * width_stats['mean_width_ratio']:.2f}%), "
                f"entropy {width_stats['router_entropy']:.4f}, {dist}"
            )
            for layer in width_stats["layers"]:
                layer_dist = ", ".join(
                    f"w{width}={layer['width_fractions'][str(width)]:.3f}"
                    for width in width_stats["width_choices"]
                )
                print(
                    f"    eval layer {layer['layer']}: mean_width {layer['mean_effective_width']:.2f}, "
                    f"entropy {layer['router_entropy']:.4f}, {layer_dist}"
                )
        free_stats = raw_model.last_free_channel_stats
        if free_stats is not None:
            hist = ", ".join(
                f"{bucket}={frac:.3f}"
                for bucket, frac in free_stats["active_width_histogram"].items()
            )
            q = free_stats["active_width_quantiles"]
            print(
                f"  eval_free_channel: mean_active {free_stats['mean_active_channels']:.2f} "
                f"({100 * free_stats['mean_active_ratio']:.2f}%), "
                f"median {free_stats['median_active_channels']:.2f}, "
                f"std {free_stats['std_active_channels']:.2f}, "
                f"entropy {free_stats['gate_entropy']:.2f}"
            )
            print(
                f"    quantiles: p10 {q['p10']:.1f}, p25 {q['p25']:.1f}, "
                f"p50 {q['p50']:.1f}, p75 {q['p75']:.1f}, p90 {q['p90']:.1f}"
            )
            print(f"    active_hist: {hist}")
            print(
                f"    channel_usage: mean {free_stats['mean_channel_usage_rate']:.4f}, "
                f"std {free_stats['std_channel_usage_rate']:.4f}, "
                f"min {free_stats['min_channel_usage_rate']:.4f}, "
                f"max {free_stats['max_channel_usage_rate']:.4f}"
            )
        block_precision_stats = raw_model.last_block_precision_stats
        if block_precision_stats is not None:
            bit_dist = ", ".join(
                f"b{bit}={block_precision_stats['bit_fractions'][str(bit)]:.3f}"
                for bit in block_precision_stats["bit_choices"]
            )
            print(
                f"  eval_block_precision: mean_blocks {block_precision_stats['mean_active_blocks']:.2f}, "
                f"mean_active {block_precision_stats['mean_active_channels']:.2f} "
                f"({100 * block_precision_stats['mean_active_ratio']:.2f}%), "
                f"mean_bit {block_precision_stats['mean_active_bit']:.2f}, "
                f"weight_bits/token {block_precision_stats['mean_weight_bits_per_token']:.1f} "
                f"({100 * block_precision_stats['mean_weight_bit_fraction']:.2f}%), "
                f"entropy {block_precision_stats['bit_entropy']:.4f}, {bit_dist}"
            )
        if wandb_log:
            wandb_metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if width_stats is not None:
                wandb_metrics.update({
                    "dynamic_width/mean_effective_width": width_stats["mean_effective_width"],
                    "dynamic_width/mean_width_ratio": width_stats["mean_width_ratio"],
                    "dynamic_width/router_entropy": width_stats["router_entropy"],
                    "dynamic_width/temperature": raw_model.config.dynamic_width_temperature,
                })
                for width in width_stats["width_choices"]:
                    wandb_metrics[f"dynamic_width/width{width}_fraction"] = width_stats["width_fractions"][str(width)]
            if free_stats is not None:
                q = free_stats["active_width_quantiles"]
                wandb_metrics.update({
                    "free_channel/mean_active_channels": free_stats["mean_active_channels"],
                    "free_channel/mean_active_ratio": free_stats["mean_active_ratio"],
                    "free_channel/median_active_channels": free_stats["median_active_channels"],
                    "free_channel/std_active_channels": free_stats["std_active_channels"],
                    "free_channel/gate_entropy": free_stats["gate_entropy"],
                    "free_channel/fraction_gate_gt_0_5": free_stats["fraction_gate_gt_0_5"],
                    "free_channel/fraction_gate_gt_0_9": free_stats["fraction_gate_gt_0_9"],
                    "free_channel/fraction_gate_lt_0_1": free_stats["fraction_gate_lt_0_1"],
                    "free_channel/channel_usage_std": free_stats["std_channel_usage_rate"],
                    "free_channel/temperature": raw_model.config.free_channel_temperature,
                    "free_channel/p10_width": q["p10"],
                    "free_channel/p50_width": q["p50"],
                    "free_channel/p90_width": q["p90"],
                })
                for bucket, frac in free_stats["active_width_histogram"].items():
                    wandb_metrics[f"free_channel/hist_{bucket}"] = frac
            if block_precision_stats is not None:
                wandb_metrics.update({
                    "block_precision/mean_active_blocks": block_precision_stats["mean_active_blocks"],
                    "block_precision/mean_active_channels": block_precision_stats["mean_active_channels"],
                    "block_precision/mean_active_ratio": block_precision_stats["mean_active_ratio"],
                    "block_precision/mean_active_bit": block_precision_stats["mean_active_bit"],
                    "block_precision/mean_weight_bits_per_token": block_precision_stats["mean_weight_bits_per_token"],
                    "block_precision/mean_weight_bit_fraction": block_precision_stats["mean_weight_bit_fraction"],
                    "block_precision/bit_entropy": block_precision_stats["bit_entropy"],
                    "block_precision/temperature": raw_model.config.block_precision_temperature,
                })
                for bit in block_precision_stats["bit_choices"]:
                    wandb_metrics[f"block_precision/bit{bit}_fraction"] = block_precision_stats["bit_fractions"][str(bit)]
            wandb.log(wandb_metrics)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                if dynamic_width:
                    model_args['dynamic_width_temperature'] = raw_model.config.dynamic_width_temperature
                if free_channel_mlp:
                    model_args['free_channel_temperature'] = raw_model.config.free_channel_temperature
                if block_sparse_mlp:
                    model_args['block_sparse_temperature'] = raw_model.config.block_sparse_temperature
                if block_precision_mlp:
                    model_args['block_precision_temperature'] = raw_model.config.block_precision_temperature
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                if save_checkpoint_every_eval:
                    torch.save(checkpoint, os.path.join(out_dir, f'ckpt_iter_{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        mlp_stats = raw_model.last_dynamic_mlp_stats
        if mlp_stats is not None:
            print(
                f"  dynamic_mlp: mean_gate {mlp_stats['mean_gate']:.4f}, "
                f"slow_soft_fraction {mlp_stats['slow_soft_fraction']:.4f}"
            )
        width_stats = raw_model.last_dynamic_width_stats
        if width_stats is not None:
            loss_stats = raw_model.last_loss_stats or {}
            dist = ", ".join(
                f"w{width}={width_stats['width_fractions'][str(width)]:.3f}"
                for width in width_stats["width_choices"]
            )
            print(
                f"  dynamic_width: mean_width {width_stats['mean_effective_width']:.2f} "
                f"({100 * width_stats['mean_width_ratio']:.2f}%), "
                f"entropy {width_stats['router_entropy']:.4f}, {dist}"
            )
            if loss_stats:
                print(
                    f"  losses: task_ce {loss_stats.get('task_loss', 0.0):.4f}, "
                    f"width_cost {loss_stats.get('dynamic_width_cost', 0.0):.4f}, "
                    f"hard_ce {loss_stats.get('dynamic_width_hard_loss', 0.0):.4f}, "
                    f"total {loss_stats.get('total_loss', lossf):.4f}"
                )
            for layer in width_stats["layers"]:
                layer_dist = ", ".join(
                    f"w{width}={layer['width_fractions'][str(width)]:.3f}"
                    for width in width_stats["width_choices"]
                )
                print(
                    f"    layer {layer['layer']}: mean_width {layer['mean_effective_width']:.2f}, "
                    f"entropy {layer['router_entropy']:.4f}, {layer_dist}"
                )
        free_stats = raw_model.last_free_channel_stats
        if free_stats is not None:
            loss_stats = raw_model.last_loss_stats or {}
            q = free_stats["active_width_quantiles"]
            print(
                f"  free_channel: mean_active {free_stats['mean_active_channels']:.2f} "
                f"({100 * free_stats['mean_active_ratio']:.2f}%), "
                f"median {free_stats['median_active_channels']:.2f}, "
                f"std {free_stats['std_active_channels']:.2f}, "
                f"p10/p50/p90 {q['p10']:.1f}/{q['p50']:.1f}/{q['p90']:.1f}, "
                f"entropy {free_stats['gate_entropy']:.2f}"
            )
            print(
                f"  gate_probs: >0.5 {free_stats['fraction_gate_gt_0_5']:.4f}, "
                f">0.9 {free_stats['fraction_gate_gt_0_9']:.4f}, "
                f"<0.1 {free_stats['fraction_gate_lt_0_1']:.4f}, "
                f"channel_usage_std {free_stats['std_channel_usage_rate']:.4f}"
            )
            if loss_stats:
                print(
                    f"  losses: task_ce {loss_stats.get('task_loss', 0.0):.4f}, "
                    f"budget {loss_stats.get('free_channel_budget_loss', 0.0):.6f}, "
                    f"cost {loss_stats.get('free_channel_cost', 0.0):.4f}, "
                    f"total {loss_stats.get('total_loss', lossf):.4f}"
                )
        block_precision_stats = raw_model.last_block_precision_stats
        if block_precision_stats is not None:
            loss_stats = raw_model.last_loss_stats or {}
            bit_dist = ", ".join(
                f"b{bit}={block_precision_stats['bit_fractions'][str(bit)]:.3f}"
                for bit in block_precision_stats["bit_choices"]
            )
            print(
                f"  block_precision: mean_blocks {block_precision_stats['mean_active_blocks']:.2f}, "
                f"mean_active {block_precision_stats['mean_active_channels']:.2f} "
                f"({100 * block_precision_stats['mean_active_ratio']:.2f}%), "
                f"mean_bit {block_precision_stats['mean_active_bit']:.2f}, "
                f"weight_bits/token {block_precision_stats['mean_weight_bits_per_token']:.1f} "
                f"({100 * block_precision_stats['mean_weight_bit_fraction']:.2f}%), "
                f"entropy {block_precision_stats['bit_entropy']:.4f}, {bit_dist}"
            )
            if loss_stats:
                print(
                    f"  losses: task_ce {loss_stats.get('task_loss', 0.0):.4f}, "
                    f"weight_bit_cost {loss_stats.get('block_precision_cost', 0.0):.4f}, "
                    f"total {loss_stats.get('total_loss', lossf):.4f}"
                )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
