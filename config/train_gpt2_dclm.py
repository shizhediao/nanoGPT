# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name='gpt2-xl-dclm'
dataset = 'dclm_baseline'
out_dir = 'gpt2-xl-dclm'
init_from = 'resume'

# 12 batch size * 1024 block size * 1 gradaccum * 256 GPUs = 3.1M
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 1 * 256

# this makes total number of tokens be 100B
max_iters = 35000
lr_decay_iters = 35000
warmup_iters = 1000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# model
n_layer = 48
n_head = 25
n_embd = 1600

