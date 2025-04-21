# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'nanogpt'
wandb_run_name='gpt2-xl-climbmix-8k-lr1e-4'
dataset = 'climbmix'
out_dir = 'gpt2-xl-climbmix-8k-lr1e-4'
init_from = 'resume'

# 1 batch size * 1024 block size * 16 gradaccum * 256 GPUs = 32M
batch_size = 1
block_size = 8192
gradient_accumulation_steps = 16 * 256

# this makes total number of tokens be 100B
learning_rate = 1e-4
min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 500

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# model
n_layer = 48
n_head = 25
n_embd = 1600

