#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_RUN_ID="nanogpt-dclm"

cmd='
source /lustre/fsw/portfolios/nvr/users/sdiao/anaconda3/bin/activate nanogpt;
which pip;
which python;
pip list;
torchrun --nproc_per_node=$SUBMIT_GPUS --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/train_gpt2_dclm.py
'

submit_job --gpu 8 --nodes 32 --account nvr_lpr_agentic --partition backfill,batch --notify_on_start --duration 4 -n nanogpt-dclm --autoresume_before_timelimit 10 --image /lustre/fsw/portfolios/nvr/users/sdiao/docker/megatron_py25.sqsh --command ''"${cmd}"'' 
# batch_singlenode,batch_short,backfill,batch_block1
