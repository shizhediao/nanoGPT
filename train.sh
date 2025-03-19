#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1

cmd='
source /lustre/fsw/portfolios/nvr/users/sdiao/anaconda3/bin/activate nanogpt;
which pip;
which python;
pip list;
torchrun --nproc_per_node=$SUBMIT_GPUS --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/train_gpt2.py
'

submit_job --gpu 8 --nodes 4 --account llmservice_fm_vision --partition batch_short,backfill,batch --notify_on_start --duration 2 -n nanogpt --autoresume_before_timelimit 10 --image /lustre/fsw/portfolios/nvr/users/sdiao/docker/megatron_py25.sqsh --command ''"${cmd}"'' 
# batch_singlenode,batch_short,backfill,batch_block1
