#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1

cmd='
source /lustre/fsw/portfolios/nvr/users/sdiao/anaconda3/bin/activate nanogpt;
which pip;
which python;
pip list;
cd /lustre/fsw/portfolios/nvr/users/sdiao/nanoGPT;
python data/dclm_baseline/prepare.py --seed 6
'

submit_job --gpu 8 --nodes 1 --account llmservice_fm_vision --partition interactive,backfill,batch --notify_on_start --duration 4 -n tokenize_dclm_baseline --autoresume_before_timelimit 5 --image /lustre/fsw/portfolios/nvr/users/sdiao/docker/megatron_py25.sqsh --command ''"${cmd}"'' 
# batch_singlenode,batch_short,backfill,batch_block1
