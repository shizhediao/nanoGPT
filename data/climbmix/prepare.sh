#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1

for i in {0..9}
do
    cmd='
    source /lustre/fsw/portfolios/nvr/users/sdiao/anaconda3/bin/activate nanogpt;
    which pip;
    which python;
    pip list;
    cd /lustre/fsw/portfolios/nvr/users/sdiao/nanoGPT;
    python data/climbmix/prepare.py --file_name part_'"$i"'.jsonl
    '

    submit_job --gpu 8 --nodes 1 --account nvr_lpr_agentic --partition interactive,backfill,batch --notify_on_start --duration 4 -n tokenize_climbmix_part_$i --autoresume_before_timelimit 5 --image /lustre/fsw/portfolios/nvr/users/sdiao/docker/megatron_py25.sqsh --command ''"${cmd}"'' 
done
# batch_singlenode,batch_short,backfill,batch_block1
