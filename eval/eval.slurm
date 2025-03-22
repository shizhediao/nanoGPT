#!/bin/bash

#SBATCH --job-name=eval_nanogpt
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/sdiao/logs/eval_nanogpt_%j/eval_nanogpt.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/sdiao/logs/eval_nanogpt_%j/eval_nanogpt.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --mem=1500G
#SBATCH --partition=interactive,backfill,batch_block1
#SBATCH --account=nvr_lpr_llm

source /lustre/fsw/portfolios/nvr/users/sdiao/anaconda3/bin/activate nanogpt
export TOKENIZERS_PARALLELISM=false
cd /lustre/fsw/portfolios/nvr/users/sdiao/nanoGPT

MODEL_PATH="/lustre/fsw/portfolios/nvr/users/sdiao/nanoGPT/gpt2-xl-finewebedu"
LOG_PATH=${MODEL_PATH}/eval_logs

mkdir -p ${LOG_PATH}
# Add timestamp to the output file name
TIMESTAMP=$(date +%Y%m%d%H%M%S)

CUDA_VISIBLE_DEVICES=0 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks arc_easy arc_challenge --num_fewshot 0 > ${LOG_PATH}/arc_${TIMESTAMP}.log 2> ${LOG_PATH}/arc_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=1 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks mmlu_continuation --num_fewshot 5 > ${LOG_PATH}/mmlu_continuation_${TIMESTAMP}.log 2> ${LOG_PATH}/mmlu_continuation_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=2 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks truthfulqa --num_fewshot 0 > ${LOG_PATH}/truthfulqa_${TIMESTAMP}.log 2> ${LOG_PATH}/truthfulqa_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=3 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks gpqa_main_zeroshot winogrande --num_fewshot 0 > ${LOG_PATH}/gpqa_winogrande_${TIMESTAMP}.log 2> ${LOG_PATH}/gpqa_winogrande_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=4 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks wikitext lambada_openai --num_fewshot 0 > ${LOG_PATH}/wikitext_lambda_${TIMESTAMP}.log 2> ${LOG_PATH}/wikitext_lambda_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=5 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks hellaswag --num_fewshot 0 > ${LOG_PATH}/hellaswag_${TIMESTAMP}.log 2> ${LOG_PATH}/hellaswag_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=6 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks piqa --num_fewshot 0 > ${LOG_PATH}/piqa_${TIMESTAMP}.log 2> ${LOG_PATH}/piqa_${TIMESTAMP}.err &

CUDA_VISIBLE_DEVICES=7 python eval/lm_eval_harness.py --out_dir ${MODEL_PATH} --tasks social_iqa --num_fewshot 0 > ${LOG_PATH}/siqa_${TIMESTAMP}.log 2> ${LOG_PATH}/siqa_${TIMESTAMP}.err &

wait

echo "Finished evaluating model: ${MODEL_PATH}"