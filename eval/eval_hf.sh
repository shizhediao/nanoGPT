#!/bin/bash

# Base script that will be used to generate individual job scripts
cat > eval_template.sh << 'EOL'
#!/bin/bash

#SBATCH --job-name=eval_nanogpt_ITER_NUM
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/sdiao/logs/eval_nanogpt_%j/eval_nanogpt.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/sdiao/logs/eval_nanogpt_%j/eval_nanogpt.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --mem=1500G
#SBATCH --partition=interactive,batch_short,batch_singlenode,backfill,batch_block1
#SBATCH --account=nvr_lpr_llm

source /lustre/fsw/portfolios/nvr/users/sdiao/anaconda3/bin/activate nanogpt
export TOKENIZERS_PARALLELISM=false
cd /lustre/fsw/portfolios/nvr/users/sdiao/nanoGPT

iter_num=ITER_NUM

model_name="gpt2-xl-smollm-8k-lr1e-4"
original_checkpoint_path="${model_name}/ckpt_${iter_num}.pt"
hf_model_path="${model_name}/hf_model_${iter_num}"
result_path="${model_name}/eval_results_${iter_num}"
LOG_PATH=${model_name}/eval_logs
mkdir -p ${LOG_PATH}
TIMESTAMP=$(date +%Y%m%d%H%M%S)

python convert_to_hf.py --checkpoint ${original_checkpoint_path} --output_dir ${hf_model_path}

# 为每个评估任务指定不同的端口
MASTER_PORT=29500 lm_eval --model hf \
    --model_args pretrained=${hf_model_path},dtype="bfloat16" \
    --tasks arc_easy,arc_challenge,race \
    --device cuda:0 \
    --batch_size auto:4 \
    --seed 1337 \
    --num_fewshot 0 \
    --output_path ${result_path} > ${LOG_PATH}/arc_${TIMESTAMP}.log 2> ${LOG_PATH}/arc_${TIMESTAMP}.err &

MASTER_PORT=29501 lm_eval --model hf \
    --model_args pretrained=${hf_model_path},dtype="bfloat16" \
    --tasks truthfulqa,openbookqa \
    --device cuda:1 \
    --batch_size auto:4 \
    --seed 1337 \
    --num_fewshot 0 \
    --output_path ${result_path} > ${LOG_PATH}/truthfulqa_${TIMESTAMP}.log 2> ${LOG_PATH}/truthfulqa_${TIMESTAMP}.err &

MASTER_PORT=29502 lm_eval --model hf \
    --model_args pretrained=${hf_model_path},dtype="bfloat16" \
    --tasks gpqa_main_zeroshot,winogrande \
    --device cuda:2 \
    --batch_size auto:4 \
    --seed 1337 \
    --num_fewshot 0 \
    --output_path ${result_path} > ${LOG_PATH}/gpqa_winogrande_${TIMESTAMP}.log 2> ${LOG_PATH}/gpqa_winogrande_${TIMESTAMP}.err &

MASTER_PORT=29503 lm_eval --model hf \
    --model_args pretrained=${hf_model_path},dtype="bfloat16" \
    --tasks wikitext,lambada_openai,boolq \
    --device cuda:3 \
    --batch_size auto:4 \
    --seed 1337 \
    --num_fewshot 0 \
    --output_path ${result_path} > ${LOG_PATH}/wikitext_${TIMESTAMP}.log 2> ${LOG_PATH}/wikitext_${TIMESTAMP}.err &

MASTER_PORT=29504 lm_eval --model hf \
    --model_args pretrained=${hf_model_path},dtype="bfloat16" \
    --tasks hellaswag,piqa,social_iqa \
    --device cuda:4 \
    --batch_size auto:4 \
    --seed 1337 \
    --num_fewshot 0 \
    --output_path ${result_path} > ${LOG_PATH}/hellaswag_${TIMESTAMP}.log 2> ${LOG_PATH}/hellaswag_${TIMESTAMP}.err &

MASTER_PORT=29505 lm_eval --model hf \
    --model_args pretrained=${hf_model_path},dtype="bfloat16" \
    --tasks mmlu_continuation \
    --device cuda:5 \
    --batch_size auto:4 \
    --seed 1337 \
    --num_fewshot 5 \
    --output_path ${result_path} > ${LOG_PATH}/mmlu_continuation_${TIMESTAMP}.log 2> ${LOG_PATH}/mmlu_continuation_${TIMESTAMP}.err &

wait

echo "Evaluation completed. Results saved to ${result_path}"
EOL

# Loop through iterations from 1000 to 30000 with step size 1000
for iter in $(seq 0 2000 11000); do
    # Create a job script for this iteration
    job_script="eval_iter_${iter}.sh"
    cp eval_template.sh ${job_script}
    
    # Replace the placeholder with the current iteration number
    sed -i "s/ITER_NUM/${iter}/g" ${job_script}
    
    # Submit the job
    echo "Submitting job for iteration ${iter}..."
    sbatch ${job_script}
    rm ${job_script}
    
    # Optional: add a small delay between submissions to avoid overwhelming the scheduler
    sleep 2
done

# Clean up the template file
rm eval_template.sh

echo "All evaluation jobs have been submitted."