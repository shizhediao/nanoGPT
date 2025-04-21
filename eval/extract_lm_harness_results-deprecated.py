import os
import json
import csv
import pandas as pd
import numpy as np
import yaml

def extract_results(folder_path, specific_model_name=None):
    all_results = {}
    for root, dirs, files in os.walk(folder_path):
        if "mamba" in root or 'v2_' in root or 'archive' in root: #  or 'archive' in root
            continue
        
        # if specific_model_name and specific_model_name not in root:
        #     continue
        
        one_results = {}
        model_name = root.split("__")[-1]
        print(f"Processing {model_name}")
        for file in files:
            if file.startswith('metrics'):
                with open(os.path.join(root, file), 'r') as f:
                    file_content = json.load(f)
                    # print(f"File content: {file_content}")

                    for result in file_content:
                        alias = result.get('alias')
                        if alias == 'wikitext':
                            one_results['wiki_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'lambada_openai':
                            one_results['lambda_ppl'] = result.get('perplexity,none')
                            one_results['lambda_acc'] = result.get('acc,none') * 100
                        elif alias == 'piqa':
                            one_results['piqa_acc_norm'] = result.get('acc_norm,none') * 100
                        if alias == 'arc_challenge':
                            one_results['arc_challenge_acc_norm'] = result.get('acc_norm,none') * 100
                        elif alias == 'arc_easy':
                            one_results['arc_easy_acc'] = result.get('acc,none') * 100
                        elif alias == 'hellaswag':
                            one_results['hellaswag_acc_norm'] = result.get('acc_norm,none') * 100
                        elif alias == 'winogrande':
                            one_results['winogrande_acc'] = result.get('acc,none') * 100
                        elif alias == 'truthfulqa_mc2':
                            one_results['truthfulqa_mc2_acc'] = result.get('acc,none') * 100
                        elif alias == 'social_iqa':
                            one_results['siqa'] = result.get('acc,none') * 100
                        elif alias == 'C4 100 Domains':
                            one_results['paloma_c4_100_domains_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'C4':
                            one_results['paloma_c4_ppl'] = result.get('word_perplexity,none')
                        elif alias == '100 PLs':
                            one_results['paloma_dolma_100_PLs_ppl'] = result.get('word_perplexity,none')
                        elif alias == '100 Subreddits':
                            one_results['paloma_dolma_100_subreddits_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'Dolma V1.5':
                            one_results['paloma_dolma_v1_5_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'Falcon':
                            one_results['paloma_falcon_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'M2D2 S2ORC':
                            one_results['paloma_m2d2_s2orc_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'M2D2 Wikipedia':
                            one_results['paloma_m2d2_wikipedia_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'PTB':
                            one_results['paloma_ptb_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'mC4':
                            one_results['paloma_mc4_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'RedPajama':
                            one_results['paloma_redpajama_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'Wikitext-103':
                            one_results['paloma_wikitext_103_ppl'] = result.get('word_perplexity,none')
                        elif alias == 'mmlu':
                            one_results['mmlu_acc'] = result.get('acc,none') * 100
                        elif alias == 'mmlu (continuation)':
                            one_results['mmlu_cloze_avg'] = result.get('acc,none') * 100
                        elif 'mmlu_continuation' in file and alias == ' - stem':
                            one_results['mmlu_cloze_stem'] = result.get('acc,none') * 100
                        elif 'mmlu_continuation' in file and alias == ' - humanities':
                            one_results['mmlu_cloze_humanities'] = result.get('acc,none') * 100
                        elif 'mmlu_continuation' in file and alias == ' - social sciences':
                            one_results['mmlu_cloze_social_sciences'] = result.get('acc,none') * 100
                        elif 'mmlu_continuation' in file and alias == ' - other':
                            one_results['mmlu_cloze_other'] = result.get('acc,none') * 100
                    # wiki_ppl = results.get('wikitext', {}).get('word_perplexity,none')
                    # lambda_ppl = results.get('lambada_openai', {}).get('perplexity,none')
                    # lambda_acc = results.get('lambada_openai', {}).get('acc,none')
                    # piqa_acc_norm = results.get('piqa', {}).get('acc_norm,none')
                    # arc_challenge_acc_norm = results.get('arc_challenge', {}).get('acc_norm,none')
                    # arc_easy_acc = results.get('arc_easy', {}).get('acc,none')
                    # # gsm8k_value = results.get('gsm8k', {}).get('exact_match,flexible-extract')
                    # # mmlu_value = results.get('mmlu', {}).get('acc,none')
                    # # ifeval_value_prompt_level_loose_acc = results.get('ifeval', {}).get('prompt_level_loose_acc,none')
                    # # ifeval_value_inst_level_loose_acc = results.get('ifeval', {}).get('inst_level_loose_acc,none')
                    # hellaswag_value = results.get('hellaswag', {}).get('acc_norm,none')
                    # winogrande_value = results.get('winogrande', {}).get('acc,none')
                    # truthfulqa_mc2_value = results.get('truthfulqa_mc2', {}).get('acc,none')
                    # siqa_value = results.get('social_iqa', {}).get('acc,none')

        if one_results:
            all_results[model_name] = one_results
    
    return all_results

if __name__ == "__main__":
    folder_path = "/lustre/fsw/portfolios/nvr/users/sdiao/nanoGPT/gpt2-xl-finewebedu"
    # specific_model_name = "gpt2-xl-smollm-8k"
    specific_model_name = folder_path.split("/")[-1]

    results = extract_results(folder_path, specific_model_name)

    # csv_columns = ['model_name', 'wiki_ppl', 'lambda_ppl', 'lambda_acc', 'piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa', 'avg', 'valid_avg', 'mmlu_acc', 'mmlu_cloze_avg', 'mmlu_cloze_stem', 'mmlu_cloze_humanities', 'mmlu_cloze_social_sciences', 'mmlu_cloze_other', 'condensed_topics', 'paloma_c4_100_domains_ppl', 'paloma_c4_ppl', 'paloma_dolma_100_PLs_ppl', 'paloma_dolma_100_subreddits_ppl', 'paloma_dolma_v1_5_ppl', 'paloma_falcon_ppl', 'paloma_m2d2_s2orc_ppl', 'paloma_m2d2_wikipedia_ppl', 'paloma_ptb_ppl', 'paloma_mc4_ppl', 'paloma_redpajama_ppl', 'paloma_wikitext_103_ppl', 'config_name', 'super_cluster_1', 'super_cluster_2', 'super_cluster_3', 'super_cluster_4', 'super_cluster_5', 'super_cluster_6', 'super_cluster_7', 'super_cluster_8', 'super_cluster_9', 'super_cluster_10', 'super_cluster_11', 'super_cluster_12', 'super_cluster_13', 'super_cluster_14', 'super_cluster_15', 'super_cluster_16', 'sft_cluster', 'smollm-cosmopedia', 'smollm-finewebedu', 'smollm-pythonedu', 'fake_acad']
    csv_columns = ['model_name', 'wiki_ppl', 'lambda_ppl', 'lambda_acc', 'piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa', 'avg', 'valid_avg', 'mmlu_acc', 'mmlu_cloze_avg', 'mmlu_cloze_stem', 'mmlu_cloze_humanities', 'mmlu_cloze_social_sciences', 'mmlu_cloze_other', 'avg_w_mmlu', 'condensed_topics', 'paloma_c4_100_domains_ppl', 'paloma_c4_ppl', 'paloma_dolma_100_PLs_ppl', 'paloma_dolma_100_subreddits_ppl', 'paloma_dolma_v1_5_ppl', 'paloma_falcon_ppl', 'paloma_m2d2_s2orc_ppl', 'paloma_m2d2_wikipedia_ppl', 'paloma_ptb_ppl', 'paloma_mc4_ppl', 'paloma_redpajama_ppl', 'paloma_wikitext_103_ppl', 'config_name', 'low', 'synthetic-high-knowledge_list', 'synthetic-high-diverse_qa_pairs', 'synthetic-high-wrap_medium', 'synthetic-high-distill', 'synthetic-low-wrap_medium', 'synthetic-high-extract_knowledge', 'high', 'medium-low', 'medium', 'medium-high']

    csv_file = f"lm_harness_results_{specific_model_name}.csv"
    error_cluster = []
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for model_name, model_results in results.items():
                print(f"model_name: {model_name}")
                print(f"model_results: {model_results}")
                model_name = model_name.split("/")[-1]
                row = {'model_name': model_name}
                row.update(model_results)

                # 根据这几列'lambda_acc', 'piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa'，计算avg
                try:
                    row['avg'] = sum([row[key] for key in ['piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa']]) / len(['piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa'])
                    row['valid_avg'] = sum([row[key] for key in ['piqa_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm']]) / len(['piqa_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm'])
                    row['avg_w_mmlu'] = sum([row[key] for key in ['piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa', 'mmlu_cloze_stem', 'mmlu_cloze_humanities', 'mmlu_cloze_social_sciences', 'mmlu_cloze_other']]) / len(['piqa_acc_norm', 'arc_challenge_acc_norm', 'arc_easy_acc', 'hellaswag_acc_norm', 'winogrande_acc', 'truthfulqa_mc2_acc', 'siqa', 'mmlu_cloze_stem', 'mmlu_cloze_humanities', 'mmlu_cloze_social_sciences', 'mmlu_cloze_other'])
                except:
                    # error_cluster.append(int(model_name.split("_cluster_")[-1]))
                    error_cluster.append(model_name)
                    print(f"SHIZHE DEBUG:model_name: {model_name} has no avg")
                    # continue

                writer.writerow(row)

        # Read the CSV file we just wrote
        df = pd.read_csv(csv_file)
        
        # Calculate correlations with 'avg'
        correlations = {}
        for column in csv_columns:
            if column not in ['model_name', 'avg', 'condensed_topics']:
                try:
                    correlation = df[column].corr(df['avg'])
                    if not np.isnan(correlation):  # Only store non-NaN correlations
                        correlations[column] = correlation
                except:
                    continue
        
        # Sort correlations by absolute value
        sorted_correlations = dict(sorted(correlations.items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True))
        
        # Print correlations
        print("\nCorrelations with average score:")
        for metric, corr in sorted_correlations.items():
            print(f"{metric}: {corr:.3f}")      

        print(f"SHIZHE DEBUG:error_cluster: {error_cluster}")
    except IOError:
        print("I/O error")
