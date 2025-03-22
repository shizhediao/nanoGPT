# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 96  # Reduced from 128 to avoid potential memory issues

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = 96  # Reduced to troubleshoot the subprocess error

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load local jsonl files instead of downloading from HuggingFace

    data_dir = "/lustre/fsw/portfolios/nvr/users/sdiao/data/smollm_raw_jsonl"
    file_name = "cosmopedia-v2.jsonl" #python-edu.jsonl   fineweb-edu-dedup.jsonl
    jsonl_files = [os.path.join(data_dir, file_name)]
    
    print(f"Found {len(jsonl_files)} jsonl files: {jsonl_files}")
    
    # Load all jsonl files at once
    try:
        dataset = load_dataset("json", data_files=jsonl_files, num_proc=num_proc_load_dataset)
        print(f"Dataset loaded: {dataset}")
        
        # # Print the structure of the first example to understand the data format
        # print("Dataset structure example:")
        # if len(dataset["train"]) > 0:
        #     print(dataset["train"][0])
        #     print(f"Available fields: {dataset['train'].features}")
        
        # Extract only the 'text' field which is common across all files
        # def extract_text(example):
        #     return {"text": example['text']}
        
        # print("Extracting text field from all samples", flush=True)
        # text_dataset = dataset["train"].map(extract_text, num_proc=num_proc, remove_columns=[col for col in dataset["train"].column_names if col != 'text'])
        
    except Exception as e:
        print(f"Error processing files: {e}")
        raise
    
    # Create train/val split from the loaded data
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # Print dataset info
    print(f"Dataset split: {split_dataset}", flush=True)

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{file_name.split(".")[0]}_{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            print(f'batch_idx: {batch_idx}')
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
