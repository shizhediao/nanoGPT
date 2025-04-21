import os
import numpy as np
from tqdm import tqdm

def merge_bin_files(input_files, output_file):
    """
    merge multiple binary token files into one file
    
    Args:
        input_files: list of input bin file paths
        output_file: output bin file path
    """
    # calculate the total length
    total_length = 0
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file not found: {file_path}")
        
        arr = np.memmap(file_path, dtype=np.uint16, mode='r')
        total_length += len(arr)
        print(f"file {os.path.basename(file_path)} contains {len(arr)} tokens")
    
    print(f"total tokens: {total_length}")
    
    # create the output file
    merged_arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_length,))
    
    # merge files
    current_idx = 0
    for file_path in tqdm(input_files, desc="merge files"):
        arr = np.memmap(file_path, dtype=np.uint16, mode='r')
        file_length = len(arr)
        
        # use batch processing to avoid memory issues
        batch_size = 1024 * 1024 * 10  # about 20MB per batch
        for i in tqdm(range(0, file_length, batch_size), desc=f"processing {os.path.basename(file_path)}"):
            end_idx = min(i + batch_size, file_length)
            batch = arr[i:end_idx]
            merged_arr[current_idx:current_idx + len(batch)] = batch
            current_idx += len(batch)
    
    # ensure data is written to disk
    merged_arr.flush()
    print(f"merge completed! output file: {output_file}")

if __name__ == "__main__":
    # set input files and output file
    data_dir = os.path.dirname(__file__)
    
    # list the files to merge
    input_files = [
        os.path.join(data_dir, "part_0_train.bin"),
        os.path.join(data_dir, "part_1_train.bin"),
        os.path.join(data_dir, "part_2_train.bin"),
        os.path.join(data_dir, "part_3_train.bin"),
        os.path.join(data_dir, "part_4_train.bin"),
        os.path.join(data_dir, "part_5_train.bin"),
        os.path.join(data_dir, "part_6_train.bin"),
        os.path.join(data_dir, "part_7_train.bin"),
        os.path.join(data_dir, "part_8_train.bin"),
        os.path.join(data_dir, "part_9_train.bin")
    ]
    
    # output file path
    output_file = os.path.join(data_dir, "train.bin")
    
    # execute the merge
    merge_bin_files(input_files, output_file)
    
    # optional: merge the validation set
    val_input_files = [
        os.path.join(data_dir, "part_0_val.bin"),
        os.path.join(data_dir, "part_1_val.bin"),
        os.path.join(data_dir, "part_2_val.bin"),
        os.path.join(data_dir, "part_3_val.bin"),
        os.path.join(data_dir, "part_4_val.bin"),
        os.path.join(data_dir, "part_5_val.bin"),
        os.path.join(data_dir, "part_6_val.bin"),
        os.path.join(data_dir, "part_7_val.bin"),
        os.path.join(data_dir, "part_8_val.bin"),
        os.path.join(data_dir, "part_9_val.bin")
    ]
    val_output_file = os.path.join(data_dir, "val.bin")
    
    # execute the validation set merge
    merge_bin_files(val_input_files, val_output_file)