import os
import numpy as np
from tqdm import tqdm

def merge_bin_files(input_files, output_file):
    """
    合并多个二进制token文件到一个文件
    
    Args:
        input_files: 输入bin文件路径列表
        output_file: 输出bin文件路径
    """
    # 计算总长度
    total_length = 0
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        arr = np.memmap(file_path, dtype=np.uint16, mode='r')
        total_length += len(arr)
        print(f"文件 {os.path.basename(file_path)} 包含 {len(arr)} 个token")
    
    print(f"合并后总token数: {total_length}")
    
    # 创建输出文件
    merged_arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_length,))
    
    # 合并文件
    current_idx = 0
    for file_path in tqdm(input_files, desc="合并文件"):
        arr = np.memmap(file_path, dtype=np.uint16, mode='r')
        file_length = len(arr)
        
        # 使用分批处理来避免内存问题
        batch_size = 1024 * 1024 * 10  # 每批约20MB
        for i in tqdm(range(0, file_length, batch_size), desc=f"处理 {os.path.basename(file_path)}"):
            end_idx = min(i + batch_size, file_length)
            batch = arr[i:end_idx]
            merged_arr[current_idx:current_idx + len(batch)] = batch
            current_idx += len(batch)
    
    # 确保数据写入磁盘
    merged_arr.flush()
    print(f"合并完成! 输出文件: {output_file}")

if __name__ == "__main__":
    # 设置输入文件和输出文件
    data_dir = os.path.dirname(__file__)
    
    # 这里列出你要合并的三个文件
    input_files = [
        os.path.join(data_dir, "train_seed1.bin"),
        os.path.join(data_dir, "train_seed3.bin"),
        os.path.join(data_dir, "train_seed4.bin"),
        os.path.join(data_dir, "train_seed5.bin"),
        os.path.join(data_dir, "train_seed6.bin")
    ]
    
    # 输出文件路径
    output_file = os.path.join(data_dir, "train.bin")
    
    # 执行合并
    merge_bin_files(input_files, output_file)
    
    # 可选: 同样合并验证集
    val_input_files = [
        os.path.join(data_dir, "val_seed1.bin"),
        os.path.join(data_dir, "val_seed3.bin"),
        os.path.join(data_dir, "val_seed4.bin"),
        os.path.join(data_dir, "val_seed5.bin"),
        os.path.join(data_dir, "val_seed6.bin")
    ]
    val_output_file = os.path.join(data_dir, "val.bin")
    
    # 执行验证集合并
    merge_bin_files(val_input_files, val_output_file)