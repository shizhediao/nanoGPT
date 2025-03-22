import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer, GPT2TokenizerFast


def convert_to_huggingface(checkpoint_path, output_dir, device=None, validate=True, debug=True):
    """
    Convert a trained nanoGPT model to Hugging Face GPT2LMHeadModel format.
    
    Args:
        checkpoint_path: Path to the nanoGPT checkpoint file
        output_dir: Directory to save the Hugging Face model
        device: Device to load the model on
        validate: Whether to validate the conversion by comparing outputs
        debug: Whether to perform additional debugging checks
    
    Returns:
        The converted Hugging Face model
    """
    # Load the nanoGPT checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    model_args = checkpoint.get('model_args', {})
    if not model_args and 'config' in checkpoint:
        # Try to get config from older checkpoint format
        model_args = checkpoint['config'].get('model', {})
    
    # Initialize nanoGPT model
    gptconf = GPTConfig(**model_args)
    nano_model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    # Handle potential '_orig_mod.' prefix from torch.compile
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    nano_model.load_state_dict(state_dict)
    nano_model.eval()
    
    # Create equivalent Hugging Face config
    hf_config = GPT2Config(
        vocab_size=gptconf.vocab_size,
        n_positions=gptconf.block_size,
        n_embd=gptconf.n_embd,
        n_layer=gptconf.n_layer,
        n_head=gptconf.n_head,
        activation_function="gelu",
        resid_pdrop=gptconf.dropout,
        embd_pdrop=gptconf.dropout,
        attn_pdrop=gptconf.dropout,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None
    )
    
    # Create a new Hugging Face model with random weights
    hf_model = GPT2LMHeadModel(hf_config)
    
    # Get state dicts
    nano_sd = nano_model.state_dict()
    hf_sd = hf_model.state_dict()
    
    # Create mapping between nanoGPT and Hugging Face model keys
    mapping = {}
    
    # Embedding layers
    mapping['transformer.wte.weight'] = 'transformer.wte.weight'
    mapping['transformer.wpe.weight'] = 'transformer.wpe.weight'
    
    # Final layer norm
    mapping['transformer.ln_f.weight'] = 'transformer.ln_f.weight'
    if gptconf.bias:
        mapping['transformer.ln_f.bias'] = 'transformer.ln_f.bias'
    
    # LM head
    mapping['lm_head.weight'] = 'lm_head.weight'
    
    # Transformer blocks
    for i in range(gptconf.n_layer):
        # Layer norms
        mapping[f'transformer.h.{i}.ln_1.weight'] = f'transformer.h.{i}.ln_1.weight'
        mapping[f'transformer.h.{i}.ln_2.weight'] = f'transformer.h.{i}.ln_2.weight'
        if gptconf.bias:
            mapping[f'transformer.h.{i}.ln_1.bias'] = f'transformer.h.{i}.ln_1.bias'
            mapping[f'transformer.h.{i}.ln_2.bias'] = f'transformer.h.{i}.ln_2.bias'
        
        # Attention weights - need to be transposed
        mapping[f'transformer.h.{i}.attn.c_attn.weight'] = f'transformer.h.{i}.attn.c_attn.weight'
        mapping[f'transformer.h.{i}.attn.c_proj.weight'] = f'transformer.h.{i}.attn.c_proj.weight'
        
        # MLP weights - need to be transposed
        mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'transformer.h.{i}.mlp.c_fc.weight'
        mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'transformer.h.{i}.mlp.c_proj.weight'
        
        # Biases
        if gptconf.bias:
            mapping[f'transformer.h.{i}.attn.c_attn.bias'] = f'transformer.h.{i}.attn.c_attn.bias'
            mapping[f'transformer.h.{i}.attn.c_proj.bias'] = f'transformer.h.{i}.attn.c_proj.bias'
            mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'transformer.h.{i}.mlp.c_fc.bias'
            mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'transformer.h.{i}.mlp.c_proj.bias'
    
    # Copy weights from nanoGPT to Hugging Face model
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    
    for nano_key, hf_key in mapping.items():
        if any(nano_key.endswith(w) for w in transposed):
            # 特殊处理需要转置的权重
            # 注意：这里是将 nano 模型的权重转置后复制到 hf 模型
            hf_sd[hf_key].copy_(nano_sd[nano_key].t())
        else:
            # 直接复制其他参数
            hf_sd[hf_key].copy_(nano_sd[nano_key])
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
        
    print(f"Model successfully converted and saved to {output_dir}")

    # 加载 GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # 设置 padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"GPT-2 tokenizer saved to {output_dir}")

    
    # 调试模式
    if debug:
        debug_architecture_differences(nano_model, hf_model, gptconf, mapping)
        fix_weight_mapping(nano_model, hf_model, gptconf)
    
    # 验证转换
    if validate:
        print("Validating conversion...")
        nano_model.eval()
        hf_model.eval()
        
        # 创建随机输入
        input_ids = torch.randint(0, gptconf.vocab_size, (1, 10), device=device)
        
        # 比较详细的层输出
        if debug:
            compare_attention_outputs(nano_model, hf_model, input_ids, device)
        
        # 比较最终输出
        with torch.no_grad():
            # 对于 nanoGPT，我们需要明确指定不需要计算损失
            nano_logits, _ = nano_model(input_ids, targets=None)
            
            # 对于 Hugging Face 模型，确保我们获取完整的 logits
            hf_outputs = hf_model(input_ids, return_dict=True)
            hf_logits = hf_outputs.logits
            
            print(f"nano_logits shape: {nano_logits.shape}")
            print(f"hf_logits shape: {hf_logits.shape}")
            
            # 确保我们比较相同形状的输出
            if nano_logits.shape[1] == 1 and hf_logits.shape[1] > 1:
                # nanoGPT 只返回最后一个位置，取 HF 的最后一个位置
                hf_logits = hf_logits[:, -1:, :]
                print("Using only the last position from HF model")
            elif hf_logits.shape[1] == 1 and nano_logits.shape[1] > 1:
                # HF 只返回最后一个位置，取 nanoGPT 的最后一个位置
                nano_logits = nano_logits[:, -1:, :]
                print("Using only the last position from nanoGPT model")
        
        # 打印每个位置的差异 (根据可用的位置数量)
        seq_len = min(nano_logits.size(1), hf_logits.size(1))
        for pos in range(seq_len):
            pos_diff = (nano_logits[0, pos] - hf_logits[0, pos]).abs().max().item()
            print(f"Position {pos} max diff: {pos_diff:.6f}")
        
        # 总体差异
        max_diff = (nano_logits[:, :seq_len] - hf_logits[:, :seq_len]).abs().max().item()
        print(f"Maximum difference in logits: {max_diff:.6f}")
        
        if max_diff > 1e-3:
            print("WARNING: Large difference detected between original and converted model outputs!")
        else:
            print("Validation successful! Models produce similar outputs.")
    
    return hf_model

# 修改权重映射部分
def fix_weight_mapping(nano_model, hf_model, gptconf):
    nano_sd = nano_model.state_dict()
    hf_sd = hf_model.state_dict()
    
    # 重新检查权重映射
    for i in range(gptconf.n_layer):
        # 注意力层权重
        attn_c_attn_w = f'transformer.h.{i}.attn.c_attn.weight'
        attn_c_proj_w = f'transformer.h.{i}.attn.c_proj.weight'
        
        # 检查形状并打印
        print(f"Layer {i} attention weights:")
        print(f"  nano {attn_c_attn_w}: {nano_sd[attn_c_attn_w].shape}")
        print(f"  hf {attn_c_attn_w}: {hf_sd[attn_c_attn_w].shape}")
        
        # MLP层权重
        mlp_c_fc_w = f'transformer.h.{i}.mlp.c_fc.weight'
        mlp_c_proj_w = f'transformer.h.{i}.mlp.c_proj.weight'
        
        print(f"Layer {i} MLP weights:")
        print(f"  nano {mlp_c_fc_w}: {nano_sd[mlp_c_fc_w].shape}")
        print(f"  hf {mlp_c_fc_w}: {hf_sd[mlp_c_fc_w].shape}")
        
        # 检查是否需要转置
        if nano_sd[attn_c_attn_w].shape != hf_sd[attn_c_attn_w].t().shape:
            print(f"WARNING: Shape mismatch for {attn_c_attn_w}")
            
def compare_attention_outputs(nano_model, hf_model, input_ids, device):
    # 获取中间层输出
    nano_outputs = []
    hf_outputs = []
    
    # 为每个注意力层添加钩子
    nano_hooks = []
    for i, block in enumerate(nano_model.transformer.h):
        hook = block.attn.register_forward_hook(lambda m, i, o, idx=i: nano_outputs.append((idx, o)))
        nano_hooks.append(hook)
    
    hf_hooks = []
    for i, block in enumerate(hf_model.transformer.h):
        # 修改这里的钩子函数来处理元组输出
        hook = block.attn.register_forward_hook(lambda m, i, o, idx=i: hf_outputs.append((idx, o[0] if isinstance(o, tuple) else o)))
        hf_hooks.append(hook)
    
    # 运行前向传播
    with torch.no_grad():
        nano_model(input_ids)
        hf_model(input_ids)
    
    # 移除钩子
    for hook in nano_hooks:
        hook.remove()
    for hook in hf_hooks:
        hook.remove()
    
    # 比较每层的输出
    nano_outputs.sort(key=lambda x: x[0])
    hf_outputs.sort(key=lambda x: x[0])
    
    for (nano_idx, nano_out), (hf_idx, hf_out) in zip(nano_outputs, hf_outputs):
        assert nano_idx == hf_idx
        # 确保我们比较的是张量而不是元组
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]  # 取元组的第一个元素，通常是注意力输出
        
        diff = (nano_out - hf_out).abs().max().item()
        print(f"Layer {nano_idx} attention output diff: {diff:.6f}")

# 检查模型架构差异
def debug_architecture_differences(nano_model, hf_model, gptconf, mapping):
    print("\n=== Architecture Comparison ===")
    
    # 检查关键配置
    print(f"nanoGPT config: n_layer={gptconf.n_layer}, n_head={gptconf.n_head}, n_embd={gptconf.n_embd}")
    print(f"HF config: n_layer={hf_model.config.n_layer}, n_head={hf_model.config.n_head}, n_embd={hf_model.config.n_embd}")
    
    # 检查激活函数
    print(f"HF activation function: {hf_model.config.activation_function}")
    
    # 检查层归一化参数
    print(f"nanoGPT layer norm epsilon: 1e-5 (hardcoded)")
    print(f"HF layer norm epsilon: {hf_model.config.layer_norm_epsilon}")
    
    # 检查注意力机制
    print(f"nanoGPT attention: {'flash attention' if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 'manual attention'}")
    print(f"HF attention: scaled_dot_product")
    
    # 检查权重形状
    nano_sd = nano_model.state_dict()
    hf_sd = hf_model.state_dict()
    
    print("\nKey weight shapes comparison:")
    for nano_key in nano_sd:
        if nano_key in mapping:
            hf_key = mapping[nano_key]
            print(f"{nano_key}: {nano_sd[nano_key].shape} -> {hf_key}: {hf_sd[hf_key].shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert nanoGPT checkpoint to Hugging Face format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to nanoGPT checkpoint")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on")    
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    
    convert_to_huggingface(checkpoint_path, output_dir, args.device)