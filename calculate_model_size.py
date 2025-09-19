"""
Calculate model sizes for different configurations.
"""
import torch
import torch.nn as nn
from model_te_final import FinalGPT2Model, get_gpt2_small_config, get_gpt2_medium_config, get_gpt2_large_config, get_rtx5090_optimized_config

def calculate_model_size(model):
    """Calculate total parameters and model size in MB."""
    total_params = 0
    param_details = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params = param.numel()
        total_params += params

        # Group by component
        if 'wte' in name:
            param_details['embeddings'] = param_details.get('embeddings', 0) + params
        elif 'wpe' in name:
            param_details['pos_embeddings'] = param_details.get('pos_embeddings', 0) + params
        elif 'ln_f' in name or 'norm' in name:
            param_details['layer_norms'] = param_details.get('layer_norms', 0) + params
        elif 'attn' in name:
            param_details['attention'] = param_details.get('attention', 0) + params
        elif 'mlp' in name or 'fc' in name or 'gate' in name or 'up' in name or 'down' in name:
            param_details['mlp'] = param_details.get('mlp', 0) + params
        elif 'lm_head' in name:
            param_details['lm_head'] = param_details.get('lm_head', 0) + params
        else:
            param_details['other'] = param_details.get('other', 0) + params

    # Size in MB (assuming BF16 = 2 bytes per param)
    size_mb = total_params * 2 / (1024 * 1024)

    return total_params, size_mb, param_details

def main():
    print("=" * 80)
    print("MODEL SIZE ANALYSIS")
    print("=" * 80)

    configs = [
        ("GPT-2 Small (current)", get_gpt2_small_config()),
        ("GPT-2 Medium", get_gpt2_medium_config()),
        ("GPT-2 Large", get_gpt2_large_config()),
        ("RTX 5090 Optimized", get_rtx5090_optimized_config()),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        print("-" * 60)

        # Key config details
        print(f"  Layers: {config.n_layer}")
        print(f"  Hidden size: {config.n_embd}")
        print(f"  Attention heads: {config.n_head}")
        print(f"  KV heads: {config.n_kv_head} (GQA ratio {config.n_head}:{config.n_kv_head})")
        print(f"  FFN hidden: {config.ffn_hidden_size}")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  MLP type: {config.mlp_type}")

        # Create model and calculate size
        model = FinalGPT2Model(config)
        total_params, size_mb, details = calculate_model_size(model)

        print(f"\n  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Model size (BF16): {size_mb:.1f} MB")
        print(f"  Model size (FP32): {size_mb*2:.1f} MB")

        print("\n  Parameter breakdown:")
        for component, count in sorted(details.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_params * 100
            print(f"    {component:<15}: {count:>12,} ({pct:>5.1f}%)")

        # Memory estimation for training
        # Rough estimate: model weights + gradients + optimizer states (Adam uses 2x params)
        # Total = weights + gradients + 2*optimizer = 4x model size
        training_memory_mb = size_mb * 4
        print(f"\n  Estimated training memory (Adam): {training_memory_mb:.1f} MB")

        # With batch size 12, seq 2048
        batch_size = 12
        seq_len = 2048
        # Activations: roughly batch_size * seq_len * hidden_size * n_layers * 4 (for intermediate)
        activation_memory = (batch_size * seq_len * config.n_embd * config.n_layer * 4 * 2) / (1024 * 1024)
        print(f"  Activation memory (BS=12, Seq=2048): {activation_memory:.1f} MB")
        print(f"  Total training memory estimate: {training_memory_mb + activation_memory:.1f} MB")

    print("\n" + "=" * 80)
    print("NOTES:")
    print("-" * 80)
    print("1. Model sizes assume BF16 (2 bytes per parameter)")
    print("2. Training memory = 4x model size (weights + gradients + Adam states)")
    print("3. Activation memory scales with batch size and sequence length")
    print("4. GQA reduces KV cache memory by the compression ratio")
    print("5. SwiGLU MLP uses fewer parameters than vanilla MLP (2/3 ratio)")
    print("=" * 80)

if __name__ == "__main__":
    main()