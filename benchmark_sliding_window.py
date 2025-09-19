"""
Comprehensive benchmark comparing sliding window attention vs full attention.
"""
import torch
import torch.nn.functional as F
import time
from model_te_final_v2 import FinalGPT2Model, FinalConfig, get_sliding_window_config, get_gpt2_small_config

def benchmark_attention_patterns():
    """Compare different attention patterns and their performance."""
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("ATTENTION PATTERN COMPARISON")
    print("=" * 80)

    # Configurations to test
    configs_to_test = [
        ("Full Attention", get_gpt2_small_config()),
        ("Sliding Window (512)", get_sliding_window_config()),
        ("Sliding Window (256)", create_custom_sliding_config(256)),
        ("Sliding Window (128)", create_custom_sliding_config(128)),
        ("Mixed Pattern", create_mixed_pattern_config()),
    ]

    # Test with different sequence lengths
    test_cases = [
        (8, 512),
        (8, 1024),
        (8, 2048),
        (4, 4096),  # Very long sequence
    ]

    results = []

    for config_name, config in configs_to_test:
        print(f"\n{config_name}:")
        print("-" * 60)

        if hasattr(config, 'use_sliding_window') and config.use_sliding_window:
            print(f"  Window size: {config.sliding_window_size}")
            print(f"  Sliding layers: {config.sliding_window_layers}")

        for batch_size, seq_len in test_cases:
            print(f"\n  Testing BS={batch_size}, Seq={seq_len}:")

            try:
                model = FinalGPT2Model(config).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
                labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

                # Memory before
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                # Warmup
                for _ in range(5):
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits, loss = model(input_ids, labels=labels)
                    loss.backward()
                    optimizer.step()

                # Benchmark
                torch.cuda.synchronize()
                times = []

                for _ in range(20):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits, loss = model(input_ids, labels=labels)
                    loss.backward()
                    optimizer.step()

                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

                avg_time = sum(times) / len(times)
                tokens_per_sec = (batch_size * seq_len) / avg_time
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

                print(f"    Tokens/sec: {tokens_per_sec:,.0f}")
                print(f"    Memory: {peak_memory:.0f} MB")
                print(f"    ms/iter: {avg_time * 1000:.1f}")

                results.append({
                    'config': config_name,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'tokens_per_sec': tokens_per_sec,
                    'memory_mb': peak_memory,
                    'ms_per_iter': avg_time * 1000
                })

                del model
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"    OOM - sequence too long for this configuration")
                results.append({
                    'config': config_name,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'tokens_per_sec': 0,
                    'memory_mb': float('inf'),
                    'ms_per_iter': float('inf')
                })
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find best config for each sequence length
    for _, seq_len in test_cases:
        seq_results = [r for r in results if r['seq_len'] == seq_len and r['tokens_per_sec'] > 0]
        if seq_results:
            best = max(seq_results, key=lambda x: x['tokens_per_sec'])
            print(f"\nBest for Seq={seq_len}: {best['config']}")
            print(f"  Speed: {best['tokens_per_sec']:,.0f} tok/s")
            print(f"  Memory: {best['memory_mb']:.0f} MB")

    # Compare sliding window vs full attention
    print("\n" + "=" * 80)
    print("SLIDING WINDOW ANALYSIS")
    print("=" * 80)

    full_results = [r for r in results if r['config'] == "Full Attention"]
    sliding_512 = [r for r in results if "512" in r['config']]

    for seq_len in [512, 1024, 2048]:
        full = next((r for r in full_results if r['seq_len'] == seq_len), None)
        slide = next((r for r in sliding_512 if r['seq_len'] == seq_len), None)

        if full and slide and full['tokens_per_sec'] > 0 and slide['tokens_per_sec'] > 0:
            speed_ratio = slide['tokens_per_sec'] / full['tokens_per_sec']
            memory_ratio = slide['memory_mb'] / full['memory_mb']
            print(f"\nSeq={seq_len}:")
            print(f"  Speed ratio (sliding/full): {speed_ratio:.3f}x")
            print(f"  Memory ratio (sliding/full): {memory_ratio:.3f}x")


def create_custom_sliding_config(window_size):
    """Create a sliding window config with custom window size."""
    config = get_gpt2_small_config()
    config.use_sliding_window = True
    config.sliding_window_size = window_size
    # Use sliding window on all layers for aggressive memory savings
    config.sliding_window_layers = list(range(config.n_layer))
    return config


def create_mixed_pattern_config():
    """Create a mixed attention pattern:
    - First 2 layers: full attention (capture global context)
    - Middle layers: sliding window (efficient processing)
    - Last 2 layers: full attention (final aggregation)
    """
    config = get_gpt2_small_config()
    config.use_sliding_window = True
    config.sliding_window_size = 512
    # Use sliding window only in middle layers (2-9)
    config.sliding_window_layers = list(range(2, 10))
    return config


def analyze_attention_pattern_distribution():
    """Analyze how different attention patterns affect computation."""
    print("\n" + "=" * 80)
    print("ATTENTION PATTERN DISTRIBUTION ANALYSIS")
    print("=" * 80)

    config = get_gpt2_small_config()

    patterns = [
        ("All Full", []),
        ("All Sliding", list(range(12))),
        ("Alternating", [i for i in range(12) if i % 2 == 1]),
        ("Bottom Heavy", [0, 1, 2, 3, 4, 5]),  # First half sliding
        ("Top Heavy", [6, 7, 8, 9, 10, 11]),   # Second half sliding
        ("Sandwich", [2, 3, 4, 5, 6, 7, 8, 9]), # Middle sliding
    ]

    seq_len = 2048
    window_size = 512

    for pattern_name, sliding_layers in patterns:
        full_layers = [i for i in range(12) if i not in sliding_layers]

        # Calculate theoretical FLOP reduction
        # Full attention: O(n^2) per layer
        # Sliding window: O(n*w) per layer where w is window size
        full_flops = len(full_layers) * (seq_len ** 2)
        sliding_flops = len(sliding_layers) * (seq_len * window_size)
        total_flops = full_flops + sliding_flops
        baseline_flops = 12 * (seq_len ** 2)
        flop_reduction = 1 - (total_flops / baseline_flops)

        print(f"\n{pattern_name}:")
        print(f"  Full attention layers: {full_layers}")
        print(f"  Sliding window layers: {sliding_layers}")
        print(f"  Theoretical FLOP reduction: {flop_reduction * 100:.1f}%")


if __name__ == "__main__":
    benchmark_attention_patterns()
    analyze_attention_pattern_distribution()