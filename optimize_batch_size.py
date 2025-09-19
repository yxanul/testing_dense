"""
Script to find optimal batch size and gradient accumulation for RTX 5090.
Tests different configurations to maximize throughput.
"""
import torch
import time
from model_te_gated import GatedGPT2Model, get_gated_gpt2_config
import transformer_engine.pytorch as te


def benchmark_config(batch_size, seq_length, grad_acc_steps, use_fp8=True, use_gated=True):
    """Benchmark a specific configuration."""
    device = "cuda"

    # Create model
    config = get_gated_gpt2_config(
        vocab_size=32768,
        use_gated_attention=use_gated,
        use_fp8=use_fp8
    )
    model = GatedGPT2Model(config).to(device)
    model = model.to(dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for _ in range(5):
        input_ids = torch.randint(0, 32768, (batch_size, seq_length)).to(device)
        labels = input_ids

        for _ in range(grad_acc_steps):
            logits, loss = model(input_ids, labels=labels)
            loss = loss / grad_acc_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    num_steps = 10
    for _ in range(num_steps):
        input_ids = torch.randint(0, 32768, (batch_size, seq_length)).to(device)
        labels = input_ids

        for _ in range(grad_acc_steps):
            logits, loss = model(input_ids, labels=labels)
            loss = loss / grad_acc_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Calculate throughput
    total_tokens = batch_size * seq_length * grad_acc_steps * num_steps
    tokens_per_sec = total_tokens / elapsed

    # Check memory
    memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return tokens_per_sec, memory_gb, elapsed / num_steps


def main():
    print("="*70)
    print("FINDING OPTIMAL BATCH SIZE FOR RTX 5090")
    print("="*70)

    # Configurations to test
    configs = [
        # (batch_size, seq_length, grad_acc_steps)
        (8, 2048, 1),    # Smaller batch
        (12, 2048, 1),   # Your current config
        (16, 2048, 1),   # Larger batch
        (8, 2048, 2),    # Grad acc = 2
        (6, 2048, 2),    # Smaller with grad acc
        (4, 2048, 3),    # Even smaller with more grad acc
        (12, 1024, 1),   # Shorter sequence
        (16, 1024, 1),   # Shorter sequence, larger batch
    ]

    print("\nTesting configurations...")
    print("-"*70)
    print(f"{'Config':<30} {'Tokens/s':<15} {'Memory (GB)':<12} {'Time/step':<10}")
    print("-"*70)

    results = []
    for bs, seq, ga in configs:
        try:
            tokens_per_sec, memory_gb, time_per_step = benchmark_config(bs, seq, ga)

            config_str = f"BS={bs}, Seq={seq}, GA={ga}"
            print(f"{config_str:<30} {tokens_per_sec:>14,.0f} {memory_gb:>11.2f} {time_per_step:>9.3f}s")

            results.append({
                'batch_size': bs,
                'seq_length': seq,
                'grad_acc': ga,
                'effective_batch': bs * ga,
                'tokens_per_sec': tokens_per_sec,
                'memory_gb': memory_gb,
                'time_per_step': time_per_step
            })

            # Clear cache
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{f'BS={bs}, Seq={seq}, GA={ga}':<30} {'OOM':<15}")
            else:
                raise e

    print("-"*70)

    # Find best config
    if results:
        best = max(results, key=lambda x: x['tokens_per_sec'])
        print(f"\nBest configuration:")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Sequence length: {best['seq_length']}")
        print(f"  Gradient accumulation: {best['grad_acc']}")
        print(f"  Effective batch size: {best['effective_batch']}")
        print(f"  Tokens/sec: {best['tokens_per_sec']:,.0f}")
        print(f"  Memory usage: {best['memory_gb']:.2f} GB")

        print(f"\nTo use this configuration:")
        print(f"  python train_gated.py --batch_size {best['batch_size']} "
              f"--sequence_length {best['seq_length']} "
              f"--gradient_accumulation_steps {best['grad_acc']}")


if __name__ == "__main__":
    main()