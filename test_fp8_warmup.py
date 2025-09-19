"""
Test if FP8 needs more warmup iterations to show benefits.
FP8 uses delayed scaling which collects statistics over time.
"""
import torch
import torch.nn.functional as F
from model_te_final import FinalGPT2Model, FinalConfig

# Use optimized vocab size
VOCAB_SIZE = 32768  # Power of 2, better for GPUs
import time
import matplotlib.pyplot as plt


def benchmark_with_warmup(model, batch_size, seq_len, vocab_size, n_warmup, n_bench):
    """Benchmark with specified warmup iterations."""
    device = "cuda"

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    all_times = []

    # Track performance over iterations
    for i in range(n_warmup + n_bench):
        torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad()
        logits, loss = model(input_ids, labels=labels)
        if loss is None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1)
            )
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        all_times.append(elapsed)

    # Calculate metrics
    warmup_times = all_times[:n_warmup]
    bench_times = all_times[n_warmup:]

    # Average performance after warmup
    avg_after_warmup = sum(bench_times) / len(bench_times) if bench_times else 0
    tokens_per_sec = (batch_size * seq_len) / avg_after_warmup if avg_after_warmup > 0 else 0

    return all_times, tokens_per_sec


def test_fp8_warmup():
    """Test if FP8 improves with more warmup."""
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("FP8 WARMUP ANALYSIS")
    print("=" * 80)
    print("Testing if FP8 needs more iterations to show benefits")
    print()

    batch_size = 8
    seq_len = 512
    total_tokens = batch_size * seq_len

    # Test configurations
    configs = [
        ("BF16 (no FP8)", FinalConfig(
            n_layer=12,
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=False,  # No FP8
            mlp_type="swiglu"
        )),
        ("FP8 HYBRID", FinalConfig(
            n_layer=12,
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=True,  # With FP8
            mlp_type="swiglu"
        )),
    ]

    # Test with different warmup lengths
    warmup_lengths = [3, 10, 20, 50, 100]
    n_bench = 50  # Benchmark iterations after warmup

    print(f"Configuration: B={batch_size}, S={seq_len}, Tokens={total_tokens}")
    print(f"Benchmark iterations: {n_bench}")
    print()

    results = {}

    for name, config in configs:
        print(f"\nTesting: {name}")
        print("-" * 40)

        config_results = []

        for n_warmup in warmup_lengths:
            model = FinalGPT2Model(config).to(device)

            times, tokens_per_sec = benchmark_with_warmup(
                model, batch_size, seq_len, config.vocab_size,
                n_warmup, n_bench
            )

            print(f"  Warmup {n_warmup:3d} iters: {tokens_per_sec:8,.0f} tokens/sec")

            config_results.append({
                'warmup': n_warmup,
                'tokens_per_sec': tokens_per_sec,
                'times': times
            })

            del model
            torch.cuda.empty_cache()

        results[name] = config_results

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\n{'Warmup':<10} {'BF16':<15} {'FP8':<15} {'FP8/BF16':<10}")
    print("-" * 50)

    for i, n_warmup in enumerate(warmup_lengths):
        bf16_tps = results["BF16 (no FP8)"][i]['tokens_per_sec']
        fp8_tps = results["FP8 HYBRID"][i]['tokens_per_sec']
        ratio = fp8_tps / bf16_tps if bf16_tps > 0 else 0

        print(f"{n_warmup:<10} {bf16_tps:<15,.0f} {fp8_tps:<15,.0f} {ratio:<10.2f}x")

    # Test very long run to see if FP8 eventually wins
    print("\n" + "=" * 80)
    print("LONG RUN TEST (warmup=100, bench=200)")
    print("=" * 80)

    for name, config in configs:
        model = FinalGPT2Model(config).to(device)

        times, tokens_per_sec = benchmark_with_warmup(
            model, batch_size, seq_len, config.vocab_size,
            100, 200  # Long run
        )

        # Analyze performance over time
        window = 20
        rolling_avg = []
        for i in range(window, len(times)):
            avg = sum(times[i-window:i]) / window
            tps = total_tokens / avg
            rolling_avg.append(tps)

        if rolling_avg:
            early_perf = sum(rolling_avg[:20]) / 20
            late_perf = sum(rolling_avg[-20:]) / 20
            improvement = (late_perf - early_perf) / early_perf * 100

            print(f"\n{name}:")
            print(f"  Early performance:  {early_perf:,.0f} tokens/sec")
            print(f"  Late performance:   {late_perf:,.0f} tokens/sec")
            print(f"  Improvement:        {improvement:+.1f}%")
            print(f"  Final avg:          {tokens_per_sec:,.0f} tokens/sec")

        del model
        torch.cuda.empty_cache()


def test_fp8_recipe_tuning():
    """Test different FP8 recipe configurations."""
    from transformer_engine.common.recipe import DelayedScaling, Format

    print("\n" + "=" * 80)
    print("FP8 RECIPE TUNING")
    print("=" * 80)

    batch_size = 8
    seq_len = 512

    # Different FP8 recipes to test
    recipes = [
        ("E4M3 margin=0", DelayedScaling(margin=0, fp8_format=Format.E4M3)),
        ("E4M3 margin=2", DelayedScaling(margin=2, fp8_format=Format.E4M3)),
        ("HYBRID margin=0", DelayedScaling(margin=0, fp8_format=Format.HYBRID)),
        ("HYBRID amax=32", DelayedScaling(
            margin=0,
            fp8_format=Format.HYBRID,
            amax_history_len=32,
            amax_compute_algo="max"
        )),
        ("HYBRID amax=64", DelayedScaling(
            margin=0,
            fp8_format=Format.HYBRID,
            amax_history_len=64,
            amax_compute_algo="most_recent"
        )),
    ]

    print(f"Testing different FP8 recipes (B={batch_size}, S={seq_len})")
    print(f"Warmup: 50 iterations")
    print()

    for name, recipe in recipes:
        try:
            config = FinalConfig(
                n_layer=12,
                use_fused_qkv=True,
                use_pytorch_sdpa=True,
                use_fp8=True,
                mlp_type="swiglu"
            )

            model = FinalGPT2Model(config).to("cuda")
            # Override the FP8 recipe
            model.fp8_recipe = recipe

            _, tokens_per_sec = benchmark_with_warmup(
                model, batch_size, seq_len, config.vocab_size,
                50, 50  # 50 warmup, 50 bench
            )

            print(f"{name:<20}: {tokens_per_sec:>10,.0f} tokens/sec")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{name:<20}: Failed - {str(e)[:30]}")


if __name__ == "__main__":
    test_fp8_warmup()
    test_fp8_recipe_tuning()