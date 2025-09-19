"""
Simple tokens/second benchmark for the final model.
"""
import torch
import torch.nn.functional as F
from model_te_final import FinalGPT2Model, FinalConfig
import time


def benchmark_tokens_per_second():
    """Simple benchmark to measure tokens/second."""
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("TOKENS/SECOND BENCHMARK - Final Optimized Model")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Configuration (with optimized vocab_size)
    config = FinalConfig(
        vocab_size=32768,  # Power of 2 - better for consumer GPUs
        n_layer=12,
        n_head=12,
        n_kv_head=4,  # GQA 4:1 ratio
        mlp_type="swiglu",
        use_fused_qkv=True,   # ✅ Fused QKV (1.84x)
        use_fused_mlp=False,  # ❌ No MLP fusion (slower)
        use_pytorch_sdpa=True,  # ✅ SDPA (10x)
        use_fp8=True  # ✅ FP8 (1.2x)
    )

    model = FinalGPT2Model(config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")
    print()

    # Test different configurations
    test_configs = [
        (1, 512),   # Small
        (2, 512),   # Small batch
        (4, 512),   # Medium batch
        (8, 512),   # Large batch
        (16, 512),  # Very large batch
        (8, 256),   # Short sequence
        (8, 1024),  # Long sequence
        (4, 2048),  # Very long sequence
    ]

    print(f"{'Batch':<8} {'Seq':<8} {'Tokens':<10} {'ms/iter':<10} {'Tokens/sec':<15} {'GB/s':<10}")
    print("-" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for batch_size, seq_len in test_configs:
        try:
            # Prepare data
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                logits, loss = model(input_ids, labels=labels)
                if loss is None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        labels.reshape(-1)
                    )
                loss.backward()
                optimizer.step()

            # Benchmark
            torch.cuda.synchronize()

            times = []
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()

                optimizer.zero_grad()
                logits, loss = model(input_ids, labels=labels)
                if loss is None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        labels.reshape(-1)
                    )
                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            # Calculate metrics
            avg_time = sum(times) / len(times)
            ms_per_iter = avg_time * 1000
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / avg_time

            # Estimate memory bandwidth (rough)
            # Each token processes ~4 * n_params bytes (forward + backward + optimizer)
            bytes_per_token = 4 * n_params * 2  # bfloat16 = 2 bytes
            gb_per_sec = (bytes_per_token * tokens_per_sec) / 1e9

            print(f"{batch_size:<8} {seq_len:<8} {total_tokens:<10} "
                  f"{ms_per_iter:<10.2f} {tokens_per_sec:<15,.0f} {gb_per_sec:<10.1f}")

        except Exception as e:
            print(f"{batch_size:<8} {seq_len:<8} {'Failed':<10} {str(e)[:40]}")

    print()
    print("=" * 80)
    print("Optimizations applied:")
    print("✅ Fused QKV projection (3→1 GEMM)")
    print("✅ PyTorch SDPA (10x faster attention)")
    print("✅ FP8 HYBRID (E4M3 fwd, E5M2 bwd)")
    print("✅ GQA 4:1 (memory efficient)")
    print("✅ SwiGLU activation")
    print("❌ NO MLP fusion (benchmarked slower)")
    print("❌ NO torch.compile (slower with FP8)")
    print("=" * 80)


def compare_configurations():
    """Compare different optimization combinations."""
    torch.manual_seed(42)
    device = "cuda"

    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)

    batch_size = 8
    seq_len = 512
    total_tokens = batch_size * seq_len

    configs_to_test = [
        ("Baseline (no opts)", FinalConfig(
            n_layer=12,
            use_fused_qkv=False,
            use_pytorch_sdpa=False,
            use_fp8=False,
            mlp_type="vanilla"
        )),
        ("+ Fused QKV", FinalConfig(
            n_layer=12,
            use_fused_qkv=True,
            use_pytorch_sdpa=False,
            use_fp8=False,
            mlp_type="vanilla"
        )),
        ("+ SDPA", FinalConfig(
            n_layer=12,
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=False,
            mlp_type="vanilla"
        )),
        ("+ FP8", FinalConfig(
            n_layer=12,
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=True,
            mlp_type="vanilla"
        )),
        ("+ SwiGLU", FinalConfig(
            n_layer=12,
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=True,
            mlp_type="swiglu",
            use_fused_mlp=False
        )),
        ("+ GQA 4:1", FinalConfig(
            n_layer=12,
            n_kv_head=3,  # 12/4 = 3
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=True,
            mlp_type="swiglu",
            use_fused_mlp=False
        )),
        ("FINAL (all opts)", FinalConfig(
            n_layer=12,
            n_kv_head=4,
            use_fused_qkv=True,
            use_pytorch_sdpa=True,
            use_fp8=True,
            mlp_type="swiglu",
            use_fused_mlp=False
        )),
    ]

    print(f"Test configuration: B={batch_size}, S={seq_len}, Tokens={total_tokens}")
    print()
    print(f"{'Configuration':<25} {'ms/iter':<12} {'Tokens/sec':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline_tokens = None

    for name, config in configs_to_test:
        try:
            model = FinalGPT2Model(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                logits, loss = model(input_ids, labels=labels)
                if loss is None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        labels.reshape(-1)
                    )
                loss.backward()
                optimizer.step()

            # Benchmark
            torch.cuda.synchronize()
            times = []

            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()

                optimizer.zero_grad()
                logits, loss = model(input_ids, labels=labels)
                if loss is None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        labels.reshape(-1)
                    )
                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            ms_per_iter = avg_time * 1000
            tokens_per_sec = total_tokens / avg_time

            if baseline_tokens is None:
                baseline_tokens = tokens_per_sec
                speedup = 1.0
            else:
                speedup = tokens_per_sec / baseline_tokens

            print(f"{name:<25} {ms_per_iter:<12.2f} {tokens_per_sec:<15,.0f} {speedup:<10.2f}x")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{name:<25} Failed: {str(e)[:40]}")


if __name__ == "__main__":
    benchmark_tokens_per_second()
    compare_configurations()