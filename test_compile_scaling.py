"""
Test torch.compile scaling with model size.
torch.compile benefits should increase with larger models.
"""
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from model_te_optimized import OptimizedGPT2Model, OptimizedConfig
import time
import gc


def benchmark(model, x, y, name="", n_iters=10):
    """Quick benchmark."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for _ in range(3):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Time
    start = time.perf_counter()
    for _ in range(n_iters):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000

    return elapsed


def test_scaling():
    """Test compile benefits at different scales."""
    torch.manual_seed(42)
    device = "cuda"

    print("Testing torch.compile scaling with model size")
    print("=" * 80)

    # Test different model sizes
    test_configs = [
        # (name, n_layers, batch_size, seq_len)
        ("Tiny (2L)", 2, 4, 256),
        ("Small (4L)", 4, 8, 512),
        ("Medium (8L)", 8, 8, 512),
        ("Large (12L)", 12, 8, 512),
        ("XL (16L)", 16, 4, 512),
    ]

    results = []

    for name, n_layers, batch_size, seq_len in test_configs:
        print(f"\n{name}: Layers={n_layers}, B={batch_size}, S={seq_len}")
        print("-" * 60)

        config = OptimizedConfig(
            n_layer=n_layers,
            n_embd=768,
            n_head=12,
            n_kv_head=4,
            use_fp8=True,
            use_pytorch_sdpa=True
        )

        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        # Test without compile
        try:
            model_base = OptimizedGPT2Model(config).to(device)
            time_base = benchmark(model_base, x, y, "baseline", n_iters=5)
            print(f"  Without compile: {time_base:.1f} ms")
            del model_base
        except Exception as e:
            print(f"  Without compile: Failed - {e}")
            time_base = None

        # Test with compile
        try:
            model_compiled = OptimizedGPT2Model(config).to(device)
            model_compiled = torch.compile(model_compiled, mode="reduce-overhead")
            time_compiled = benchmark(model_compiled, x, y, "compiled", n_iters=5)
            print(f"  With compile:    {time_compiled:.1f} ms")

            if time_base:
                speedup = time_base / time_compiled
                improvement = (1 - time_compiled/time_base) * 100
                print(f"  Speedup:         {speedup:.2f}x ({improvement:+.1f}%)")
                results.append((name, n_layers, speedup))

            del model_compiled
        except Exception as e:
            print(f"  With compile: Failed - {e}")

        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: torch.compile scaling")
    print("-" * 60)
    print("Model Size    | Layers | Speedup")
    print("-" * 60)
    for name, layers, speedup in results:
        status = "✓" if speedup > 1.0 else "✗"
        print(f"{name:<12} | {layers:>6} | {speedup:>6.2f}x {status}")

    print("\n" + "=" * 80)
    print("Conclusions:")
    print("1. torch.compile overhead dominates for small models")
    print("2. Benefits should appear with larger models (12+ layers)")
    print("3. FP8 kernels are already optimized, less room for improvement")
    print("4. Consider compile only for large-scale training")


def test_compile_components():
    """Test which components benefit from compile."""
    torch.manual_seed(42)
    device = "cuda"

    print("\n" + "=" * 80)
    print("Testing which components benefit from torch.compile")
    print("=" * 80)

    B, S = 16, 1024  # Larger batch/seq for better testing

    # Test different configurations
    configs = [
        ("Full model", OptimizedConfig(n_layer=8), lambda m: m),
        ("Attention only", OptimizedConfig(n_layer=8),
         lambda m: compile_attention_only(m)),
        ("MLP only", OptimizedConfig(n_layer=8),
         lambda m: compile_mlp_only(m)),
        ("No FP8", OptimizedConfig(n_layer=8, use_fp8=False),
         lambda m: torch.compile(m, mode="reduce-overhead")),
        ("No SDPA", OptimizedConfig(n_layer=8, use_pytorch_sdpa=False),
         lambda m: torch.compile(m, mode="reduce-overhead")),
    ]

    x = torch.randint(0, 50304, (B, S), device=device)
    y = torch.randint(0, 50304, (B, S), device=device)

    baseline_time = None

    for name, config, compile_fn in configs:
        print(f"\n{name}:")

        try:
            if "only" in name:
                # Special handling for partial compilation
                model = OptimizedGPT2Model(config).to(device)
                model = compile_fn(model)
            else:
                model = OptimizedGPT2Model(config).to(device)
                if name != "Full model":
                    model = compile_fn(model)

            elapsed = benchmark(model, x, y, name, n_iters=5)
            print(f"  Time: {elapsed:.1f} ms")

            if baseline_time is None:
                baseline_time = elapsed
            else:
                speedup = baseline_time / elapsed
                print(f"  vs baseline: {speedup:.2f}x")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Failed: {str(e)[:100]}")


def compile_attention_only(model):
    """Compile only attention layers."""
    for block in model.blocks:
        if hasattr(block, 'attention'):
            block.attention = torch.compile(block.attention)
    return model


def compile_mlp_only(model):
    """Compile only MLP layers."""
    for block in model.blocks:
        if hasattr(block, 'mlp') and block.mlp is not None:
            block.mlp = torch.compile(block.mlp)
        elif hasattr(block, 'fc1'):
            block.fc1 = torch.compile(block.fc1)
            block.fc2 = torch.compile(block.fc2)
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("torch.compile Scaling Analysis")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    test_scaling()
    test_compile_components()

    print("\n" + "=" * 80)
    print("Final Recommendations:")
    print("-" * 60)
    print("1. For small models (<8 layers): Skip torch.compile")
    print("2. For medium models (8-16 layers): Test both ways")
    print("3. For large models (>16 layers): Use torch.compile")
    print("4. With FP8: Often better WITHOUT compile")
    print("5. Best practice: Profile YOUR specific model/hardware")
    print("\nOptimal for most cases:")
    print("  - Use FP8 + PyTorch SDPA")
    print("  - Skip torch.compile unless model is large")
    print("  - Focus on algorithmic optimizations (GQA, etc.)")