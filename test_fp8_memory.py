"""
Test to verify FP8 actually reduces memory usage.
"""
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format


def check_memory_with_fp8():
    """Compare memory usage between BF16 and FP8."""
    device = "cuda"
    batch_size = 8
    seq_len = 512
    hidden_size = 768

    print("=" * 60)
    print("FP8 MEMORY TEST")
    print("=" * 60)

    # Test 1: BF16 model
    print("\n1. BF16 Model:")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model_bf16 = nn.Sequential(
        te.Linear(hidden_size, 3072, bias=True),
        te.Linear(3072, hidden_size, bias=True),
        te.Linear(hidden_size, 3072, bias=True),
        te.Linear(3072, hidden_size, bias=True),
    ).to(device).to(torch.bfloat16)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    optimizer_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=3e-4)

    # Run forward and backward
    y = model_bf16(x)
    loss = y.sum()
    loss.backward()
    optimizer_bf16.step()

    torch.cuda.synchronize()
    mem_bf16 = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Count parameters
    n_params = sum(p.numel() for p in model_bf16.parameters())
    param_memory_bf16 = n_params * 2 / 1024 / 1024  # 2 bytes per param

    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Param memory (theoretical): {param_memory_bf16:.1f} MB")
    print(f"  Peak memory: {mem_bf16:.1f} MB")

    del model_bf16, x, y, loss, optimizer_bf16
    torch.cuda.empty_cache()

    # Test 2: FP8 model (without proper init)
    print("\n2. FP8 Model (naive):")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model_fp8_naive = nn.Sequential(
        te.Linear(hidden_size, 3072, bias=True),
        te.Linear(3072, hidden_size, bias=True),
        te.Linear(hidden_size, 3072, bias=True),
        te.Linear(3072, hidden_size, bias=True),
    ).to(device).to(torch.bfloat16)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    optimizer_fp8_naive = torch.optim.AdamW(model_fp8_naive.parameters(), lr=3e-4)

    fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)

    # Run with FP8 autocast
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        y = model_fp8_naive(x)
        loss = y.sum()
    loss.backward()
    optimizer_fp8_naive.step()

    torch.cuda.synchronize()
    mem_fp8_naive = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Peak memory: {mem_fp8_naive:.1f} MB")
    print(f"  vs BF16: {mem_fp8_naive/mem_bf16:.2f}x")

    # Check for FP8 weights
    for name, module in model_fp8_naive.named_modules():
        if isinstance(module, te.Linear):
            print(f"  Module {name}:")
            print(f"    Has fp8_meta: {hasattr(module, 'fp8_meta')}")
            print(f"    Has weight_fp8: {hasattr(module, 'weight_fp8')}")
            if hasattr(module, 'fp8_meta'):
                print(f"    FP8 meta keys: {module.fp8_meta.keys() if hasattr(module.fp8_meta, 'keys') else 'N/A'}")
            break

    del model_fp8_naive, x, y, loss, optimizer_fp8_naive
    torch.cuda.empty_cache()

    # Test 3: FP8 with proper initialization
    print("\n3. FP8 Model (with calibration):")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model_fp8 = nn.Sequential(
        te.Linear(hidden_size, 3072, bias=True),
        te.Linear(3072, hidden_size, bias=True),
        te.Linear(hidden_size, 3072, bias=True),
        te.Linear(3072, hidden_size, bias=True),
    ).to(device).to(torch.bfloat16)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    optimizer_fp8 = torch.optim.AdamW(model_fp8.parameters(), lr=3e-4)

    # Calibration phase - run a few iterations to collect statistics
    print("  Running calibration...")
    fp8_recipe = DelayedScaling(
        margin=0,
        fp8_format=Format.HYBRID,
        amax_history_len=32,  # Shorter history for quicker calibration
        amax_compute_algo="max"
    )

    for i in range(10):
        optimizer_fp8.zero_grad()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = model_fp8(x)
            loss = y.sum()
        loss.backward()
        optimizer_fp8.step()

    # Now check memory after calibration
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for i in range(10):
        optimizer_fp8.zero_grad()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, calibration=False):
            y = model_fp8(x)
            loss = y.sum()
        loss.backward()
        optimizer_fp8.step()

    torch.cuda.synchronize()
    mem_fp8_calibrated = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Peak memory: {mem_fp8_calibrated:.1f} MB")
    print(f"  vs BF16: {mem_fp8_calibrated/mem_bf16:.2f}x")

    # Check FP8 internals
    for name, module in model_fp8.named_modules():
        if isinstance(module, te.Linear):
            print(f"  Module {name} after calibration:")
            print(f"    Has fp8_meta: {hasattr(module, 'fp8_meta')}")
            print(f"    Weight dtype: {module.weight.dtype}")
            if hasattr(module, 'fp8_meta') and hasattr(module.fp8_meta, '__dict__'):
                for key in ['scaling_fwd', 'scaling_bwd', 'amax_history']:
                    if hasattr(module.fp8_meta, key):
                        attr = getattr(module.fp8_meta, key)
                        print(f"    {key}: exists (type: {type(attr).__name__})")
            break

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"BF16 memory:           {mem_bf16:.1f} MB")
    print(f"FP8 naive memory:      {mem_fp8_naive:.1f} MB ({mem_fp8_naive/mem_bf16:.2f}x)")
    print(f"FP8 calibrated memory: {mem_fp8_calibrated:.1f} MB ({mem_fp8_calibrated/mem_bf16:.2f}x)")
    print("=" * 60)

    if mem_fp8_calibrated >= mem_bf16:
        print("\n⚠️ WARNING: FP8 is NOT reducing memory!")
        print("This suggests FP8 quantization is not actually happening.")
        print("Possible reasons:")
        print("1. GPU doesn't support FP8 (requires Hopper/Ada)")
        print("2. TransformerEngine FP8 is only for computation, not storage")
        print("3. FP8 metadata overhead exceeds savings")
    else:
        print(f"\n✅ FP8 reduces memory by {(1 - mem_fp8_calibrated/mem_bf16)*100:.1f}%")


if __name__ == "__main__":
    check_memory_with_fp8()