"""
Profile and optimize GEMM operations:
1. Fused QKV projection (1 GEMM instead of 3)
2. Fused Gate-Up-Down for SwiGLU/GeGLU MLPs (1 GEMM instead of 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
import time
from torch.profiler import profile, ProfilerActivity, record_function
from dataclasses import dataclass


@dataclass
class ProfileConfig:
    hidden_size: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden_size: int = 3072
    batch_size: int = 8
    seq_len: int = 512
    use_fp8: bool = True


# ============================================================================
# QKV Projection Variants
# ============================================================================

class SeparateQKV(nn.Module):
    """Traditional separate Q, K, V projections (3 GEMMs)."""
    def __init__(self, config: ProfileConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads

        # Separate projections (3 GEMMs)
        self.q_proj = te.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = te.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = te.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)

    def forward(self, x):
        with record_function("separate_qkv"):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v


class FusedQKV(nn.Module):
    """Fused QKV projection (1 GEMM)."""
    def __init__(self, config: ProfileConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads

        # Fused projection (1 GEMM)
        total_proj_size = (self.hidden_size +  # Q
                          self.n_kv_heads * self.head_dim +  # K
                          self.n_kv_heads * self.head_dim)   # V
        self.qkv_proj = te.Linear(self.hidden_size, total_proj_size, bias=False)

    def forward(self, x):
        with record_function("fused_qkv"):
            qkv = self.qkv_proj(x)

            # Split QKV
            q_size = self.hidden_size
            kv_size = self.n_kv_heads * self.head_dim

            q = qkv[..., :q_size]
            k = qkv[..., q_size:q_size + kv_size]
            v = qkv[..., q_size + kv_size:]

        return q, k, v


# ============================================================================
# MLP Variants
# ============================================================================

class SeparateMLP(nn.Module):
    """Traditional MLP with separate up/down projections."""
    def __init__(self, config: ProfileConfig):
        super().__init__()
        self.fc1 = te.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.fc2 = te.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        with record_function("separate_mlp"):
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
        return x


class FusedGateUpMLP(nn.Module):
    """SwiGLU-style MLP with fused gate and up projections."""
    def __init__(self, config: ProfileConfig):
        super().__init__()
        # Fused gate + up projection (1 GEMM instead of 2)
        self.gate_up_proj = te.Linear(
            config.hidden_size,
            2 * config.ffn_hidden_size,  # gate + up
            bias=False
        )
        self.down_proj = te.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        with record_function("fused_gate_up_mlp"):
            # Single GEMM for gate and up
            gate_up = self.gate_up_proj(x)

            # Split and apply activation
            gate, up = gate_up.chunk(2, dim=-1)
            x = F.silu(gate) * up  # SwiGLU activation

            # Down projection
            x = self.down_proj(x)
        return x


class FusedGeGLUMLP(nn.Module):
    """GeGLU-style MLP with fused gate and up projections."""
    def __init__(self, config: ProfileConfig):
        super().__init__()
        # Fused gate + up projection
        self.gate_up_proj = te.Linear(
            config.hidden_size,
            2 * config.ffn_hidden_size,
            bias=False
        )
        self.down_proj = te.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        with record_function("fused_geglu_mlp"):
            # Single GEMM for gate and up
            gate_up = self.gate_up_proj(x)

            # Split and apply GeGLU
            gate, up = gate_up.chunk(2, dim=-1)
            x = F.gelu(gate) * up  # GeGLU activation

            # Down projection
            x = self.down_proj(x)
        return x


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_module(module, x, name="", n_iters=100, warmup=10, use_fp8=True):
    """Benchmark a module with optional FP8."""
    module.eval()

    if use_fp8:
        fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.E4M3)
    else:
        fp8_recipe = None

    # Warmup
    for _ in range(warmup):
        if fp8_recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _ = module(x)
        else:
            _ = module(x)

    torch.cuda.synchronize()

    # Benchmark forward
    fwd_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        if fp8_recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = module(x)
        else:
            out = module(x)

        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - start) * 1000)

    # Benchmark backward
    module.train()
    bwd_times = []
    for _ in range(n_iters):
        if fp8_recipe:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = module(x)
        else:
            out = module(x)

        # Handle tuple output for QKV
        if isinstance(out, tuple):
            loss = sum(o.sum() for o in out)
        else:
            loss = out.sum()

        torch.cuda.synchronize()
        start = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - start) * 1000)

    return {
        'name': name,
        'fwd_mean': sum(fwd_times) / len(fwd_times),
        'bwd_mean': sum(bwd_times) / len(bwd_times),
        'total': sum(fwd_times) / len(fwd_times) + sum(bwd_times) / len(bwd_times)
    }


def profile_with_torch_profiler(module, x, name="", use_fp8=True):
    """Use PyTorch profiler to analyze GEMM operations."""
    if use_fp8:
        fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.E4M3)
    else:
        fp8_recipe = None

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function(f"profile_{name}"):
            for _ in range(10):
                if fp8_recipe:
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        out = module(x)
                else:
                    out = module(x)

                if isinstance(out, tuple):
                    loss = sum(o.sum() for o in out)
                else:
                    loss = out.sum()
                loss.backward()

    # Print profiler results
    print(f"\nProfiler results for {name}:")
    print("-" * 60)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Count GEMM operations
    gemm_count = 0
    for event in prof.key_averages():
        if "gemm" in event.key.lower() or "mm" in event.key.lower():
            gemm_count += 1

    print(f"GEMM operations detected: {gemm_count}")

    return gemm_count


def main():
    torch.manual_seed(42)
    device = "cuda"

    print("=" * 80)
    print("GEMM Fusion Analysis: QKV and MLP Optimizations")
    print("=" * 80)

    config = ProfileConfig()
    print(f"\nConfiguration:")
    print(f"  Hidden: {config.hidden_size}, Heads: {config.n_heads}, KV Heads: {config.n_kv_heads}")
    print(f"  FFN: {config.ffn_hidden_size}, Batch: {config.batch_size}, Seq: {config.seq_len}")

    # Prepare input
    x = torch.randn(
        config.seq_len,
        config.batch_size,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True
    )

    # ========================================================================
    # Test QKV Fusion
    # ========================================================================
    print("\n" + "=" * 80)
    print("QKV Projection Comparison")
    print("=" * 80)

    # Create modules
    separate_qkv = SeparateQKV(config).to(device).bfloat16()
    fused_qkv = FusedQKV(config).to(device).bfloat16()

    # Benchmark
    results_qkv = []

    print("\n1. Separate Q, K, V projections (3 GEMMs):")
    result = benchmark_module(separate_qkv, x, "Separate QKV")
    results_qkv.append(result)
    print(f"   Forward: {result['fwd_mean']:.3f}ms, Backward: {result['bwd_mean']:.3f}ms")

    print("\n2. Fused QKV projection (1 GEMM):")
    result = benchmark_module(fused_qkv, x, "Fused QKV")
    results_qkv.append(result)
    print(f"   Forward: {result['fwd_mean']:.3f}ms, Backward: {result['bwd_mean']:.3f}ms")

    speedup = results_qkv[0]['total'] / results_qkv[1]['total']
    print(f"\n   Speedup from fusion: {speedup:.2f}x")

    # ========================================================================
    # Test MLP Fusion
    # ========================================================================
    print("\n" + "=" * 80)
    print("MLP Projection Comparison")
    print("=" * 80)

    # Create modules
    separate_mlp = SeparateMLP(config).to(device).bfloat16()
    fused_swiglu = FusedGateUpMLP(config).to(device).bfloat16()
    fused_geglu = FusedGeGLUMLP(config).to(device).bfloat16()

    results_mlp = []

    print("\n1. Traditional MLP (2 GEMMs):")
    result = benchmark_module(separate_mlp, x, "Separate MLP")
    results_mlp.append(result)
    print(f"   Forward: {result['fwd_mean']:.3f}ms, Backward: {result['bwd_mean']:.3f}ms")

    print("\n2. Fused SwiGLU MLP (gate+up = 1 GEMM):")
    result = benchmark_module(fused_swiglu, x, "Fused SwiGLU")
    results_mlp.append(result)
    print(f"   Forward: {result['fwd_mean']:.3f}ms, Backward: {result['bwd_mean']:.3f}ms")

    print("\n3. Fused GeGLU MLP (gate+up = 1 GEMM):")
    result = benchmark_module(fused_geglu, x, "Fused GeGLU")
    results_mlp.append(result)
    print(f"   Forward: {result['fwd_mean']:.3f}ms, Backward: {result['bwd_mean']:.3f}ms")

    speedup_swiglu = results_mlp[0]['total'] / results_mlp[1]['total']
    speedup_geglu = results_mlp[0]['total'] / results_mlp[2]['total']
    print(f"\n   SwiGLU speedup: {speedup_swiglu:.2f}x")
    print(f"   GeGLU speedup: {speedup_geglu:.2f}x")

    # ========================================================================
    # Detailed Profiling
    # ========================================================================
    print("\n" + "=" * 80)
    print("Detailed GEMM Profiling")
    print("=" * 80)

    print("\nProfiling separate QKV:")
    gemm_separate = profile_with_torch_profiler(separate_qkv, x, "separate_qkv")

    print("\nProfiling fused QKV:")
    gemm_fused = profile_with_torch_profiler(fused_qkv, x, "fused_qkv")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nQKV Fusion Benefits:")
    print(f"  - Reduces 3 GEMMs to 1 GEMM")
    print(f"  - Speedup: {speedup:.2f}x")
    print(f"  - Better memory locality")

    print("\nMLP Fusion Benefits (SwiGLU/GeGLU):")
    print(f"  - Reduces 2 GEMMs to 1 for gate+up")
    print(f"  - SwiGLU speedup: {speedup_swiglu:.2f}x")
    print(f"  - GeGLU speedup: {speedup_geglu:.2f}x")
    print(f"  - Better activation function")

    print("\nRecommendations:")
    print("1. Always use fused QKV projection")
    print("2. Use SwiGLU/GeGLU instead of vanilla MLP")
    print("3. Combine with FP8 for maximum performance")
    print("4. These fusions stack with other optimizations")


if __name__ == "__main__":
    main()