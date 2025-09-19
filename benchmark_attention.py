"""
Comprehensive benchmark comparing all attention backends:
- Standard PyTorch attention
- FlashAttention 2
- cuDNN Fused Attention
- TransformerEngine's optimized attention
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
import time
import gc
from contextlib import contextmanager


@contextmanager
def set_env(**env_vars):
    """Context manager to temporarily set environment variables."""
    old_env = {k: os.environ.get(k) for k in env_vars}
    os.environ.update({k: str(v) for k, v in env_vars.items()})
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None and k in os.environ:
                del os.environ[k]
            elif v is not None:
                os.environ[k] = v


def benchmark_attention(attn_func, q, k, v, n_iters=100, warmup=10, name=""):
    """Benchmark an attention function."""
    # Warmup
    for _ in range(warmup):
        out = attn_func(q, k, v)
        if out.requires_grad:
            out.sum().backward(retain_graph=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Forward timing
    fwd_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        out = attn_func(q, k, v)

        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - start) * 1000)

    # Backward timing
    bwd_times = []
    for _ in range(n_iters):
        out = attn_func(q, k, v)

        torch.cuda.synchronize()
        start = time.perf_counter()

        out.sum().backward(retain_graph=True)

        torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - start) * 1000)

    # Memory usage
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        'name': name,
        'forward': sum(fwd_times) / len(fwd_times),
        'backward': sum(bwd_times) / len(bwd_times),
        'total': sum(fwd_times) / len(fwd_times) + sum(bwd_times) / len(bwd_times),
        'memory': memory_mb
    }


class StandardAttention(nn.Module):
    """Standard PyTorch scaled dot-product attention."""

    def __init__(self, n_heads, head_dim, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.dropout = dropout

    def forward(self, q, k, v):
        # q, k, v: [seq, batch, heads, dim]
        S = q.shape[0]

        # Reshape for batch matrix multiplication
        q = q.permute(1, 2, 0, 3)  # [B, H, S, D]
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        # Apply attention
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.permute(2, 0, 1, 3)  # [S, B, H, D]
        return out


class SDPAttention(nn.Module):
    """PyTorch 2.0+ scaled_dot_product_attention."""

    def __init__(self, n_heads, head_dim, dropout=0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v):
        # q, k, v: [seq, batch, heads, dim]
        q = q.permute(1, 2, 0, 3)  # [B, H, S, D]
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Use PyTorch's optimized SDP attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.permute(2, 0, 1, 3)  # [S, B, H, D]
        return out


class TEAttention(nn.Module):
    """TransformerEngine DotProductAttention."""

    def __init__(self, n_heads, head_dim, dropout=0.0):
        super().__init__()
        self.attn = te.attention.DotProductAttention(
            num_attention_heads=n_heads,
            kv_channels=head_dim,
            attention_dropout=dropout,
            attn_mask_type="causal"
        ).cuda().bfloat16()

    def forward(self, q, k, v):
        return self.attn(q, k, v)


class FlashAttention(nn.Module):
    """FlashAttention 2 (if available)."""

    def __init__(self, n_heads, head_dim, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.available = False

        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.available = True
        except ImportError:
            pass

    def forward(self, q, k, v):
        if not self.available:
            # Fallback to standard attention
            return StandardAttention(q.shape[2], q.shape[3], self.dropout)(q, k, v)

        # flash_attn expects [batch, seq, heads, dim]
        q = q.permute(1, 0, 2, 3)  # [B, S, H, D]
        k = k.permute(1, 0, 2, 3)
        v = v.permute(1, 0, 2, 3)

        out = self.flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True
        )

        out = out.permute(1, 0, 2, 3)  # [S, B, H, D]
        return out


def main():
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    print("=" * 80)
    print("Attention Backend Benchmark")
    print("=" * 80)

    # Test different configurations
    test_configs = [
        (2, 512, 16, 64),   # Small: B=2, S=512, H=16, D=64
        (4, 1024, 16, 64),  # Medium: B=4, S=1024, H=16, D=64
        (8, 2048, 16, 64),  # Large: B=8, S=2048, H=16, D=64
    ]

    for B, S, H, D in test_configs:
        print(f"\nConfiguration: B={B}, S={S}, H={H}, D={D}")
        print(f"Total params: {B * S * H * D / 1e6:.1f}M")
        print("-" * 60)

        # Prepare inputs
        q = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(S, B, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)

        results = []

        # Test 1: Standard PyTorch attention
        print("\n1. Standard PyTorch Attention")
        try:
            attn = StandardAttention(H, D).cuda().bfloat16()
            result = benchmark_attention(attn, q, k, v, name="Standard")
            results.append(result)
            print(f"   Forward: {result['forward']:.2f}ms, Backward: {result['backward']:.2f}ms")
        except Exception as e:
            print(f"   Failed: {e}")

        # Test 2: PyTorch 2.0 scaled_dot_product_attention
        print("\n2. PyTorch 2.0 SDP Attention")
        try:
            attn = SDPAttention(H, D).cuda().bfloat16()
            result = benchmark_attention(attn, q, k, v, name="PyTorch SDP")
            results.append(result)
            print(f"   Forward: {result['forward']:.2f}ms, Backward: {result['backward']:.2f}ms")
        except Exception as e:
            print(f"   Failed: {e}")

        # Test 3: TransformerEngine (default backend)
        print("\n3. TransformerEngine (default)")
        try:
            with set_env(NVTE_FLASH_ATTN="0", NVTE_FUSED_ATTN="0"):
                attn = TEAttention(H, D)
                result = benchmark_attention(attn, q, k, v, name="TE Default")
                results.append(result)
                print(f"   Forward: {result['forward']:.2f}ms, Backward: {result['backward']:.2f}ms")
        except Exception as e:
            print(f"   Failed: {e}")

        # Test 4: TransformerEngine with FlashAttention
        print("\n4. TransformerEngine (FlashAttention)")
        try:
            with set_env(NVTE_FLASH_ATTN="1", NVTE_FUSED_ATTN="0"):
                attn = TEAttention(H, D)
                result = benchmark_attention(attn, q, k, v, name="TE Flash")
                results.append(result)
                print(f"   Forward: {result['forward']:.2f}ms, Backward: {result['backward']:.2f}ms")
        except Exception as e:
            print(f"   Failed: {e}")

        # Test 5: TransformerEngine with cuDNN Fused
        print("\n5. TransformerEngine (cuDNN Fused)")
        try:
            with set_env(NVTE_FLASH_ATTN="0", NVTE_FUSED_ATTN="1"):
                attn = TEAttention(H, D)
                result = benchmark_attention(attn, q, k, v, name="TE cuDNN")
                results.append(result)
                print(f"   Forward: {result['forward']:.2f}ms, Backward: {result['backward']:.2f}ms")
        except Exception as e:
            print(f"   Failed: {e}")

        # Test 6: Native FlashAttention (if available)
        print("\n6. Native FlashAttention 2")
        try:
            attn = FlashAttention(H, D).cuda().bfloat16()
            if attn.available:
                result = benchmark_attention(attn, q, k, v, name="Flash Native")
                results.append(result)
                print(f"   Forward: {result['forward']:.2f}ms, Backward: {result['backward']:.2f}ms")
            else:
                print("   Not available (install with: pip install flash-attn)")
        except Exception as e:
            print(f"   Failed: {e}")

        # Summary for this configuration
        if results:
            print("\n" + "=" * 60)
            print(f"Summary for B={B}, S={S}:")
            print("-" * 60)

            baseline = results[0]['total']
            print(f"{'Backend':<20} {'Forward':>10} {'Backward':>10} {'Total':>10} {'Speedup':>8} {'Memory':>10}")
            print("-" * 60)

            for r in sorted(results, key=lambda x: x['total']):
                speedup = baseline / r['total']
                print(f"{r['name']:<20} {r['forward']:>8.2f}ms {r['backward']:>8.2f}ms "
                      f"{r['total']:>8.2f}ms {speedup:>6.2f}x {r['memory']:>8.1f}MB")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("Recommendations:")
    print("1. FlashAttention 2 is typically fastest for long sequences")
    print("2. cuDNN Fused Attention works well for shorter sequences")
    print("3. PyTorch 2.0 SDP auto-selects backend (Flash if available)")
    print("4. Set NVTE_FLASH_ATTN=1 to force FlashAttention in TE")


if __name__ == "__main__":
    main()