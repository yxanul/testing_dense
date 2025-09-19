"""
Test if using te.RMSNorm/te.LayerNorm properly enables FP8 flow.
Also test LayerNormLinear fusion for even better FP8 performance.
"""
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
import time


class ModelWithCustomNorm(nn.Module):
    """Model using custom RMSNorm (breaks FP8 flow)."""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.norm = CustomRMSNorm(hidden_size)
        self.linear = te.Linear(hidden_size, hidden_size * 4)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)


class CustomRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class ModelWithTENorm(nn.Module):
    """Model using te.RMSNorm (preserves FP8 flow)."""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.norm = te.RMSNorm(hidden_size)
        self.linear = te.Linear(hidden_size, hidden_size * 4)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)


class ModelWithFusedNormLinear(nn.Module):
    """Model using LayerNormLinear (optimal FP8 flow)."""
    def __init__(self, hidden_size=768):
        super().__init__()
        # Fused norm + linear in one operation
        self.norm_linear = te.LayerNormLinear(
            hidden_size,
            hidden_size * 4,
            normalization="RMSNorm"
        )

    def forward(self, x):
        return self.norm_linear(x)


def benchmark_model(model, name, batch_size=8, seq_len=512, hidden_size=768):
    """Benchmark a model with FP8."""
    device = "cuda"
    model = model.to(device).to(torch.bfloat16)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    fp8_recipe = DelayedScaling(
        margin=0,
        fp8_format=Format.HYBRID,
        amax_history_len=32,
        amax_compute_algo="max"
    )

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = model(x)
            loss = y.sum()
        loss.backward()
        optimizer.step()

    # Benchmark
    torch.cuda.synchronize()
    times = []

    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad()
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = model(x)
            loss = y.sum()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    throughput = (batch_size * seq_len) / avg_time

    print(f"\n{name}:")
    print(f"  Avg time: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:,.0f} tokens/sec")

    return throughput


def main():
    print("=" * 70)
    print("FP8 FLOW TEST: Impact of Normalization Layers")
    print("=" * 70)

    # Test 1: Custom norm (breaks FP8 flow)
    model1 = ModelWithCustomNorm()
    throughput1 = benchmark_model(model1, "Custom RMSNorm (breaks FP8 flow)")

    # Test 2: te.RMSNorm (preserves FP8 flow)
    model2 = ModelWithTENorm()
    throughput2 = benchmark_model(model2, "te.RMSNorm (preserves FP8 flow)")

    # Test 3: Fused LayerNormLinear (optimal FP8 flow)
    model3 = ModelWithFusedNormLinear()
    throughput3 = benchmark_model(model3, "LayerNormLinear (fused, optimal FP8)")

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("-" * 70)
    print(f"Custom RMSNorm:     {throughput1:>10,.0f} tokens/sec (1.00x)")
    print(f"te.RMSNorm:         {throughput2:>10,.0f} tokens/sec ({throughput2/throughput1:.2f}x)")
    print(f"LayerNormLinear:    {throughput3:>10,.0f} tokens/sec ({throughput3/throughput1:.2f}x)")
    print("=" * 70)

    if throughput2 > throughput1 * 1.05:
        print("\n✅ Using te.RMSNorm improves FP8 performance!")
    else:
        print("\n⚠️ te.RMSNorm shows no improvement - FP8 may not be supported on this GPU")

    if throughput3 > throughput2 * 1.05:
        print("✅ LayerNormLinear fusion provides additional speedup!")


if __name__ == "__main__":
    main()