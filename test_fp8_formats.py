"""
Test different FP8 formats to confirm E4M3/E5M2 usage in backward pass.
"""
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Simple test to verify FP8 formats
print("FP8 Format Test")
print("=" * 50)

# Test configurations
configs = [
    ("E4M3 only", Format.E4M3),
    ("HYBRID (E4M3 fwd, E5M2 bwd)", Format.HYBRID),
]

for name, fmt in configs:
    print(f"\n{name}:")
    print("-" * 30)

    recipe = DelayedScaling(margin=0, fp8_format=fmt)

    # Create simple layer
    linear = te.Linear(768, 768, bias=False).cuda().bfloat16()
    x = torch.randn(128, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Forward with FP8
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        y = linear(x)

    # Check FP8 metadata
    print(f"Recipe format: {recipe.fp8_format}")
    print(f"Forward format: {recipe.fp8_format.value.fwd}")
    print(f"Backward format: {recipe.fp8_format.value.bwd}")
    print(f"E4M3 max: {recipe.fp8_format.value.fwd.value.max_fwd}")
    print(f"E5M2 max (for HYBRID bwd): {recipe.fp8_format.value.bwd.value.max_bwd if fmt == Format.HYBRID else 'N/A'}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"✓ Forward and backward completed")

print("\n" + "=" * 50)
print("Format Details:")
print("- E4M3: 1 sign, 4 exponent, 3 mantissa bits (range: ±448)")
print("- E5M2: 1 sign, 5 exponent, 2 mantissa bits (range: ±57344)")
print("- HYBRID: Uses E4M3 for activations (forward), E5M2 for gradients (backward)")
print("  This provides better gradient precision at the cost of range")