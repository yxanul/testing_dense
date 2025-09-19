import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

# Test different shapes to find cuBLAS FP8 limits
def test_linear_shape(M, K, N, name):
    try:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        lin = te.Linear(K, N, bias=False).to("cuda", dtype=torch.bfloat16)

        fp8_recipe = te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3)

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = lin(x)
        y.sum().backward()
        print(f"✓ {name}: {M}×{K} @ {K}×{N} = {M}×{N}")
        return True
    except RuntimeError as e:
        if "cuBLAS" in str(e):
            print(f"✗ {name}: {M}×{K} @ {K}×{N} = {M}×{N} - cuBLAS error")
        else:
            print(f"✗ {name}: {M}×{K} @ {K}×{N} = {M}×{N} - {e}")
        return False

# Test cases
print("Testing FP8 Linear with different shapes:")
print("-" * 50)

# Original test2.py shape (works)
test_linear_shape(64, 4096, 4096, "Original (test2.py)")

# GPT-2 shapes
test_linear_shape(256, 768, 768, "GPT-2 attention QKV")
test_linear_shape(256, 768, 3072, "GPT-2 FFN up")
test_linear_shape(256, 3072, 768, "GPT-2 FFN down")

# Per-head shapes (12 heads, 64 dims each)
test_linear_shape(256, 64, 64, "Per-head attention")
test_linear_shape(128, 64, 64, "Per-head small batch")
test_linear_shape(32, 64, 64, "Per-head tiny batch")

# Small shapes
test_linear_shape(16, 32, 32, "Very small")
test_linear_shape(32, 32, 32, "Minimum 32")
test_linear_shape(64, 64, 64, "Small cube")

print("\nConclusion: FP8 GEMMs likely have minimum size requirements.")
print("TransformerLayer's head-splitting creates smaller GEMMs that fail.")