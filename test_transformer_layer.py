import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

# Test TransformerLayer vs individual components
B, S, H = 2, 128, 768

print("Testing TransformerLayer components with FP8:")
print("-" * 50)

# Test 1: Standalone LayerNorm + Linear (similar to what's inside TransformerLayer)
print("\n1. Testing LayerNorm + Linear (manual attention-like):")
try:
    x = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ln = te.LayerNorm(H).cuda().bfloat16()
    qkv_linear = te.Linear(H, 3*H, bias=True).cuda().bfloat16()
    proj_linear = te.Linear(H, H, bias=True).cuda().bfloat16()

    fp8_recipe = te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        x_ln = ln(x)
        qkv = qkv_linear(x_ln)
        # Simple mock attention output (not actual attention)
        attn_out = proj_linear(qkv[:, :, :H])

    attn_out.sum().backward()
    print("✓ LayerNorm + Linear components work")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: DotProductAttention module
print("\n2. Testing DotProductAttention:")
try:
    x = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    attn = te.DotProductAttention(
        num_attention_heads=12,
        kv_channels=64,
        attention_dropout=0.0
    ).cuda().bfloat16()

    # Prepare Q, K, V tensors
    qkv = x.unsqueeze(2).repeat(1, 1, 3, 1)  # [S, B, 3, H]

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = attn(qkv, qkv, qkv)

    out.sum().backward()
    print("✓ DotProductAttention works")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: MultiheadAttention module
print("\n3. Testing MultiheadAttention:")
try:
    x = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mha = te.MultiheadAttention(
        hidden_size=H,
        num_attention_heads=12
    ).cuda().bfloat16()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = mha(x)

    out.sum().backward()
    print("✓ MultiheadAttention works")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: Full TransformerLayer
print("\n4. Testing TransformerLayer:")
try:
    x = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    transformer = te.TransformerLayer(
        hidden_size=H,
        ffn_hidden_size=4*H,
        num_attention_heads=12
    ).cuda().bfloat16()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = transformer(x)

    out.sum().backward()
    print("✓ TransformerLayer works")
except Exception as e:
    if "cuBLAS" in str(e):
        print(f"✗ Failed with cuBLAS error")
    else:
        print(f"✗ Failed: {str(e)[:200]}")

# Test 5: TransformerLayer without FP8
print("\n5. Testing TransformerLayer WITHOUT FP8:")
try:
    x = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    transformer = te.TransformerLayer(
        hidden_size=H,
        ffn_hidden_size=4*H,
        num_attention_heads=12
    ).cuda().bfloat16()

    out = transformer(x)  # No FP8 autocast
    out.sum().backward()
    print("✓ TransformerLayer works without FP8")
except Exception as e:
    print(f"✗ Failed: {e}")