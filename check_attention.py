"""
Simple check for available attention backends.
"""
import os
import torch
import transformer_engine.pytorch as te

print("Checking Attention Backends")
print("=" * 60)

# 1. Check PyTorch version
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Check TransformerEngine
print(f"\nTransformerEngine version: {te.__version__}")

# 3. Check FlashAttention
print("\nFlashAttention:")
try:
    import flash_attn
    print(f"  ✓ Version {flash_attn.__version__} installed")

    # Check if it actually works
    from flash_attn import flash_attn_func
    print(f"  ✓ flash_attn_func available")
except ImportError as e:
    print(f"  ✗ Not installed: {e}")
    print(f"  Install with: pip install flash-attn --no-build-isolation")

# 4. Check environment variables
print("\nEnvironment Variables:")
env_vars = ["NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_ALLOW_NONDETERMINISTIC_ALGO"]
for var in env_vars:
    val = os.environ.get(var, "not set")
    print(f"  {var}: {val}")

# 5. Test TE's DotProductAttention
print("\nTesting TransformerEngine DotProductAttention:")
try:
    B, S, H, D = 2, 128, 12, 64

    # Create attention module
    attn = te.attention.DotProductAttention(
        num_attention_heads=H,
        kv_channels=D,
        attention_dropout=0.0,
        attn_mask_type="causal"
    ).cuda().bfloat16()

    # Test input
    q = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)

    # Run attention
    out = attn(q, k, v)
    print(f"  ✓ DotProductAttention works: output shape {out.shape}")

except Exception as e:
    print(f"  ✗ Error: {e}")

# 6. Test with different backends
print("\nTesting with different backend settings:")

configs = [
    ("Default", {}),
    ("Force Flash", {"NVTE_FLASH_ATTN": "1"}),
    ("Force cuDNN", {"NVTE_FUSED_ATTN": "1"}),
]

for name, env in configs:
    # Save old env
    old_env = {k: os.environ.get(k) for k in env}

    # Set new env
    os.environ.update(env)

    try:
        attn = te.attention.DotProductAttention(
            num_attention_heads=H,
            kv_channels=D,
            attention_dropout=0.0,
            attn_mask_type="causal"
        ).cuda().bfloat16()

        out = attn(q, k, v)
        print(f"  ✓ {name}: Works")

    except Exception as e:
        print(f"  ✗ {name}: {str(e)[:50]}")

    # Restore env
    for k, v in old_env.items():
        if v is None and k in os.environ:
            del os.environ[k]
        elif v is not None:
            os.environ[k] = v

# 7. PyTorch 2.0 scaled_dot_product_attention
print("\nPyTorch 2.0 scaled_dot_product_attention:")
try:
    import torch.nn.functional as F

    # Test if SDPA is available
    q_test = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    k_test = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    v_test = torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16)

    out = F.scaled_dot_product_attention(q_test, k_test, v_test, is_causal=True)
    print(f"  ✓ Available and working")

    # Check which backend it's using
    import torch._C as C
    if hasattr(C, '_fused_sdp_choice'):
        backend = C._fused_sdp_choice(q_test, k_test, v_test)
        backends = ['math', 'flash', 'mem_efficient']
        if backend < len(backends):
            print(f"  Using backend: {backends[backend]}")

except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("- Use NVTE_FLASH_ATTN=1 to enable FlashAttention in TE")
print("- Use NVTE_FUSED_ATTN=1 to enable cuDNN fused attention")
print("- PyTorch 2.0+ SDPA auto-selects the best backend")