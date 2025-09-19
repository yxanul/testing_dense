"""
Test if FlashAttention is available and working with TransformerEngine.
"""
import torch
import transformer_engine.pytorch as te

print("Checking FlashAttention support in TransformerEngine:")
print("=" * 60)

# Check TE's attention backends
try:
    from transformer_engine.pytorch.attention import (
        DotProductAttention,
        MultiheadAttention,
        get_attention_backend
    )

    print("✓ TransformerEngine attention modules available")

    # Check available backends
    print("\nChecking attention backends...")

    # Test DotProductAttention with different backends
    backends = ["FlashAttention", "FusedAttention", "UnfusedDotProductAttention"]

    for backend in backends:
        try:
            # Try to get backend info
            print(f"\n{backend}:")

            if backend == "FlashAttention":
                try:
                    import flash_attn
                    print(f"  ✓ flash_attn version: {flash_attn.__version__}")
                except ImportError:
                    print(f"  ✗ flash_attn not installed")

            elif backend == "FusedAttention":
                # Check if NVTE_FUSED_ATTN is enabled
                import os
                if os.environ.get("NVTE_FUSED_ATTN") == "1":
                    print(f"  ✓ NVTE_FUSED_ATTN=1 (cuDNN backend enabled)")
                else:
                    print(f"  - NVTE_FUSED_ATTN not set (cuDNN backend disabled)")

        except Exception as e:
            print(f"  Error: {e}")

except ImportError as e:
    print(f"✗ Error importing TE attention: {e}")

print("\n" + "=" * 60)
print("Environment variables for attention backends:")
print("- NVTE_FLASH_ATTN=0/1  : Enable FlashAttention")
print("- NVTE_FUSED_ATTN=0/1  : Enable cuDNN fused attention")
print("- NVTE_ALLOW_NONDETERMINISTIC_ALGO=0/1 : Allow non-deterministic algorithms")

# Test TE's DotProductAttention
print("\n" + "=" * 60)
print("Testing DotProductAttention:")

try:
    B, S, H, D = 2, 512, 12, 64  # batch, seq, heads, dim

    # Create attention module
    attn = te.attention.DotProductAttention(
        num_attention_heads=H,
        kv_channels=D,
        attention_dropout=0.0
    ).cuda().bfloat16()

    # Test input
    qkv = torch.randn(S, B, 3, H, D, device="cuda", dtype=torch.bfloat16)

    # Run attention
    import os

    # Test with different backends
    configs = [
        ("Default", {}),
        ("FlashAttention", {"NVTE_FLASH_ATTN": "1"}),
        ("Fused (cuDNN)", {"NVTE_FUSED_ATTN": "1"}),
    ]

    for name, env_vars in configs:
        # Set environment variables
        old_env = {k: os.environ.get(k) for k in env_vars}
        os.environ.update(env_vars)

        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = attn(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
            print(f"✓ {name}: Output shape {out.shape}")
        except Exception as e:
            print(f"✗ {name}: {str(e)[:100]}")

        # Restore environment
        for k, v in old_env.items():
            if v is None and k in os.environ:
                del os.environ[k]
            elif v is not None:
                os.environ[k] = v

except Exception as e:
    print(f"Error in DotProductAttention test: {e}")

print("\n" + "=" * 60)
print("To enable FlashAttention in your model:")
print("1. Install: pip install flash-attn --no-build-isolation")
print("2. Set: export NVTE_FLASH_ATTN=1")
print("3. Or use TE's MultiheadAttention which auto-selects best backend")