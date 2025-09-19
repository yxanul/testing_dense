import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

# Test different tensor layouts
B, S, H = 2, 128, 768
fp8_recipe = te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3)

print("Testing tensor layouts with FP8:")
print("-" * 50)

# Test 1: [S, B, H] layout (TransformerEngine default)
print("\n1. [Seq, Batch, Hidden] layout:")
try:
    x_sbh = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear = te.Linear(H, H, bias=False).cuda().bfloat16()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Linear applies to last dimension
        out = linear(x_sbh)

    out.sum().backward()
    print(f"✓ Shape {x_sbh.shape} -> {out.shape} works")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: [B, S, H] layout (typical PyTorch)
print("\n2. [Batch, Seq, Hidden] layout:")
try:
    x_bsh = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear = te.Linear(H, H, bias=False).cuda().bfloat16()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = linear(x_bsh)

    out.sum().backward()
    print(f"✓ Shape {x_bsh.shape} -> {out.shape} works")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Flattened [S*B, H] layout
print("\n3. [SeqBatch, Hidden] flattened layout:")
try:
    x_flat = torch.randn(S*B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear = te.Linear(H, H, bias=False).cuda().bfloat16()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = linear(x_flat)

    out.sum().backward()
    print(f"✓ Shape {x_flat.shape} -> {out.shape} works")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: TransformerLayer with different sequence lengths
print("\n4. TransformerLayer with various sequence lengths:")
for seq_len in [32, 64, 128, 256, 512]:
    try:
        x = torch.randn(seq_len, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        transformer = te.TransformerLayer(
            hidden_size=H,
            ffn_hidden_size=4*H,
            num_attention_heads=12
        ).cuda().bfloat16()

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = transformer(x)

        out.sum().backward()
        print(f"✓ Seq length {seq_len} works")
    except Exception as e:
        if "cuBLAS" in str(e):
            print(f"✗ Seq length {seq_len} - cuBLAS error")
        else:
            print(f"✗ Seq length {seq_len} - Other error")
        break