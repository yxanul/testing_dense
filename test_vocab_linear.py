import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

# Test the problematic vocab_size dimension
H = 768
vocab_sizes = [50264, 50272, 50304, 32000, 32768, 65536]

print("Testing Linear layers with different vocab sizes:")
print("-" * 50)

fp8_recipe = te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3)

for vocab_size in vocab_sizes:
    try:
        x = torch.randn(128, 2, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        linear = te.Linear(H, vocab_size, bias=False).cuda().bfloat16()

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = linear(x)

        out.sum().backward()
        print(f"✓ vocab_size={vocab_size:6d} (divisible by 32: {vocab_size % 32 == 0})")
    except RuntimeError as e:
        if "cuBLAS" in str(e):
            print(f"✗ vocab_size={vocab_size:6d} - cuBLAS GEMM error")
        else:
            print(f"✗ vocab_size={vocab_size:6d} - {str(e)[:100]}")

print("\nTesting with TransformerLayer + output projection:")
for vocab_size in [50264, 50304]:
    try:
        x = torch.randn(128, 2, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        transformer = te.TransformerLayer(
            hidden_size=H,
            ffn_hidden_size=4*H,
            num_attention_heads=12
        ).cuda().bfloat16()

        lm_head = te.Linear(H, vocab_size, bias=False).cuda().bfloat16()

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            h = transformer(x)
            logits = lm_head(h)

        logits.sum().backward()
        print(f"✓ TransformerLayer + lm_head with vocab_size={vocab_size}")
    except RuntimeError as e:
        if "cuBLAS" in str(e):
            print(f"✗ TransformerLayer + lm_head with vocab_size={vocab_size} - cuBLAS error")
        else:
            print(f"✗ TransformerLayer + lm_head with vocab_size={vocab_size} - Other error")