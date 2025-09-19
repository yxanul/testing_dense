import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# Test numerical stability of TransformerLayer with FP8
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda"

B, S, H = 2, 128, 768
vocab_size = 50304

print("Testing numerical stability with FP8:")
print("-" * 50)

# Test different configurations
configs = [
    ("E4M3 margin=0", DelayedScaling(margin=0, fp8_format=Format.E4M3)),
    ("E4M3 margin=1", DelayedScaling(margin=1, fp8_format=Format.E4M3)),
    ("E4M3 amax=16", DelayedScaling(margin=0, fp8_format=Format.E4M3, amax_history_len=16)),
    ("E5M2 margin=0", DelayedScaling(margin=0, fp8_format=Format.E5M2)),
    ("HYBRID default", DelayedScaling(fp8_format=Format.HYBRID)),
]

for name, recipe in configs:
    print(f"\n{name}:")

    # Create model components
    transformer = te.TransformerLayer(
        hidden_size=H,
        ffn_hidden_size=4*H,
        num_attention_heads=12,
        layernorm_epsilon=1e-5
    ).cuda().bfloat16()

    lm_head = te.Linear(H, vocab_size, bias=False).cuda().bfloat16()

    # Initialize weights with smaller values for stability
    for param in transformer.parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
    torch.nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)

    # Test data
    x = torch.randn(S, B, H, device=device, dtype=torch.bfloat16) * 0.1  # Scale down input
    y = torch.randint(0, vocab_size, (B, S), device=device)

    try:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            h = transformer(x)
            logits = lm_head(h)

        logits_t = logits.transpose(0, 1).contiguous()
        loss = F.cross_entropy(logits_t.view(-1, vocab_size), y.view(-1))

        if torch.isnan(loss):
            print(f"  ✗ Loss is NaN")
            # Check intermediate values
            print(f"    h stats: min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f}")
            print(f"    logits stats: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
        else:
            loss.backward()
            print(f"  ✓ Loss: {loss.item():.4f}")

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")

print("\n" + "="*50)
print("Testing without FP8:")
transformer = te.TransformerLayer(
    hidden_size=H,
    ffn_hidden_size=4*H,
    num_attention_heads=12
).cuda().bfloat16()
lm_head = te.Linear(H, vocab_size, bias=False).cuda().bfloat16()

x = torch.randn(S, B, H, device=device, dtype=torch.bfloat16) * 0.1
y = torch.randint(0, vocab_size, (B, S), device=device)

h = transformer(x)
logits = lm_head(h)
logits_t = logits.transpose(0, 1).contiguous()
loss = F.cross_entropy(logits_t.view(-1, vocab_size), y.view(-1))
print(f"Without FP8 - Loss: {loss.item():.4f}")