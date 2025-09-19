import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda"
B, S, H = 2, 128, 768
vocab_size = 50304

print("Debugging NaN issue in TransformerLayer:")
print("-" * 50)

# Create components
print("\n1. Testing individual components:")

# Test embeddings
emb = nn.Embedding(vocab_size, H).cuda().bfloat16()
nn.init.normal_(emb.weight, 0, 0.02)
x_ids = torch.randint(0, vocab_size, (B, S), device=device)
x_emb = emb(x_ids)
print(f"Embeddings: min={x_emb.min():.3f}, max={x_emb.max():.3f}, mean={x_emb.mean():.3f}, std={x_emb.std():.3f}")

# Test single TransformerLayer
x = x_emb.transpose(0, 1).contiguous()  # [S, B, H]
print(f"Input shape: {x.shape}, stats: min={x.min():.3f}, max={x.max():.3f}")

transformer = te.TransformerLayer(
    hidden_size=H,
    ffn_hidden_size=4*H,
    num_attention_heads=12,
    layernorm_epsilon=1e-5,
).cuda().bfloat16()

# Check layer initialization
print("\n2. Layer weight statistics:")
for name, param in transformer.named_parameters():
    if param.dim() > 1:
        print(f"  {name}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")

fp8_recipe = DelayedScaling(margin=1, fp8_format=Format.E4M3)

print("\n3. Forward pass with FP8:")
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = transformer(x)

print(f"Output: min={out.min():.3f}, max={out.max():.3f}, mean={out.mean():.3f}, std={out.std():.3f}")

# Test with multiple layers
print("\n4. Testing with 2 layers:")
x = x_emb.transpose(0, 1).contiguous()

for i in range(2):
    transformer = te.TransformerLayer(
        hidden_size=H,
        ffn_hidden_size=4*H,
        num_attention_heads=12,
    ).cuda().bfloat16()

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        x = transformer(x)

    print(f"  After layer {i}: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")

    if torch.isnan(x).any():
        print(f"  WARNING: NaN detected after layer {i}!")
        break

# Test final projection
print("\n5. Testing final projection:")
if not torch.isnan(x).any():
    lm_head = te.Linear(H, vocab_size, bias=False).cuda().bfloat16()
    nn.init.normal_(lm_head.weight, 0, 0.002)  # Smaller init for large layer

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        logits = lm_head(x)

    print(f"Logits: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")

    # Test loss
    logits_t = logits.transpose(0, 1).contiguous()
    y = torch.randint(0, vocab_size, (B, S), device=device)
    loss = F.cross_entropy(logits_t.view(-1, vocab_size), y.view(-1))
    print(f"Loss: {loss.item():.4f} (NaN: {torch.isnan(loss)})")

print("\n6. Testing WITHOUT FP8 (same setup):")
x = x_emb.transpose(0, 1).contiguous()

for i in range(2):
    transformer = te.TransformerLayer(
        hidden_size=H,
        ffn_hidden_size=4*H,
        num_attention_heads=12,
    ).cuda().bfloat16()

    x = transformer(x)  # No FP8
    print(f"  After layer {i}: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")

lm_head = te.Linear(H, vocab_size, bias=False).cuda().bfloat16()
nn.init.normal_(lm_head.weight, 0, 0.002)
logits = lm_head(x)
logits_t = logits.transpose(0, 1).contiguous()
loss = F.cross_entropy(logits_t.view(-1, vocab_size), y.view(-1))
print(f"Final loss without FP8: {loss.item():.4f}")