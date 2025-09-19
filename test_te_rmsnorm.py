"""
Check if TransformerEngine has RMSNorm support.
"""
import torch
import transformer_engine.pytorch as te

# Check available normalization layers
print("TransformerEngine modules:")
for attr in dir(te):
    if "norm" in attr.lower() or "Norm" in attr:
        print(f"  - {attr}")

# Test if RMSNorm exists
if hasattr(te, 'RMSNorm'):
    print("\n✅ te.RMSNorm is available!")

    # Test it
    x = torch.randn(2, 512, 768, device="cuda", dtype=torch.bfloat16)
    norm = te.RMSNorm(768).cuda().to(torch.bfloat16)
    y = norm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
else:
    print("\n❌ te.RMSNorm not found")

# Check LayerNorm
print("\n✅ te.LayerNorm is available")
norm = te.LayerNorm(768).cuda().to(torch.bfloat16)
print(f"LayerNorm initialized: {norm}")