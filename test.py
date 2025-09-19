# test.py
import os
# (Optional) pick attention backend BEFORE importing/creating TE modules:
os.environ.setdefault("NVTE_FLASH_ATTN", "0")  # avoid FA if you saw version warnings
os.environ.setdefault("NVTE_FUSED_ATTN", "1")  # use cuDNN fused attention

import torch
import transformer_engine as te                 # <-- has __version__
import transformer_engine.pytorch as te_pt

print("GPU:", torch.cuda.get_device_name(0))
print("CC:", torch.cuda.get_device_capability(0))
print("CUDA:", torch.version.cuda)
print("TE:", te.__version__)

# Show flash-attn version if installed (just informational)
try:
    import flash_attn
    print("flash-attn:", flash_attn.__version__)
except Exception as e:
    print("flash-attn: not installed or failed to import ->", e)

# FP8 smoke test for TE Linear (should pass on Blackwell/Hopper/Ada)
torch.backends.cuda.matmul.allow_tf32 = True
x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lin = te_pt.Linear(4096, 4096, bias=False).to("cuda", dtype=torch.bfloat16)

with te_pt.fp8_autocast(enabled=True):        # FP8 activations, BF16 weights
    y = lin(x)
    loss = y.pow(2).mean()
loss.backward()
print("FP8 TE Linear backward OK")
