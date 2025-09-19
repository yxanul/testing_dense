import os
os.environ.setdefault("NVTE_FLASH_ATTN", "0")  # optional: avoid FA for this smoke test
os.environ.setdefault("NVTE_FUSED_ATTN", "1")

import torch
import transformer_engine as te
import transformer_engine.pytorch as te_pt

print("GPU:", torch.cuda.get_device_name(0))
print("CC:", torch.cuda.get_device_capability(0))
print("CUDA:", torch.version.cuda)
print("TE:", te.__version__)

try:
    import flash_attn
    print("flash-attn:", flash_attn.__version__)
except Exception:
    print("flash-attn: not installed")

torch.backends.cuda.matmul.allow_tf32 = True

# Make BOTH dims multiples of 32
M, K = 32, 4096   # was 8, 4096 -> fails due to M%32 != 0
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lin = te_pt.Linear(K, K, bias=False).to("cuda", dtype=torch.bfloat16)

with te_pt.fp8_autocast(enabled=True):  # FP8 activations (MXFP8), BF16 weights
    y = lin(x)
    loss = y.pow(2).mean()

loss.backward()
print("FP8 TE Linear backward OK")
