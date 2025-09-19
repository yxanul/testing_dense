import os
os.environ.setdefault("NVTE_FLASH_ATTN","0")
os.environ.setdefault("NVTE_FUSED_ATTN","1")

import torch, transformer_engine as te
import transformer_engine.pytorch as te_pt
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "TE:", te.__version__)

# FP8 Linear smoke test – pick multiples of 32 for MXFP8
M,K,N = 64, 4096, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lin = te_pt.Linear(K, N, bias=False).to("cuda", dtype=torch.bfloat16)

with te_pt.fp8_autocast(enabled=True):   # FP8 activations, BF16 weights
    y = lin(x)
    y.sum().backward()
print("FP8 TE Linear backward OK")
