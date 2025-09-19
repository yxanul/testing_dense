import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

M,K,N = 64, 4096, 4096   # multiples of 32 (MXFP8 rule)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lin = te.Linear(K, N, bias=False).to("cuda", dtype=torch.bfloat16)

fp8_recipe = te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = lin(x)
y.sum().backward()
print("FP8 (E4M3) Linear backward OK")
