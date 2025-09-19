import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

torch.cuda.set_device(0)

M, K, N = 64, 4096, 4096   # multiples of 32
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

lin = te.Linear(K, N, bias=False)  # weights in BF16, executed in FP8 via autocast

# Try classic FP8 (E4M3) first; on some setups it avoids unlucky cuBLASLt heuristics.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = lin(x)
    y.sum().backward()

print("OK")
