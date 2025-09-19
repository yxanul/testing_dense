import torch, transformer_engine as _; import transformer_engine.pytorch as te
print("GPU:", torch.cuda.get_device_name(0))
print("CC:", torch.cuda.get_device_capability(0))
print("CUDA:", torch.version.cuda)
print("TE:", te.__version__)
import flash_attn
print("flash-attn:", flash_attn.__version__)
