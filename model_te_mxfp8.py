from transformer_engine.common.recipe import Format, DelayedScaling, MXFP8BlockScaling

# HYBRID: E4M3 during forward pass, E5M2 during backward pass
fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

# MXFP8 block scaling with E4M3 used everywhere
mxfp8_format = Format.E4M3
mxfp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)
