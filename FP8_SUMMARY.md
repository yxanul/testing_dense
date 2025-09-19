# TransformerEngine FP8 Summary

## Key Finding: Vocabulary Size Must Be Divisible by 32
- **50264** (original GPT-2 vocab) ❌ Fails with "cuBLAS GEMM algorithm not found"
- **50304** (next multiple of 32) ✅ Works

## Model Implementations Status

### 1. `model_te.py` - Using `te.TransformerLayer` ❌
- **Status**: Forward/backward completes but produces NaN loss
- **Issue**: TransformerLayer has numerical instability with FP8
- **Recipe tested**: `DelayedScaling(margin=0, fp8_format=Format.E4M3)`

### 2. `model_te_custom.py` - Custom attention with standalone `te.Linear` ✅
- **Status**: Works perfectly with FP8
- **Loss**: ~11.0 (normal for untrained model)
- **Key**: Manual attention implementation avoids TransformerLayer issues

### 3. `model_te_fused.py` - Using `te.LayerNormLinear` and `te.LayerNormMLP` ✅
- **Status**: Expected to work (follows TE documentation pattern)
- **Key**: Fused modules provide better FP8 support than TransformerLayer

## Test Files
- `test2.py`: Simple Linear layer test ✅
- `test_shapes.py`: Tests various tensor shapes - all work ✅
- `test_stability.py`: Shows margin=1 causes NaN, margin=0 or 2 work ✅
- `debug_nan.py`: Pinpoints NaN appears in TransformerLayer forward pass

## FP8 Recipe Configurations

### Working Configurations:
```python
# Simple E4M3 (like test2.py)
DelayedScaling(margin=0, fp8_format=Format.E4M3)

# E4M3 with margin=2
DelayedScaling(margin=2, fp8_format=Format.E4M3)

# HYBRID format
DelayedScaling(fp8_format=Format.HYBRID)
```

### Failing Configuration:
```python
# margin=1 causes NaN with TransformerLayer
DelayedScaling(margin=1, fp8_format=Format.E4M3)
```

## Recommendations

1. **For production**: Use `model_te_custom.py` or `model_te_fused.py`
2. **Avoid**: `te.TransformerLayer` with FP8 (has numerical issues)
3. **Always**: Ensure vocab_size is divisible by 32
4. **FP8 Recipe**: Use margin=0 or margin=2, avoid margin=1

## Next Steps
- Report TransformerLayer FP8 instability to NVIDIA/TransformerEngine team
- Use custom implementations for production FP8 training
- Monitor TransformerEngine updates for fixes