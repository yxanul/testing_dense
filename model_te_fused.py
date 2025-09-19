"""
GPT-2 using TransformerEngine fused modules (LayerNormLinear, LayerNormMLP).
Based on TE documentation's recommended approach for optimal FP8 performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

class FusedTransformerBlock(nn.Module):
    """Transformer block using TE fused modules for better FP8 support."""

    def __init__(self, hidden_size, num_heads, dropout=0.1, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused LayerNorm + QKV projection
        self.ln_qkv = te.LayerNormLinear(
            hidden_size, 3 * hidden_size,
            eps=eps, bias=True
        )

        # Attention output projection
        self.proj = te.Linear(hidden_size, hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        # Fused LayerNorm + MLP
        self.ln_mlp = te.LayerNormMLP(
            hidden_size, 4 * hidden_size,
            eps=eps, bias=True
        )

    def forward(self, x):
        # x: [seq, batch, hidden]
        S, B, _ = x.shape
        residual = x

        # Fused LayerNorm + QKV projection
        qkv = self.ln_qkv(x)  # [S, B, 3*H]

        # Reshape for multi-head attention
        qkv = qkv.view(S, B, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # [B, heads, S, 3*head_dim]
        q, k, v = torch.split(qkv, self.head_dim, dim=3)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_out = torch.matmul(attn_weights, v)  # [B, heads, S, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.hidden_size)
        attn_out = attn_out.transpose(0, 1)  # [S, B, H]

        # Projection + residual
        attn_out = self.proj(attn_out)
        x = residual + self.dropout(attn_out)

        # Fused LayerNorm + MLP + residual
        residual = x
        x = self.ln_mlp(x)
        x = residual + x

        return x


class GPT2FusedModel(nn.Module):
    def __init__(self, vocab_size=50304, n_positions=1024, n_embd=768,
                 n_layer=12, n_head=12, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd

        # Token & position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks with fused modules
        self.blocks = nn.ModuleList([
            FusedTransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = te.LayerNorm(n_embd)
        self.lm_head = te.Linear(n_embd, vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)
        self.to(dtype=torch.bfloat16)

        # FP8 recipe - use same as working test2.py
        self.fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.E4M3)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (te.Linear, te.LayerNormLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        # Special init for large output layer
        if isinstance(module, te.Linear) and hasattr(module, 'out_features'):
            if module.out_features > 10000:
                nn.init.normal_(module.weight, mean=0.0, std=0.002)

    def forward(self, input_ids, use_fp8=True):
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos_ids = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.drop(x)
        x = x.transpose(0, 1)  # [S, B, H]

        # Apply transformer blocks with FP8
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=self.fp8_recipe):
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)  # [S, B, V]

        return logits.transpose(0, 1)  # [B, S, V]


if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    print("Testing GPT-2 with TE fused modules (LayerNormLinear, LayerNormMLP)...")

    # Create model with fewer layers for testing
    model = GPT2FusedModel(n_layer=2).to(device)

    # Test forward/backward
    B, S = 2, 128
    vocab_size = 50304
    x = torch.randint(0, vocab_size, (B, S), device=device)
    y = torch.randint(0, vocab_size, (B, S), device=device)

    # Forward with FP8
    print("Running forward pass with FP8...")
    logits = model(x, use_fp8=True)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    print(f"Forward pass complete. Loss: {loss.item():.4f}")

    # Backward
    print("Running backward pass...")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"✓ Fused model with FP8 works! Final loss: {loss.item():.4f}")

    # Test without FP8 for comparison
    print("\nTesting without FP8...")
    logits = model(x, use_fp8=False)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    print(f"Without FP8 - Loss: {loss.item():.4f}")