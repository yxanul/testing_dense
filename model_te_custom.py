"""
GPT-2 with custom transformer blocks using standalone TransformerEngine components.
Avoids TransformerLayer to work around FP8 backward pass issues.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

class CustomTransformerBlock(nn.Module):
    """Custom transformer block using standalone TE components."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention components
        self.ln1 = te.LayerNorm(hidden_size)
        self.qkv = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = te.Linear(hidden_size, hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN components
        self.ln2 = te.LayerNorm(hidden_size)
        self.fc1 = te.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.fc2 = te.Linear(4 * hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq, batch, hidden]
        S, B, _ = x.shape

        # Self-attention
        residual = x
        x = self.ln1(x)

        # Compute Q, K, V
        qkv = self.qkv(x)  # [S, B, 3*H]
        qkv = qkv.reshape(S, B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 1, 3, 0, 4)  # [3, B, heads, S, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, S, head_dim]

        # Scaled dot-product attention (manual)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # [B, heads, S, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.hidden_size)
        attn_out = attn_out.transpose(0, 1)  # [S, B, H]

        attn_out = self.proj(attn_out)
        x = residual + self.dropout(attn_out)

        # FFN
        residual = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = residual + self.dropout(x)

        return x


class GPT2CustomModel(nn.Module):
    def __init__(self, vocab_size=50264, n_positions=1024, n_embd=768,
                 n_layer=12, n_head=12, dropout=0.1):
        super().__init__()

        # Token & position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CustomTransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = te.LayerNorm(n_embd)
        self.lm_head = te.Linear(n_embd, vocab_size, bias=False)

        # Initialize
        self.apply(self._init_weights)
        self.to(dtype=torch.bfloat16)

        # FP8 recipe
        self.fp8_recipe = DelayedScaling(margin=0, fp8_format=Format.E4M3)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding,)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (te.Linear,)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

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
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"

    print("Testing custom GPT-2 model with FP8...")
    model = GPT2CustomModel(n_layer=2).to(device)  # Fewer layers for quick test

    # Test forward/backward
    B, S = 2, 128
    x = torch.randint(0, 50264, (B, S), device=device)
    y = torch.randint(0, 50264, (B, S), device=device)

    # Forward with FP8
    logits = model(x, use_fp8=True)
    loss = F.cross_entropy(logits.view(-1, 50264), y.view(-1))

    # Backward
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"✓ Custom model with FP8 works! Loss: {loss.item():.4f}")