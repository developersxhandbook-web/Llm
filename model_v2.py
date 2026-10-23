import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / (C ** 0.5)
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0, float("-inf")
        )
        return F.softmax(scores, dim=-1) @ v

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attention = SelfAttention(n_embd, block_size)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.norm2 = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.lm_head(x)

print("Transformer file recreated successfully!")
