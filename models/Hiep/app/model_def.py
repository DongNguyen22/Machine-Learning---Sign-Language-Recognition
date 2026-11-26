"""Kiến trúc tự xây: Self-Attention + Residual MLP (SA-ResMLP)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.norm(x + self.drop(self.proj(out)))


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SA_ResMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, n_tokens: int = 8, tok_dim: int = 64):
        super().__init__()
        self.n_tokens = n_tokens
        self.tok_dim = tok_dim
        hidden_dim = n_tokens * tok_dim

        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attn = SelfAttentionBlock(embed_dim=tok_dim, num_heads=4, dropout=0.1)
        self.res1 = ResMLPBlock(hidden_dim, dropout=0.25)
        self.res2 = ResMLPBlock(hidden_dim, dropout=0.25)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.in_proj(x).view(B, self.n_tokens, self.tok_dim)
        h = self.attn(h).reshape(B, -1)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h)
