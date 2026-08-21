from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, heads: int = 4, horizon: int = 5, dropout: float = .18):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.proj = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU())
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden//2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.post = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = F.gelu(self.conv(x.transpose(1,2))).transpose(1,2)
        z, _ = self.lstm(x)
        a, _ = self.attn(z,z,z,need_weights=False)
        z = self.post(z+a)
        pooled = torch.cat([z[:,-1,:], z.mean(1)], dim=1)
        return self.head(pooled)
