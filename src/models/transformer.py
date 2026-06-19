"""Transformer-регрессор для временных рядов (PyTorch).

Реализует простой encoder-only Transformer, принимающий последовательность
лагов и возвращающий скалярный прогноз.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    """Небольшой Transformer encoder для регрессии временных рядов.

    Параметры:
    - seq_len: длина входной последовательности
    - d_model: размер эмбеддинга
    - nhead: число голов в multihead attention
    - nlayers: число encoder-слоёв
    """

    def __init__(self, seq_len: int = 56, d_model: int = 64, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.out = nn.Linear(d_model * seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) or (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, s, _ = x.shape
        x = self.input_proj(x)  # (b, s, d)
        h = x.permute(1, 0, 2)  # (s, b, d)
        h = self.transformer(h)
        h = h.permute(1, 0, 2).reshape(b, -1)
        return self.out(h).squeeze(-1)
