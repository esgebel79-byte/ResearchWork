"""LSTM модель для регрессии временных рядов (PyTorch).

Класс `LSTMRegressor` принимает на вход последовательность лагов и возвращает
скалярный прогноз. Модуль минималистичен и пригоден для интеграции в train.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Простой LSTM-регрессор.

    Параметры:
    - input_size: число входных признаков (обычно 1 — целевой ряд)
    - hidden_size: размер скрытого представления
    - num_layers: число LSTM-слоёв
    - dropout: dropout между слоями
    - seq_len: длина входной последовательности
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, seq_len: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) or (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # берем последнее скрытое состояние
        h_last = out[:, -1, :]
        return self.fc(h_last).squeeze(-1)
