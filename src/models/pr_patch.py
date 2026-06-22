"""
Patch-based модель с физической регуляризацией (SIR) и кастомным Loss.

Модуль реализует:
- `PRPatchModel` — patch-based энкодер, принимающий мультивариантный вход
    (S, I, R) и опционально EWS-признаки (AR1, variance).
- `PhysicsRegularizedLoss` — физически обоснованный лосс для SIR в дискретной
    форме. Итоговый Loss = Data_Loss + sum_t lambda_t * PhysicsResid_t,
    где lambda_t = lambda_base * (1 + alpha * EWS_score[t]).

Математическая постановка (дискретная SIR):
    S[t+1] - S[t] = -beta * S[t] * I[t] / N
    I[t+1] - I[t] = beta * S[t] * I[t] / N - gamma * I[t]

beta и gamma — обучаемые параметры (положительные через softplus).

Содержит:
- `PRPatchModel` — простая patch-based сеть (энкодер патчей -> регрессия)
- `PhysicsRegularizedLoss` — nn.Module, вычисляющий Data_Loss (MSE)
  и Physics_Loss, итоговый Loss = mse + lambda * physics_loss

Physics_Loss в этом шаблоне реализован как штраф за несоответствие
производных/баланса между компонентами ряда (для multivariate данных).
Для одномерного ряда используется штраф на вторые разности (ускорение).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PRPatchModel(nn.Module):
    """Patch-based энкодер с модулем EWS.

    Ожидает вход `x` формы (batch, seq_len, C) где C >= 3 (S,I,R,...).
    Опционально принимает `ews` формы (batch, seq_len, n_ews) и использует её
    для управления динамикой регуляризации внутри лосса.
    """

    def __init__(self, seq_len: int = 56, patch_size: int = 7, hidden: int = 64, n_inputs: int = 3, n_ews: int = 2):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len должен делиться на patch_size"
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.n_inputs = n_inputs
        self.n_ews = n_ews

        # Проекция каждой переменной в патче
        self.patch_enc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU()
            ) for _ in range(n_inputs)
        ])

        # Attention-based fusion:
        self.d_model = hidden // 2
        # Проекция EWS в d_model для каждого патча
        self.ews_proj = nn.Linear(n_ews, self.d_model) if n_ews > 0 else None
        # Multihead cross-attention: query = component-patch tokens, key/value = ews-patch tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
        # Фидфорвард для финального вывода
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_inputs)
        )

    def forward(self, x: torch.Tensor, ews: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Прямой проход.

        x: (batch, seq_len, C)
        ews: (batch, seq_len, n_ews) либо None — агрегируем EWS по последнему окну
        Возвращает preds: (batch, C) — прогноз следующего шага для каждой компоненты.
        """
        b, s, c = x.shape
        assert c >= self.n_inputs, "Ожидается как минимум n_inputs каналов (S,I,R)"
        
        # разбиваем по компонентам и патчам
        encs_per_comp = []
        for i in range(self.n_inputs):
            comp = x[:, :, i]  # (b, s)
            patches = comp.view(b, self.n_patches, self.patch_size)
            # encode each patch -> list of (b, d_model)
            enc_p = [self.patch_enc[i](patches[:, p, :]) for p in range(self.n_patches)]
            # stack to (b, n_patches, d_model)
            encs_per_comp.append(torch.stack(enc_p, dim=1))

        # concatenate along token dimension: (b, n_patches * n_inputs, d_model)
        comp_tokens = torch.cat(encs_per_comp, dim=1)

        # prepare ews tokens per patch: aggregate within each patch (mean)
        if ews is not None and self.ews_proj is not None and ews.shape[-1] > 0:
            # ews: (b, seq_len, n_ews) -> (b, n_patches, n_ews)
            ews_p = ews.view(b, self.n_patches, self.patch_size, -1).mean(dim=2)
            ews_tokens = self.ews_proj(ews_p)  # (b, n_patches, d_model)
        else:
            ews_tokens = torch.zeros(b, self.n_patches, self.d_model, device=x.device)

        # Cross-attention: query=comp_tokens (Lq), key/value=ews_tokens (Lk)
        attn_out, _ = self.cross_attn(query=comp_tokens, key=ews_tokens, value=ews_tokens)
        # pool across tokens to get global
        pooled = attn_out.mean(dim=1)  # (b, d_model)
        out = self.ff(pooled)  # (b, n_inputs)
        return out


class PhysicsRegularizedLoss(nn.Module):
    """Physics loss for discrete SIR model.

    Вход:
    - lambda_base: базовый коэффициент регуляризации
    - alpha: масштаб влияния EWS на lambda_t
    - learnable beta/gamma: обучаемые параметры модели (nn.Parameter)
    """

    def __init__(self, lambda_base: float = 0.1, alpha: float = 1.0, device: Optional[torch.device] = None):
        super().__init__()
        self.lambda_base = float(lambda_base)
        self.alpha = float(alpha)
        self.raw_beta = nn.Parameter(torch.tensor(0.1))
        self.raw_gamma = nn.Parameter(torch.tensor(0.05))
        self.softplus = nn.Softplus()
        self.mse = nn.MSELoss()
        self.device = device

    @property
    def beta(self) -> torch.Tensor:
        return self.softplus(self.raw_beta)

    @property
    def gamma(self) -> torch.Tensor:
        return self.softplus(self.raw_gamma)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                inputs: Optional[torch.Tensor] = None, ews: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute total loss and return tuple (total_loss, data_loss, phys_loss)."""
        device = preds.device
        data_loss = self.mse(preds, targets)

        # Physics residuals computed from last timestep in inputs
        if inputs is None:
            phys_loss = torch.tensor(0.0, device=device)
            return data_loss + self.lambda_base * phys_loss, data_loss.detach(), phys_loss.detach()

        if inputs.shape[2] >= 3:
            S_hist = inputs[:, :, 0]
            I_hist = inputs[:, :, 1]
            R_hist = inputs[:, :, 2]

            S_t = S_hist[:, -1]
            I_t = I_hist[:, -1]
            R_t = R_hist[:, -1]

            S_pred = preds[:, 0]
            I_pred = preds[:, 1]

            N = (S_t + I_t + R_t).clamp(min=1.0)

            beta = self.beta
            gamma = self.gamma

            resid_S = (S_pred - S_t) - (-beta * S_t * I_t / N)
            resid_I = (I_pred - I_t) - (beta * S_t * I_t / N - gamma * I_t)

            phys_per_sample = torch.abs(resid_S) + torch.abs(resid_I)
        else:
            if inputs.shape[1] >= 3:
                sec_diff = inputs[:, -1, :] - 2.0 * inputs[:, -2, :] + inputs[:, -3, :]
            else:
                sec_diff = torch.zeros(inputs.shape[0], inputs.shape[2], device=device)
            phys_per_sample = torch.mean(torch.abs(sec_diff), dim=1)

        if ews is not None:
            if ews.dim() == 3:
                score = ews.mean(dim=1).mean(dim=1)  # (b,)
            else:
                score = ews.mean(dim=1)  # (b,)
            score_mean = score.mean()
            score_std = score.std(unbiased=False) + 1e-8
            score_norm = (score - score_mean) / score_std
            lambda_t = self.lambda_base * (1.0 + self.alpha * score_norm)
            lambda_t = torch.clamp(lambda_t, min=0.0)
        else:
            lambda_t = torch.full_like(phys_per_sample, fill_value=self.lambda_base)

        phys_loss = torch.mean(lambda_t * phys_per_sample)
        total = data_loss + phys_loss
        return total, data_loss.detach(), phys_loss.detach()


def cumulative_pce(pred_seq: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    """Вычислить кумулятивную Physics Consistency Error (PCE) по всей траектории."""
    device = pred_seq.device
    beta_t = beta if isinstance(beta, torch.Tensor) else torch.tensor(beta, device=device)
    gamma_t = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(gamma, device=device)

    if pred_seq.shape[2] >= 3:
        S = pred_seq[:, :, 0]
        I = pred_seq[:, :, 1]
        R = pred_seq[:, :, 2]
        use_sir = True
    else:
        use_sir = False

    if isinstance(N, torch.Tensor):
        N_t = N.to(device)
    else:
        N_t = torch.tensor(float(N), device=device)

    if use_sir:
        if S.shape[1] < 2:
            return torch.tensor(0.0, device=device)
        S_t = S[:, :-1]
        S_tp1 = S[:, 1:]
        I_t = I[:, :-1]
        I_tp1 = I[:, 1:]

        if N_t.dim() == 1:
            N_use = N_t.unsqueeze(1).expand(-1, S_t.shape[1])
        elif N_t.dim() == 2:
            N_use = N_t[:, :-1]
        else:
            N_use = N_t

        resid_S = (S_tp1 - S_t) + beta_t * S_t * I_t / (N_use + 1e-8)
        resid_I = (I_tp1 - I_t) - (beta_t * S_t * I_t / (N_use + 1e-8) - gamma_t * I_t)

        mse_S = torch.mean(resid_S ** 2)
        mse_I = torch.mean(resid_I ** 2)
        return 0.5 * (mse_S + mse_I)
    else:
        if pred_seq.shape[1] < 3:
            return torch.tensor(0.0, device=device)
        sec = pred_seq[:, 2:, :] - 2.0 * pred_seq[:, 1:-1, :] + pred_seq[:, :-2, :]
        return torch.mean(sec ** 2)


def _synthetic_integration_test():
    """Статическая проверка: прогон через forward + backward."""
    print("Running synthetic integration test for PRPatchModel...")
    batch_size = 4
    seq_len = 56
    patch_size = 7
    n_inputs = 3
    n_ews = 2

    S = torch.abs(torch.randn(batch_size, seq_len)) * 1000 + 1e3
    I = torch.abs(torch.randn(batch_size, seq_len)) * 10 + 10
    R = torch.abs(torch.randn(batch_size, seq_len)) * 100 + 100
    X = torch.stack([S, I, R], dim=2)
    EWS = torch.randn(batch_size, seq_len, n_ews)

    model = PRPatchModel(seq_len=seq_len, patch_size=patch_size, hidden=64, n_inputs=n_inputs, n_ews=n_ews)
    loss_fn = PhysicsRegularizedLoss(lambda_base=0.05, alpha=0.5)
    model.train()
    preds = model(X, EWS)
    
    y_true = torch.stack([S[:, -1], I[:, -1], R[:, -1]], dim=1)
    total, data_l, phys_l = loss_fn(preds, y_true, inputs=X, ews=EWS)
    print("Loss components:", float(data_l), float(phys_l), float(total))
    total.backward()
    
    beta_grad = loss_fn.raw_beta.grad
    gamma_grad = loss_fn.raw_gamma.grad
    print("beta grad:", beta_grad, "gamma grad:", gamma_grad)


if __name__ == '__main__':
    _synthetic_integration_test()