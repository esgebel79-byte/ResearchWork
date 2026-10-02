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

    def __init__(self, seq_len: int = 56, patch_size: int = 7, hidden: int = 64, n_inputs: int = 3, n_ews: int = 2, forecast_horizon: int = 1):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len должен делиться на patch_size"
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.n_inputs = n_inputs
        self.n_ews = n_ews
        self.forecast_horizon = max(1, int(forecast_horizon))
        # Store last attention weights for visualization
        self.last_attn_weights = None

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
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_inputs)
        )
        self.direct_head = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_inputs * self.forecast_horizon)
        )

    @staticmethod
    def future_ews_from_history(ews: Optional[torch.Tensor], horizon: int, fill_value: Optional[float] = None, method: str = 'linear') -> Optional[torch.Tensor]:
        """Create a leakage-free future EWS tensor for a forecast horizon.

        Strategy: Linear interpolation or AR(1) prediction of EWS trend instead of just holding constant.
        This preserves the dynamics of approaching bifurcation over the forecast horizon.
        
        Args:
            ews: (batch, seq_len, n_ews) historical EWS values
            horizon: number of future steps to predict
            fill_value: override value (if provided, use constant fill)
            method: 'linear' (interpolation) [default] or 'constant' (hold last value)
        
        Returns:
            (batch, horizon, n_ews) future EWS predictions, leakage-free
        """
        if ews is None:
            return None
        if ews.dim() != 3:
            return ews
        
        if horizon <= 1:
            return ews
        
        batch_size, seq_len, n_ews_dim = ews.shape
        device = ews.device
        
        if fill_value is not None:
            # Override: use constant fill value
            last = torch.full((batch_size, 1, n_ews_dim), fill_value=float(fill_value), device=device)
            future = last.expand(-1, horizon, -1)
            return future
        
        if method == 'linear':
            # Linear interpolation: estimate trend from last k points, extrapolate forward
            # Use last 3 points to estimate slope (robust to noise)
            k = min(3, seq_len)
            if seq_len >= k:
                slope = (ews[:, -1:, :] - ews[:, -k:-k+1, :]) / (k - 1.0)  # (batch, 1, n_ews)
            else:
                slope = torch.zeros((batch_size, 1, n_ews_dim), device=device)
            
            last_ews = ews[:, -1:, :]  # (batch, 1, n_ews)
            future_steps = torch.arange(1, horizon + 1, dtype=torch.float32, device=device)  # (horizon,)
            future = last_ews + slope * future_steps.view(1, -1, 1)  # (batch, horizon, n_ews)
            return future
        
        else:  # method == 'constant'
            # Hold last value constant (original strategy)
            last = ews[:, -1:, :].clone()
            future = last.expand(-1, horizon, -1)
            return future

    def forward(self, x: torch.Tensor, ews: Optional[torch.Tensor] = None, return_attention: bool = False):
        """Прямой проход.

        x: (batch, seq_len, C)
        ews: (batch, seq_len, n_ews) либо None — агрегируем EWS по последнему окну
        Возвращает preds: (batch, C) при forecast_horizon=1 или (batch, horizon, C) при horizon>1.
        
        FIX #1 (EWS=None degeneration): When ews is None, we still project component patches through
        learnable encoders to create diverse representations. We NEVER zero-initialize ews_tokens in a
        way that would collapse all outputs. Instead, we project a learnable "default EWS" or use
        component tokens as both query AND value to avoid degeneration.
        """
        b, s, c = x.shape
        assert c >= self.n_inputs, "Ожидается как минимум n_inputs каналов (S,I,R)"

        encs_per_comp = []
        for i in range(self.n_inputs):
            comp = x[:, :, i]
            patches = comp.view(b, self.n_patches, self.patch_size)
            enc_p = [self.patch_enc[i](patches[:, p, :]) for p in range(self.n_patches)]
            encs_per_comp.append(torch.stack(enc_p, dim=1))

        comp_tokens = torch.cat(encs_per_comp, dim=1)  # (batch, n_inputs*n_patches, d_model)

        # FIX #1: Prevent degeneration when ews=None
        if ews is not None and self.ews_proj is not None and ews.shape[-1] > 0:
            # Real EWS provided: use it
            ews_p = ews.view(b, self.n_patches, self.patch_size, -1).mean(dim=2)
            ews_tokens = self.ews_proj(ews_p)
        else:
            # No EWS or empty: use component tokens as self-attention key/value
            # This ensures attention weights are DATA-DEPENDENT, not constant
            ews_tokens = comp_tokens  # Use component patches as default (ensures non-degeneracy)

        # FIX #2: Store attention weights for visualization
        attn_out, attn_weights = self.cross_attn(query=comp_tokens, key=ews_tokens, value=ews_tokens)
        self.last_attn_weights = attn_weights.detach()  # Store for visualization
        
        pooled = attn_out.mean(dim=1)  # (batch, d_model)

        # FIX #3: Direct head for multi-step forecasting (native, no rollout)
        if self.forecast_horizon > 1:
            # Out shape: (batch, n_inputs * forecast_horizon) -> reshape to (batch, forecast_horizon, n_inputs)
            out_flat = self.direct_head(pooled)  # (batch, n_inputs * horizon)
            out = out_flat.view(b, self.forecast_horizon, self.n_inputs)
            # Validate shape correctness
            assert out.shape == (b, self.forecast_horizon, self.n_inputs), \
                f"Expected {(b, self.forecast_horizon, self.n_inputs)}, got {out.shape}"
        else:
            out = self.ff(pooled)  # (batch, n_inputs)

        if return_attention:
            return out, self.last_attn_weights
        return out

    def forward_with_attention(self, x: torch.Tensor, ews: Optional[torch.Tensor] = None):
        """Helper for visualization / debugging: returns (forecast, attention_weights)."""
        return self.forward(x, ews=ews, return_attention=True)


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
        # Keep a strictly positive floor so the physical parameters do not collapse to 0.0
        self.raw_beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, requires_grad=True))
        self.raw_gamma = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, requires_grad=True))
        self.softplus = nn.Softplus()
        self.mse = nn.MSELoss()
        self.device = device

    @property
    def beta(self) -> torch.Tensor:
        return self.softplus(self.raw_beta) + 1e-3

    @property
    def gamma(self) -> torch.Tensor:
        return self.softplus(self.raw_gamma) + 1e-3

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

            phys_per_sample = torch.linalg.norm(torch.stack([resid_S, resid_I], dim=1), dim=1)
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
    """Compute the paper-style physical consistency error as the mean residual norm over a horizon.

    For a SIR trajectory, the one-step residual vector is:
      R_t = [ (S_{t+1}-S_t) + beta*S_t*I_t/N,
             (I_{t+1}-I_t) - (beta*S_t*I_t/N - gamma*I_t) ]
    and the reported score is the mean Euclidean norm of R_t over horizon H.
    """
    device = pred_seq.device
    beta_t = beta if isinstance(beta, torch.Tensor) else torch.tensor(beta, device=device)
    gamma_t = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(gamma, device=device)

    if pred_seq.dim() == 2:
        pred_seq = pred_seq.unsqueeze(1)

    if pred_seq.shape[-1] >= 3:
        S = pred_seq[:, :, 0]
        I = pred_seq[:, :, 1]
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

        if N_t.dim() == 0:
            N_use = torch.full_like(S_t, fill_value=float(N_t.item()))
        elif N_t.dim() == 1:
            N_use = N_t.unsqueeze(1).expand(-1, S_t.shape[1])
        elif N_t.dim() == 2:
            N_use = N_t[:, :-1]
        else:
            N_use = N_t

        resid_S = (S_tp1 - S_t) + beta_t * S_t * I_t / (N_use + 1e-8)
        resid_I = (I_tp1 - I_t) - (beta_t * S_t * I_t / (N_use + 1e-8) - gamma_t * I_t)
        residual_vec = torch.stack([resid_S, resid_I], dim=-1)
        residual_norm = torch.linalg.norm(residual_vec, dim=-1)
        return torch.mean(residual_norm)

    if pred_seq.shape[1] < 3:
        seq = pred_seq[:, :, 0] if pred_seq.dim() == 3 else pred_seq
        if seq.shape[1] < 3:
            return torch.tensor(1e-6, device=device)
        sec = seq[:, 2:] - 2.0 * seq[:, 1:-1] + seq[:, :-2]
        return torch.mean(torch.abs(sec)) + 1e-6

    sec = pred_seq[:, 2:, :] - 2.0 * pred_seq[:, 1:-1, :] + pred_seq[:, :-2, :]
    return torch.mean(torch.abs(sec)) + 1e-6


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