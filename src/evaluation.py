"""
Evaluation utilities focused on critical transition / bifurcation-local metrics.

Functions:
- detect_bifurcations(df, target_col, win=7, slope_change_threshold=2.0):
    detect candidate transition indices (ds values) per unique_id
- localized_metrics_at_bifurcations(df, model, dataset_windows, targets, pre_window=(7,14)):
    compute MSE_at_bifurcation and PCE_at_bifurcation for windows preceding each bifurcation

This module expects the processed & EWS-augmented DataFrame produced by
`src.features.enrich_features` and a model implementing `forward(x, ews)`.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from src.features import rolling_variance, rolling_ar1
from src.models.pr_patch import cumulative_pce


def detect_bifurcations(series: pd.Series, win: int = 7, slope_change_threshold: float = 2.0) -> List[pd.Timestamp]:
    """Detect candidate bifurcation dates in a single series.

    Returns list of timestamps corresponding to predicted target times (the day of the predicted value)
    where a slope change exceeded threshold*std.
    """
    arr = series.values.astype(float)
    n = len(arr)
    if n < win * 3:
        return []
    slopes = []
    X = np.arange(win).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression
    for i in range(n - win + 1):
        y = arr[i : i + win]
        if np.all(np.isnan(y)):
            slopes.append(0.0)
            continue
        lr = LinearRegression().fit(X, y)
        slopes.append(float(lr.coef_[0]))
    slopes = np.array(slopes)
    ds = np.diff(slopes)
    thresh = slope_change_threshold * np.nanstd(ds)
    idx = np.where(np.abs(ds) > thresh)[0]
    # map to prediction time index (last point of window + 1)
    t_indices = (idx + 1) + (win - 1)
    return t_indices.tolist()


def localized_metrics_at_bifurcations(
    df: pd.DataFrame,
    unique_id: str,
    target_col: str,
    model: torch.nn.Module,
    seq_len: int = 56,
    patch_size: int = 7,
    horizon: int = 14,
    pre_window: Tuple[int, int] = (7, 14),
    ews_cols: Optional[List[str]] = None,
    physics_loss_module: Optional[object] = None,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Compute localized MSE and cumulative PCE around detected bifurcations for a single series.

    - df: processed DataFrame with date column `ds` and `unique_id` column
    - unique_id: which series to evaluate
    - target_col: which target column to use as model target (one of configured targets)
    - model: a PRPatch-like model (expects x: (b, seq_len, C), ews: (b, seq_len, n_ews))
    - pre_window: tuple (min_days, max_days) defining windows leading up to bifurcation

    Returns dictionary with aggregated metrics: mean MSE_at_bifurcation and mean PCE_at_bifurcation.
    """
    df_loc = df[df['unique_id'] == unique_id].sort_values('ds').reset_index(drop=True)
    series = df_loc[target_col]
    idxs = detect_bifurcations(series, win=7)
    if len(idxs) == 0:
        return {'n_bifurcations': 0, 'mse_at_bifurcation': float('nan'), 'pce_at_bifurcation': float('nan')}

    mse_list = []
    pce_list = []

    # build EWS matrix if columns provided
    if ews_cols is None:
        ews_cols = [c for c in df_loc.columns if c.startswith('var_') or c.startswith('ar1_')]

    for t_idx in idxs:
        # consider a set of pre-windows between pre_window[0] and pre_window[1]
        start_min = max(0, t_idx - pre_window[1])
        start_max = max(0, t_idx - pre_window[0])
        # for each possible start in this range evaluate model predictions and metrics
        for start in range(start_min, start_max + 1):
            end = start + seq_len
            if end >= len(df_loc):
                continue
            window = df_loc.iloc[start:end]
            # build multivariate input: attempt to include S/I/R-like channels if available
            # here we use the target column only plus any additional numeric channels present
            numeric_cols = [c for c in df_loc.columns if df_loc[c].dtype.kind in 'fi' and c not in ('unique_id', 'ds')]
            # ensure target is first
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            cols = [target_col] + numeric_cols
            X = window[cols].values.astype(float)
            X = X.reshape(1, X.shape[0], X.shape[1])
            # ews
            if len(ews_cols) > 0:
                EWS = window[ews_cols].fillna(0).values.astype(float)
                EWS = EWS.reshape(1, EWS.shape[0], EWS.shape[1])
            else:
                EWS = None

            # convert to tensors
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            ews_t = torch.tensor(EWS, dtype=torch.float32, device=device) if EWS is not None else None
            model.to(device)
            model.eval()

            # Generate H-step forecast trajectory. Support two modes:
            # - model returns sequence (batch, H, C)
            # - model returns single-step (batch, C) -> use autoregressive iterative forecasting
            preds_seq = []
            with torch.no_grad():
                out = model(x_t, ews_t)
                if out.ndim == 3 and out.shape[1] >= horizon:
                    # model returned multi-step predictions
                    pred_seq_tensor = out[:, :horizon, :]
                else:
                    # iterative autoregressive forecasting
                    cur_x = x_t.clone()
                    cur_ews = ews_t.clone() if ews_t is not None else None
                    steps = horizon
                    preds_step = []
                    for h in range(steps):
                        p = model(cur_x, cur_ews)
                        # ensure p shape (batch, C)
                        if p.ndim == 1:
                            p = p.unsqueeze(0)
                        preds_step.append(p)
                        # roll input: drop first timestep and append predicted as last row
                        p_np = p.cpu().numpy()
                        # construct new cur_x by shifting along time dim
                        cur_x_np = cur_x.cpu().numpy()
                        # remove oldest and append p as last time step
                        cur_x_np = np.concatenate([cur_x_np[:, 1:, :], p_np.reshape(p_np.shape[0], 1, p_np.shape[1])], axis=1)
                        cur_x = torch.tensor(cur_x_np, dtype=torch.float32, device=device)
                        # shift EWS: repeat last row (conservative) if present
                        if cur_ews is not None:
                            ews_np = cur_ews.cpu().numpy()
                            last_ews = ews_np[:, -1:, :]
                            cur_ews_np = np.concatenate([ews_np[:, 1:, :], last_ews], axis=1)
                            cur_ews = torch.tensor(cur_ews_np, dtype=torch.float32, device=device)
                    pred_seq_tensor = torch.cat(preds_step, dim=0).permute(1, 0, 2) if len(preds_step) > 0 else torch.empty((1, 0, x_t.shape[2]), device=device)

            # pred_seq_tensor shape (1, H, C)
            # ground-truth future values if available
            future_end = end + horizon
            if future_end <= len(df_loc) - 1:
                true_future = df_loc.iloc[end: end + horizon][cols].values.astype(float)
            else:
                true_future = None

            # compute H-step mse for target channel (first column)
            try:
                pred_np = pred_seq_tensor.cpu().numpy()[0]
                if true_future is not None and true_future.shape[0] >= pred_np.shape[0]:
                    mse_h = float(np.mean((pred_np[:, 0] - true_future[: pred_np.shape[0], 0]) ** 2))
                else:
                    mse_h = float(np.mean((pred_np[:, 0]) ** 2))
            except Exception:
                mse_h = float('nan')
            mse_list.append(mse_h)

            # compute cumulative PCE across the H-step predicted trajectory
            try:
                # determine N: if 3-channel SIR, use last observed S+I+R; else use ones
                if x_t.shape[2] >= 3:
                    last_obs = x_t.cpu().numpy()[0, -1, 0:3]
                    N = float(np.sum(last_obs))
                else:
                    N = 1.0
                # retrieve beta/gamma from physics_loss_module if provided
                if physics_loss_module is not None and hasattr(physics_loss_module, 'beta') and hasattr(physics_loss_module, 'gamma'):
                    beta_val = float(physics_loss_module.beta.detach().cpu().numpy())
                    gamma_val = float(physics_loss_module.gamma.detach().cpu().numpy())
                else:
                    beta_val = 0.0
                    gamma_val = 0.0
                pce_val = float(cumulative_pce(pred_seq_tensor.cpu(), beta_val, gamma_val, N).cpu().numpy())
            except Exception:
                pce_val = float('nan')
            pce_list.append(pce_val)

    out = {
        'n_bifurcations': len(idxs),
        'mse_at_bifurcation': float(np.nanmean(mse_list)) if len(mse_list) > 0 else float('nan'),
        'pce_at_bifurcation': float(np.nanmean([v for v in pce_list if not np.isnan(v)])) if any([not np.isnan(v) for v in pce_list]) else float('nan')
    }
    return out


if __name__ == '__main__':
    print('Module src.evaluation — import functions into your run scripts and call `localized_metrics_at_bifurcations`.')
