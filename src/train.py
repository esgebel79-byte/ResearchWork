"""Скрипт обучения моделей: LSTM, Transformer, PR-Patch.

Функционал:
- загрузка конфигурации
- подготовка датасета (sliding windows) из data/processed/
- инициализация выбранной модели
- цикл обучения с логированием в TensorBoard
- вычисление PCE на валидации и сохранение лучшего чекпоинта

Запуск: `python -m src.train --model pr_patch --lambda 0.1`
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from artifacts_manager import ArtifactsManager
from neuralforecast_integration import NFTrainingMonitor

# модели
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.models.pr_patch import PRPatchModel, PhysicsRegularizedLoss, cumulative_pce

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger(__name__)


def normalize_ews_columns(df: pd.DataFrame, window: int | None = None) -> list[str]:
    """Return all EWS columns matching the supported naming styles."""
    if df.empty:
        return []
    cols = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        if c.startswith('var_') or c.startswith('ar1_'):
            if window is None:
                cols.append(c)
            elif c.endswith(f'_{window}'):
                cols.append(c)
            elif c.startswith('var_') and c.count('_') == 1 and c.endswith(str(window)):
                cols.append(c)
            elif c.startswith('ar1_') and c.count('_') == 1 and c.endswith(str(window)):
                cols.append(c)
    return cols


class SlidingWindowDataset(Dataset):
    """Dataset для скольжения окна по мультивариантному временному ряду.

    Ожидает DataFrame с колонками `unique_id`, `ds`, и компонентами `S`, `I`, `R`.
    Также может принимать EWS-признаки (например `var_{w}`, `ar1_{w}`), которые
    включаются в выборку отдельно.

    Для каждой выборки возвращается кортеж (X, ews, y, meta) где:
      - X: (seq_len, C_in)
      - ews: (seq_len, n_ews) или zeros
      - y: (C_out,) следующая точка для S,I,R
      - meta: dict с 'unique_id' и 'target_ds'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        ews_cols: list | None = None,
        comps: list | None = None,
        target_scaler: StandardScaler | None = None,
        ews_scaler: StandardScaler | None = None,
    ):
        self.seq_len = seq_len
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        self.X = []
        self.ews = []
        self.y = []
        self.meta = []
        self.comps = comps or ['S', 'I', 'R']
        self.ews_cols = list(ews_cols or [])
        self.target_scaler = target_scaler
        self.ews_scaler = ews_scaler

        for uid, g in df.groupby('unique_id'):
            g = g.reset_index(drop=True)
            if not all(c in g.columns for c in self.comps):
                if 'y' in g.columns:
                    g['I'] = g['y']
                    g['S'] = 0.0
                    g['R'] = 0.0
                else:
                    target_candidates = [
                        c for c in g.columns
                        if c.lower() in {'occupied_beds_calculated', 'y', 'target'} or 'beds' in c.lower()
                    ]
                    if not target_candidates:
                        raise ValueError(f"DataFrame must contain components {self.comps} or 'y'")
                    target_col = target_candidates[0]
                    g['I'] = pd.to_numeric(g[target_col], errors='coerce')
                    g['I'] = g['I'].ffill().fillna(0.0)
                    g['S'] = 0.0
                    g['R'] = 0.0

            if 'S' not in g.columns or 'R' not in g.columns:
                N_default = max(1.0, float(pd.to_numeric(g['I'], errors='coerce').fillna(0.0).max() if 'I' in g.columns else 1.0)) * 10.0
                g['S'] = np.maximum(N_default - pd.to_numeric(g['I'], errors='coerce').fillna(0.0), 0.0)
                g['R'] = 0.0
            elif 'S' in g.columns and 'R' in g.columns:
                g['S'] = pd.to_numeric(g['S'], errors='coerce').fillna(0.0)
                g['R'] = pd.to_numeric(g['R'], errors='coerce').fillna(0.0)

            arr_S = pd.to_numeric(g['S'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            arr_I = pd.to_numeric(g['I'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            arr_R = pd.to_numeric(g['R'], errors='coerce').fillna(0.0).to_numpy(dtype=float)

            if self.target_scaler is not None:
                arr_S = self.target_scaler.transform(arr_S.reshape(-1, 1)).reshape(-1)
                arr_I = self.target_scaler.transform(arr_I.reshape(-1, 1)).reshape(-1)
                arr_R = self.target_scaler.transform(arr_R.reshape(-1, 1)).reshape(-1)

            ews_candidates = [c for c in self.ews_cols if c in g.columns]
            if ews_candidates:
                ews_frame = g[ews_candidates].copy()
                ews_frame = ews_frame.select_dtypes(include=[np.number]).copy()
                ews_mat = ews_frame.fillna(0.0).to_numpy(dtype=float)
                if self.ews_scaler is not None and ews_mat.size > 0:
                    ews_mat = self.ews_scaler.transform(ews_mat)
            else:
                ews_mat = np.zeros((len(g), 0), dtype=float)

            for i in range(len(g) - seq_len):
                S_win = arr_S[i:i + seq_len]
                I_win = arr_I[i:i + seq_len]
                R_win = arr_R[i:i + seq_len]
                X_window = np.stack([S_win, I_win, R_win], axis=1)
                self.X.append(X_window.astype(np.float32))

                if ews_mat.shape[1] > 0:
                    self.ews.append(ews_mat[i:i + seq_len].astype(np.float32))
                else:
                    self.ews.append(np.zeros((seq_len, 0), dtype=np.float32))

                y_S = arr_S[i + seq_len]
                y_I = arr_I[i + seq_len]
                y_R = arr_R[i + seq_len]
                self.y.append(np.array([y_S, y_I, y_R], dtype=np.float32))

                target_ds = g.loc[i + seq_len, 'ds']
                if isinstance(target_ds, pd.Timestamp):
                    target_ds = target_ds.isoformat()
                self.meta.append({'unique_id': uid, 'target_ds': str(target_ds)})

        self.X = np.asarray(self.X)
        self.ews = np.asarray(self.ews)
        self.y = np.asarray(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ews[idx], self.y[idx], self.meta[idx]


def fit_scalers(df: pd.DataFrame, ews_cols: list[str]):
    """Fit StandardScaler only on the training series, never on validation data."""
    if 'I' in df.columns:
        target_values = pd.to_numeric(df['I'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    else:
        target_candidates = [c for c in df.columns if c.lower() in {'occupied_beds_calculated', 'y', 'target'} or 'beds' in c.lower()]
        if target_candidates:
            target_values = pd.to_numeric(df[target_candidates[0]], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        else:
            target_values = np.zeros(len(df), dtype=float)

    target_scaler = StandardScaler()
    target_scaler.fit(target_values.reshape(-1, 1))

    ews_scaler = StandardScaler()
    if ews_cols:
        available = [c for c in ews_cols if c in df.columns]
        if available:
            ews_values = df[available].select_dtypes(include=[np.number]).fillna(0.0).to_numpy(dtype=float)
            if ews_values.size > 0:
                ews_scaler.fit(ews_values)
            else:
                ews_scaler.fit(np.zeros((max(len(df), 1), 1), dtype=float))
        else:
            ews_scaler.fit(np.zeros((max(len(df), 1), 1), dtype=float))
    else:
        ews_scaler.fit(np.zeros((max(len(df), 1), 1), dtype=float))

    return target_scaler, ews_scaler


def train_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer, loss_fn, device: torch.device):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_phys_loss = 0.0
    for batch in loader:
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"Unexpected batch type {type(batch)} from DataLoader")
        if len(batch) == 4:
            xb, ews_batch, yb, meta_batch = batch
        elif len(batch) == 3:
            xb, ews_batch, yb = batch
        else:
            raise TypeError(f"Unexpected DataLoader batch length: {len(batch)}")

        xb = xb.to(device)
        yb = yb.to(device)
        ews_batch = ews_batch.to(device) if isinstance(ews_batch, torch.Tensor) else None
        opt.zero_grad()

        try:
            preds = model(xb, ews_batch)
        except TypeError:
            preds = model(xb)

        if isinstance(loss_fn, PhysicsRegularizedLoss):
            loss_val, data_l, phys_l = loss_fn(preds, yb, inputs=xb, ews=ews_batch)
        else:
            data_l = nn.MSELoss()(preds, yb)
            phys_l = torch.tensor(0.0, device=device)
            loss_val = data_l

        loss_val.backward()
        opt.step()
        batch_size = xb.size(0)
        total_loss += float(loss_val.detach().cpu().numpy()) * batch_size
        total_data_loss += float(data_l.detach().cpu().numpy()) * batch_size
        total_phys_loss += float(phys_l.detach().cpu().numpy()) * batch_size

    n = len(loader.dataset)
    return total_loss / n, total_data_loss / n, total_phys_loss / n


def val_epoch(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> Tuple[float, float, float, np.ndarray, np.ndarray, list]:
    model.eval()
    total_loss = 0.0
    total_data_loss = 0.0
    total_phys_loss = 0.0
    preds_list = []
    targets_list = []
    metas = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                xb, ews_batch, yb, meta_batch = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, ews_batch, yb = batch
                meta_batch = [None] * xb.shape[0]
            else:
                xb, yb = batch
                ews_batch = None
                meta_batch = [None] * xb.shape[0]

            xb = xb.to(device)
            yb = yb.to(device)
            ews_batch = ews_batch.to(device) if isinstance(ews_batch, torch.Tensor) else None
            try:
                preds = model(xb, ews_batch)
            except Exception:
                preds = model(xb)
            if isinstance(loss_fn, PhysicsRegularizedLoss):
                loss_val, data_l, phys_l = loss_fn(preds, yb, inputs=xb, ews=ews_batch)
            else:
                data_l = nn.MSELoss()(preds, yb)
                phys_l = torch.tensor(0.0, device=device)
                loss_val = data_l

            bs = xb.size(0)
            total_loss += float(loss_val.cpu().numpy()) * bs
            total_data_loss += float(data_l.cpu().numpy()) * bs
            total_phys_loss += float(phys_l.cpu().numpy()) * bs
            preds_list.append(preds.detach().cpu().numpy())
            targets_list.append(yb.detach().cpu().numpy())
            if isinstance(meta_batch, dict):
                batch_meta = []
                n_items = max(len(v) for v in meta_batch.values()) if meta_batch else 0
                for i in range(n_items):
                    batch_meta.append({key: value[i] for key, value in meta_batch.items()})
                metas.extend(batch_meta)
            elif isinstance(meta_batch, (list, tuple)):
                normalized = []
                for item in meta_batch:
                    if isinstance(item, dict):
                        normalized.append(item)
                    elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                        normalized.append({'unique_id': item[0], 'target_ds': str(item[1])})
                    elif item is None:
                        continue
                    else:
                        normalized.append({'unique_id': str(item), 'target_ds': ''})
                metas.extend(normalized)
            else:
                metas.extend([])
    preds_all = np.concatenate(preds_list)
    targets_all = np.concatenate(targets_list)
    n = len(loader.dataset)
    return total_loss / n, total_data_loss / n, total_phys_loss / n, preds_all, targets_all, metas


def physics_step_pce(preds: torch.Tensor, inputs_or_targets: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> float:
    """Compute the paper-style residual norm using the last observed SIR state and the one-step forecast.
    
    Can accept either full input history or just the target step (both shapes: (batch, 3)).
    """
    if preds.dim() != 2 or preds.shape[1] < 3:
        return float('nan')

    if inputs_or_targets.dim() == 3 and inputs_or_targets.shape[1] > 1:
        S_t = inputs_or_targets[:, -1, 0]
        I_t = inputs_or_targets[:, -1, 1]
        R_t = inputs_or_targets[:, -1, 2]
    elif inputs_or_targets.dim() == 2 and inputs_or_targets.shape[1] == 3:
        S_t = inputs_or_targets[:, 0]
        I_t = inputs_or_targets[:, 1]
        R_t = inputs_or_targets[:, 2]
    else:
        return float('nan')

    N = (S_t + I_t + R_t).clamp(min=1.0)
    beta_t = beta if isinstance(beta, torch.Tensor) else torch.tensor(float(beta), device=preds.device)
    gamma_t = gamma if isinstance(gamma, torch.Tensor) else torch.tensor(float(gamma), device=preds.device)

    resid_S = (preds[:, 0] - S_t) + beta_t * S_t * I_t / (N + 1e-8)
    resid_I = (preds[:, 1] - I_t) - (beta_t * S_t * I_t / (N + 1e-8) - gamma_t * I_t)
    residual_vec = torch.stack([resid_S, resid_I], dim=-1)
    return float(torch.mean(torch.linalg.norm(residual_vec, dim=-1)).detach().cpu().numpy())


def evaluate_physics_pce(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)):
                continue
            if len(batch) == 4:
                xb, ews_batch, yb, _ = batch
            elif len(batch) == 3:
                xb, ews_batch, yb = batch
            else:
                continue
            xb = xb.to(device)
            ews_batch = ews_batch.to(device) if isinstance(ews_batch, torch.Tensor) else None
            try:
                preds = model(xb, ews_batch)
            except Exception:
                preds = model(xb)
            beta = getattr(loss_fn, 'beta', None)
            gamma = getattr(loss_fn, 'gamma', None)
            if beta is None or gamma is None:
                continue
            score = physics_step_pce(preds, xb, beta, gamma)
            if np.isfinite(score):
                scores.append(score)
    return float(np.mean(scores)) if scores else float('inf')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'pr_patch'], default='pr_patch')
    parser.add_argument('--lambda', dest='lam', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    artifacts = ArtifactsManager(base_dir=cfg.get('artifacts', {}).get('base_dir', 'artifacts'))
    tb_dir = cfg.get('logging', {}).get('tensorboard_dir')
    monitor = NFTrainingMonitor(artifacts=artifacts, model_name=args.model, horizon=cfg['training']['horizon'])

    proc = Path(cfg['data']['processed_dir']) / 'data_daily.csv'
    if not proc.exists():
        from src.data_ingestion import DataIngestion
        ing = DataIngestion(cfg)
        proc = ing.run()

    df = pd.read_csv(proc, parse_dates=[cfg.get('data', {}).get('date_col', 'ds')] if cfg.get('data', {}).get('date_col', 'ds') in pd.read_csv(proc, nrows=0).columns else [])
    seq_len = cfg['training']['seq_len']

    ews_window = cfg.get('ews', {}).get('rolling_window', 14)
    ews_cols = [f'var_{ews_window}', f'ar1_{ews_window}']
    if not all(col in df.columns for col in ews_cols):
        from src.features import enrich_features
        df = enrich_features(proc, window=ews_window)
        ews_cols = normalize_ews_columns(df, window=ews_window)
        if not ews_cols:
            ews_cols = [c for c in df.columns if c.startswith('var_') or c.startswith('ar1_')]
        if not ews_cols:
            raise ValueError(f"No EWS columns generated for window={ews_window} from {proc}")

    if 'unique_id' not in df.columns:
        df['unique_id'] = 'series_0'
    unique_series = list(dict.fromkeys(df['unique_id'].astype(str).tolist()))

    if len(unique_series) >= 2:
        val_series_count = max(1, int(np.ceil(0.2 * len(unique_series))))
        val_series = unique_series[-val_series_count:]
        train_series = [uid for uid in unique_series if uid not in set(val_series)]
    else:
        train_series = unique_series
        val_series = []

    train_df = df[df['unique_id'].astype(str).isin(train_series)].copy() if train_series else df.copy()
    target_scaler, ews_scaler = fit_scalers(train_df, ews_cols)

    dataset = SlidingWindowDataset(
        df,
        seq_len,
        ews_cols=ews_cols,
        target_scaler=target_scaler,
        ews_scaler=ews_scaler,
    )
    if len(dataset) == 0:
        raise ValueError(f"No training windows produced for dataset: {proc}")

    if len(unique_series) >= 2:
        train_mask = np.array([meta['unique_id'] in set(train_series) for _, _, _, meta in dataset], dtype=bool)
        val_mask = np.array([meta['unique_id'] in set(val_series) for _, _, _, meta in dataset], dtype=bool)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
    else:
        val_n = max(1, int(0.2 * len(dataset)))
        train_n = len(dataset) - val_n
        indices = np.arange(len(dataset))
        train_idx = indices[:train_n]
        val_idx = indices[train_n:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    batch_size = args.batch_size if args.batch_size is not None else cfg['training'].get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device(args.device)
    if args.model == 'lstm':
        model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=2, seq_len=seq_len)
        loss_fn = nn.MSELoss()
    elif args.model == 'transformer':
        model = TransformerRegressor(seq_len=seq_len, d_model=64, nhead=4, nlayers=2)
        loss_fn = nn.MSELoss()
    else:
        n_ews = max(0, len(ews_cols))
        model = PRPatchModel(seq_len=seq_len, patch_size=cfg.get('patchtst', {}).get('patch_size', 7), hidden=64, n_inputs=3, n_ews=n_ews)
        loss_fn = PhysicsRegularizedLoss(lambda_base=args.lam)

    model.to(device)
    if hasattr(loss_fn, 'raw_beta'):
        assert loss_fn.raw_beta.requires_grad is True, 'beta must require_grad=True'
    if hasattr(loss_fn, 'raw_gamma'):
        assert loss_fn.raw_gamma.requires_grad is True, 'gamma must require_grad=True'
    optimizer_params = list(model.parameters())
    if hasattr(loss_fn, 'parameters'):
        optimizer_params.extend(loss_fn.parameters())
    opt = optim.Adam(optimizer_params, lr=cfg['training'].get('lr', 1e-3))

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_dir)
    except Exception:
        writer = None
        _LOG.warning("TensorBoard SummaryWriter not available; skipping TB logging")

    best_pce = float('inf')
    best_path = None

    epochs = args.epochs if args.epochs is not None else cfg['training'].get('epochs', 10)
    for epoch in range(1, epochs + 1):
        train_loss, train_data_loss, train_phys_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_data_loss, val_phys_loss, preds_val, targets_val, metas = val_epoch(model, val_loader, loss_fn, device)

        try:
            phys_pce = evaluate_physics_pce(model, val_loader, loss_fn, device)
        except Exception:
            phys_pce = float('inf')

        _LOG.info("Epoch %d train_loss=%.6f val_loss=%.6f physics_pce=%.6f", epoch, train_loss, val_loss, phys_pce)

        if writer is not None:
            writer.add_scalar('train/total_loss', train_loss, epoch)
            writer.add_scalar('train/data_loss', train_data_loss, epoch)
            writer.add_scalar('train/phys_loss', train_phys_loss, epoch)
            writer.add_scalar('val/total_loss', val_loss, epoch)
            writer.add_scalar('val/data_loss', val_data_loss, epoch)
            writer.add_scalar('val/phys_loss', val_phys_loss, epoch)
            writer.add_scalar('val/physics_pce', phys_pce, epoch)

        # --- Bifurcation-local metrics ---
        target_col = 'target' if 'target' in df.columns else ('I' if 'I' in df.columns else 'y')
        def detect_bifurcation_dates(df_local, id_col='unique_id', date_col='ds', target_col=target_col, win=7, threshold_scale=2.0):
            dates_map = {}
            for uid, g in df_local.groupby(id_col):
                g = g.sort_values(date_col).reset_index(drop=True)
                if target_col not in g.columns:
                    continue
                vals = g[target_col].values
                if len(vals) < win * 2:
                    dates_map[uid] = []
                    continue
                slopes = []
                X = np.arange(win).reshape(-1, 1)
                from sklearn.linear_model import LinearRegression
                for i in range(len(vals) - win + 1):
                    y = vals[i:i + win]
                    lr = LinearRegression().fit(X, y)
                    slopes.append(lr.coef_[0])
                slopes = np.array(slopes)
                ds = np.diff(slopes)
                thresh = threshold_scale * (np.nanstd(ds) + 1e-8)
                idx = np.where(np.abs(ds) > thresh)[0]
                t_indices = (idx + 1) + (win - 1)
                dates = []
                for ti in t_indices:
                    start = max(0, ti - 14)
                    end = max(0, ti - 7)
                    dates.extend(g.loc[start:end, date_col].tolist())
                dates_map[uid] = set(dates)
            return dates_map

        bif_map = detect_bifurcation_dates(df)
        bif_indices = []
        for idx, meta in enumerate(metas):
            if meta is None or not isinstance(meta, dict):
                continue
            uid = meta.get('unique_id')
            tds = meta.get('target_ds')
            if uid in bif_map and tds in bif_map[uid]:
                bif_indices.append(idx)

        if bif_indices:
            preds_bif = preds_val[bif_indices]
            targets_bif = targets_val[bif_indices]
            mse_bif = float(np.mean((preds_bif[:,1] - targets_bif[:,1])**2))
            pce_bif = phys_pce
        else:
            mse_bif = float('nan')
            pce_bif = float('nan')

        if writer is not None:
            writer.add_scalar('val/mse_bifurcation', mse_bif, epoch)
            writer.add_scalar('val/pce_bifurcation', pce_bif, epoch)

        if np.isfinite(phys_pce) and phys_pce < best_pce:
            best_pce = phys_pce
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': epoch,
                'pce': float(phys_pce),
                'lambda_base': getattr(loss_fn, 'lambda_base', None),
                'alpha': getattr(loss_fn, 'alpha', None),
                'beta': (loss_fn.beta.item() if hasattr(loss_fn, 'beta') else None),
                'gamma': (loss_fn.gamma.item() if hasattr(loss_fn, 'gamma') else None),
            }
            best_path = artifacts.save_checkpoint(state, model_name=args.model, arch=args.model, epoch=epoch)

        _LOG.info("Epoch %d cumulative_pce=%.6f best_checkpoint=%s", epoch, phys_pce, best_path)

    if writer is not None:
        writer.close()
    _LOG.info("Training finished. Best PCE=%.6f saved to %s", best_pce, best_path)


if __name__ == '__main__':
    main()
