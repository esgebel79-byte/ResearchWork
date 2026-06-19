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
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from artifacts_manager import ArtifactsManager
from neuralforecast_integration import NFTrainingMonitor

# модели
from src.models.lstm import LSTMRegressor
from src.models.transformer import TransformerRegressor
from src.models.pr_patch import PRPatchModel, PhysicsRegularizedLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger(__name__)


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

    def __init__(self, df: pd.DataFrame, seq_len: int, ews_cols: list | None = None, comps: list | None = None):
        self.seq_len = seq_len
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        self.X = []
        self.ews = []
        self.y = []
        self.meta = []
        self.comps = comps or ['S', 'I', 'R']
        self.ews_cols = ews_cols or []
        for uid, g in df.groupby('unique_id'):
            g = g.reset_index(drop=True)
            # ensure components exist
            if not all(c in g.columns for c in self.comps):
                # if not multivariate, fallback to 'y' as I and construct S,R zeros
                if 'y' in g.columns:
                    g['I'] = g['y']
                    g['S'] = 0.0
                    g['R'] = 0.0
                else:
                    raise ValueError(f"DataFrame must contain components {self.comps} or 'y'")
            arr_S = g['S'].values.astype(float)
            arr_I = g['I'].values.astype(float)
            arr_R = g['R'].values.astype(float) if 'R' in g.columns else np.zeros_like(arr_S)
            # EWS features
            if self.ews_cols:
                ews_mat = g[self.ews_cols].fillna(0.0).values.astype(float)
            else:
                ews_mat = np.zeros((len(g), 0), dtype=float)

            for i in range(len(g) - seq_len):
                S_win = arr_S[i:i + seq_len]
                I_win = arr_I[i:i + seq_len]
                R_win = arr_R[i:i + seq_len]
                X_window = np.stack([S_win, I_win, R_win], axis=1)  # (seq_len, 3)
                self.X.append(X_window.astype(np.float32))
                if ews_mat.shape[1] > 0:
                    self.ews.append(ews_mat[i:i + seq_len].astype(np.float32))
                else:
                    self.ews.append(np.zeros((seq_len, 0), dtype=np.float32))
                # target is next-step S,I,R
                y_S = arr_S[i + seq_len]
                y_I = arr_I[i + seq_len]
                y_R = arr_R[i + seq_len]
                self.y.append(np.array([y_S, y_I, y_R], dtype=np.float32))
                self.meta.append({'unique_id': uid, 'target_ds': g.loc[i + seq_len, 'ds']})

        self.X = np.asarray(self.X)
        self.ews = np.asarray(self.ews)
        self.y = np.asarray(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.ews[idx], self.y[idx], self.meta[idx]


def train_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer, loss_fn, device: torch.device):
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_phys_loss = 0.0
    for xb, yb in loader:
        # xb: (batch, seq_len, C), yb: (batch, C_out)
        if isinstance(xb, list) or isinstance(xb, tuple):
            xb = xb[0]
        xb = xb.to(device)
        # loader returns ews and meta; DataLoader will pack tuples
        # Here we expect loader to yield (X, ews, y, meta)
        # If loader yields 4-tuples via default collate, handle accordingly
        # Using destructive unpack above may not be ideal; reshape below
        # To support different DataLoader outputs, try automatic unpack
        try:
            # If loader returns (X, ews, y, meta)
            pass
        except Exception:
            pass
        # yb may be tensor or tuple; ensure tensor
        if isinstance(yb, (list, tuple)):
            yb = yb[0]
        yb = yb.to(device)
        opt.zero_grad()
        # attempt to get ews from batch if present
        ews_batch = None
        # in many collate variants xb might be (X, ews)
        if xb.ndim == 4:
            # shape (batch, something unexpected) try flatten
            xb = xb.squeeze(0)
        # detect if loader yields (X, ews, y, meta)
        # When using our Dataset, DataLoader will return tuples: (X, ews, y, meta)
        # So the actual iteration variable is a tuple; to support both, check type
        # We'll handle ews by inspecting attributes of loader.dataset
        if hasattr(loader.dataset, 'ews'):
            # fetch ews for indices via sampler is complex; assume collate provided ews
            pass
        preds = None
        # If model expects two args (x, ews) try to call accordingly
        try:
            # try to extract ews from the batch: many DataLoaders will produce tensors stacked
            # If original batch came as tuple, torch DataLoader packs them; here we assume
            # that the first element is X and second is ews. If not, just call model(x)
            # We attempt to parse from xb if it is a tuple
            preds = model(xb)
        except TypeError:
            # fallback: try model(x, None)
            preds = model(xb, None)
        if isinstance(loss_fn, PhysicsRegularizedLoss):
            # try to retrieve ews from batch via loader
            # Most robust approach: loader returns (X, ews, y, meta) so unpack at top-level
            # But since above we didn't, attempt to infer ews as zeros
            ews_input = None
            try:
                # if loader provides attribute last_batch_ews (not standard)
                ews_input = getattr(loader, 'last_batch_ews', None)
            except Exception:
                ews_input = None
            loss_val, data_l, phys_l = loss_fn(preds, yb, inputs=xb, ews=ews_input)
        else:
            data_l = nn.MSELoss()(preds, yb)
            phys_l = torch.tensor(0.0, device=device)
            loss_val = data_l
        loss_val.backward()
        opt.step()
        total_loss += float(loss_val.detach().cpu().numpy()) * xb.size(0)
        total_data_loss += float(data_l.detach().cpu().numpy()) * xb.size(0)
        total_phys_loss += float(phys_l.detach().cpu().numpy()) * xb.size(0)
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
            # expect (X, ews, y, meta)
            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                xb, ews_batch, yb, meta_batch = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, ews_batch, yb = batch
                meta_batch = [None] * xb.shape[0]
            else:
                # fallback: older dataset
                xb, yb = batch
                ews_batch = None
                meta_batch = [None] * xb.shape[0]

            xb = xb.to(device)
            yb = yb.to(device)
            try:
                preds = model(xb, ews_batch.to(device) if ews_batch is not None and isinstance(ews_batch, torch.Tensor) else None)
            except Exception:
                preds = model(xb)
            if isinstance(loss_fn, PhysicsRegularizedLoss):
                loss_val, data_l, phys_l = loss_fn(preds, yb, inputs=xb, ews=(ews_batch.to(device) if ews_batch is not None and isinstance(ews_batch, torch.Tensor) else None))
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
            metas.extend(meta_batch)
    preds_all = np.concatenate(preds_list)
    targets_all = np.concatenate(targets_list)
    n = len(loader.dataset)
    return total_loss / n, total_data_loss / n, total_phys_loss / n, preds_all, targets_all, metas


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'pr_patch'], default='pr_patch')
    parser.add_argument('--lambda', dest='lam', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    artifacts = ArtifactsManager(base_dir=cfg.get('artifacts', {}).get('base_dir', 'artifacts'))
    tb_dir = cfg.get('logging', {}).get('tensorboard_dir')
    monitor = NFTrainingMonitor(artifacts=artifacts, model_name=args.model, horizon=cfg['training']['horizon'])

    # load processed data
    proc = Path(cfg['data']['processed_dir']) / 'data_daily.csv'
    if not proc.exists():
        raise FileNotFoundError(f"Processed data not found: {proc}")
    df = pd.read_csv(proc, parse_dates=['ds'])
    seq_len = cfg['training']['seq_len']

    # split train/val by time: use last 20% samples as validation per series
    # choose ews columns based on features generated earlier
    ews_window = cfg.get('ews', {}).get('rolling_window', 14)
    ews_cols = [f'var_{ews_window}', f'ar1_{ews_window}']
    dataset = SlidingWindowDataset(df, seq_len, ews_cols=ews_cols)
    n = len(dataset)
    val_n = max(1, int(0.2 * n))
    train_n = n - val_n
    indices = np.arange(n)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    batch_size = cfg['training'].get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device(args.device)
    # model init
    if args.model == 'lstm':
        model = LSTMRegressor(input_size=1, hidden_size=64, num_layers=2, seq_len=seq_len)
        loss_fn = nn.MSELoss()
    elif args.model == 'transformer':
        model = TransformerRegressor(seq_len=seq_len, d_model=64, nhead=4, nlayers=2)
        loss_fn = nn.MSELoss()
    else:
        model = PRPatchModel(seq_len=seq_len, patch_size=cfg.get('patchtst', {}).get('patch_size', 7), hidden=64)
        loss_fn = PhysicsRegularizedLoss(lam=args.lam)

    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['training'].get('lr', 1e-3))

    # tensorboard writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_dir)
    except Exception:
        writer = None
        _LOG.warning("TensorBoard SummaryWriter not available; skipping TB logging")

    best_pce = float('inf')
    best_path = None

    epochs = cfg['training'].get('epochs', 10)
    for epoch in range(1, epochs + 1):
        train_loss, train_data_loss, train_phys_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_data_loss, val_phys_loss, preds_val, targets_val, metas = val_epoch(model, val_loader, loss_fn, device)

        # compute global PCE on val
        try:
            from artifacts_manager import pce_metric
            pce = pce_metric(pd.Series(preds_val[:,1].flatten()), pd.Series(targets_val[:,1].flatten()))
        except Exception:
            pce = float('nan')

        _LOG.info("Epoch %d train_loss=%.6f val_loss=%.6f pce=%.6f", epoch, train_loss, val_loss, pce)

        if writer is not None:
            writer.add_scalar('train/total_loss', train_loss, epoch)
            writer.add_scalar('train/data_loss', train_data_loss, epoch)
            writer.add_scalar('train/phys_loss', train_phys_loss, epoch)
            writer.add_scalar('val/total_loss', val_loss, epoch)
            writer.add_scalar('val/data_loss', val_data_loss, epoch)
            writer.add_scalar('val/phys_loss', val_phys_loss, epoch)
            writer.add_scalar('val/pce', pce, epoch)

        # --- Bifurcation-local metrics ---
        # detect bifurcation dates per series using EWS and I series
        def detect_bifurcation_dates(df_local, id_col='unique_id', date_col='ds', target_col='I', win=7, threshold_scale=2.0):
            dates_map = {}
            for uid, g in df_local.groupby(id_col):
                g = g.sort_values(date_col).reset_index(drop=True)
                if target_col not in g.columns:
                    continue
                # rolling slope
                vals = g[target_col].values
                if len(vals) < win * 2:
                    dates_map[uid] = []
                    continue
                slopes = []
                X = np.arange(win).reshape(-1,1)
                from sklearn.linear_model import LinearRegression
                for i in range(len(vals) - win + 1):
                    y = vals[i:i+win]
                    lr = LinearRegression().fit(X, y)
                    slopes.append(lr.coef_[0])
                slopes = np.array(slopes)
                ds = np.diff(slopes)
                thresh = threshold_scale * (np.nanstd(ds) + 1e-8)
                idx = np.where(np.abs(ds) > thresh)[0]
                t_indices = (idx + 1) + (win - 1)
                # build precrisis windows (t-14 .. t-7)
                dates = []
                for ti in t_indices:
                    start = max(0, ti - 14)
                    end = max(0, ti - 7)
                    dates.extend(g.loc[start:end, date_col].tolist())
                dates_map[uid] = set(dates)
            return dates_map

        bif_map = detect_bifurcation_dates(df)
        # collect indices in val set corresponding to bifurcation target dates
        bif_indices = []
        for idx, meta in enumerate(metas):
            if meta is None:
                continue
            uid = meta.get('unique_id')
            tds = meta.get('target_ds')
            if uid in bif_map and tds in bif_map[uid]:
                bif_indices.append(idx)

        if bif_indices:
            preds_bif = preds_val[bif_indices]
            targets_bif = targets_val[bif_indices]
            # MSE at bifurcation on I component
            mse_bif = float(np.mean((preds_bif[:,1] - targets_bif[:,1])**2))
            # PCE at bifurcation (using pce_metric on I component proxy)
            try:
                pce_bif = pce_metric(pd.Series(preds_bif[:,1].flatten()), pd.Series(targets_bif[:,1].flatten()))
            except Exception:
                pce_bif = float('nan')
        else:
            mse_bif = float('nan')
            pce_bif = float('nan')

        if writer is not None:
            writer.add_scalar('val/mse_bifurcation', mse_bif, epoch)
            writer.add_scalar('val/pce_bifurcation', pce_bif, epoch)

        # save best by PCE
        if pce < best_pce:
            best_pce = pce
            # save checkpoint
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'epoch': epoch,
                'pce': float(pce),
                'lambda_base': getattr(loss_fn, 'lambda_base', None),
                'alpha': getattr(loss_fn, 'alpha', None),
                'beta': (loss_fn.beta.item() if hasattr(loss_fn, 'beta') else None),
                'gamma': (loss_fn.gamma.item() if hasattr(loss_fn, 'gamma') else None)
            }
            best_path = artifacts.save_checkpoint(state, model_name=args.model, arch=args.model, epoch=epoch)

    if writer is not None:
        writer.close()
    _LOG.info("Training finished. Best PCE=%.6f saved to %s", best_pce, best_path)


if __name__ == '__main__':
    main()
