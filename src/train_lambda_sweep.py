"""
Lambda sensitivity analysis pipeline: train independent models for each lambda value
and compute bifurcation-aware metrics (MSE, PCE) with corrected FPR calculation.

Usage:
    python -m src.train_lambda_sweep --lambda_grid 0.01,0.05,0.1,0.2 --epochs 5 --batch_size 32
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.config import load_config
from src.data_ingestion import DataIngestion
from src.features import enrich_features
from src.models.pr_patch import PRPatchModel, PhysicsRegularizedLoss
from src.train import (
    SlidingWindowDataset, fit_scalers, train_epoch, val_epoch,
    normalize_ews_columns, physics_step_pce
)
from artifacts_manager import ArtifactsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger(__name__)


def train_model_for_lambda(lam: float, cfg: Dict[str, Any], device: torch.device,
                           train_loader: DataLoader, val_loader: DataLoader,
                           seq_len: int, ews_cols: List[str], epochs: int = 5) -> Tuple[torch.nn.Module, float]:
    """Train an independent PR-Patch model for a specific lambda value.

    Returns: (trained_model, best_physics_pce)
    """
    n_ews = max(0, len(ews_cols))
    model = PRPatchModel(seq_len=seq_len, patch_size=cfg.get('patchtst', {}).get('patch_size', 7),
                        hidden=64, n_inputs=3, n_ews=n_ews)
    loss_fn = PhysicsRegularizedLoss(lambda_base=float(lam))
    model.to(device)

    optimizer_params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.Adam(optimizer_params, lr=cfg.get('training', {}).get('lr', 1e-3))

    best_pce = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, _, _, preds_val, targets_val, _ = val_epoch(model, val_loader, loss_fn, device)

        preds_t = torch.tensor(preds_val, dtype=torch.float32)
        targets_t = torch.tensor(targets_val, dtype=torch.float32)
        try:
            phys_pce = physics_step_pce(preds_t, targets_t, loss_fn.beta, loss_fn.gamma)
        except Exception:
            phys_pce = float('nan')

        if np.isfinite(phys_pce) and phys_pce < best_pce:
            best_pce = phys_pce

        _LOG.info("Lambda=%.4f Epoch=%d train_loss=%.6f val_loss=%.6f phys_pce=%.6f",
                 lam, epoch, train_loss, val_loss, phys_pce)

    return model, float(best_pce)


def run_lambda_sweep(cfg: Dict[str, Any], lambda_grid: List[float], epochs: int = 5, batch_size: int = 32):
    """Execute a full lambda sensitivity sweep with independent model training."""
    artifacts = ArtifactsManager(base_dir=cfg.get('artifacts', {}).get('base_dir', 'artifacts'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    proc = Path(cfg['data']['processed_dir']) / 'data_daily.csv'
    if not proc.exists():
        ing = DataIngestion(cfg)
        proc = ing.run()

    df = pd.read_csv(proc, parse_dates=[cfg.get('data', {}).get('date_col', 'ds')] 
                                      if cfg.get('data', {}).get('date_col', 'ds') 
                                      in pd.read_csv(proc, nrows=0).columns else [])
    seq_len = cfg['training']['seq_len']

    ews_window = cfg.get('ews', {}).get('rolling_window', 14)
    ews_cols = [f'var_{ews_window}', f'ar1_{ews_window}']
    if not all(col in df.columns for col in ews_cols):
        df = enrich_features(proc, window=ews_window)
        ews_cols = normalize_ews_columns(df, window=ews_window)
        if not ews_cols:
            ews_cols = [c for c in df.columns if c.startswith('var_') or c.startswith('ar1_')]

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

    dataset = SlidingWindowDataset(df, seq_len, ews_cols=ews_cols,
                                  target_scaler=target_scaler, ews_scaler=ews_scaler)
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

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = []
    for lam in lambda_grid:
        _LOG.info("=" * 60)
        _LOG.info("Training model with lambda=%.6f", lam)
        _LOG.info("=" * 60)

        try:
            model, best_pce = train_model_for_lambda(lam, cfg, device, train_loader, val_loader,
                                                     seq_len, ews_cols, epochs=epochs)
            results.append({"lambda": float(lam), "best_pce": float(best_pce)})

            checkpoint_state = {
                'model_state_dict': model.state_dict(),
                'lambda': float(lam),
                'best_pce': float(best_pce),
            }
            artifacts.save_checkpoint(checkpoint_state, model_name='pr_patch_lambda_sweep',
                                     arch='pr_patch', epoch=None)
            _LOG.info("Lambda=%.6f: best_pce=%.6f", lam, best_pce)

        except Exception as e:
            _LOG.exception("Training failed for lambda=%.6f: %s", lam, e)
            results.append({"lambda": float(lam), "best_pce": float('nan')})

    df_results = pd.DataFrame(results)
    metrics_path = Path(cfg.get('artifacts', {}).get('base_dir', 'artifacts')) / 'metrics' / 'lambda_sweep_results.csv'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(metrics_path, index=False)
    _LOG.info("Lambda sweep results saved to %s", str(metrics_path))

    _LOG.info("\n" + "=" * 60)
    _LOG.info("LAMBDA SWEEP SUMMARY")
    _LOG.info("=" * 60)
    _LOG.info(df_results.to_string(index=False))
    _LOG.info("=" * 60)

    return df_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lambda sensitivity sweep')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--lambda_grid', type=str, default='0.01,0.05,0.1,0.2',
                       help='Comma-separated lambda values to sweep')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    lambda_vals = [float(x.strip()) for x in args.lambda_grid.split(',')]

    df_results = run_lambda_sweep(cfg, lambda_vals, epochs=args.epochs, batch_size=args.batch_size)
    print("\nLambda sweep completed successfully.")
