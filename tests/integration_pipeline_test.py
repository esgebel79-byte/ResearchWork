"""
Integration test for data pipeline -> features -> PRPatchModel attention fusion.
Run after installing dependencies (pandas, numpy, torch).

Usage:
    pip install -r requirements.txt
    python tests/integration_pipeline_test.py
"""
from pathlib import Path
import sys

try:
    import pandas as pd
    import numpy as np
    import torch
except Exception as e:
    print("Missing required packages. Install requirements and re-run.")
    print(e)
    sys.exit(2)

from src.config import load_config
from src.data_ingestion import DataIngestion
from src.features import enrich_features
from src.models.pr_patch import PRPatchModel, PhysicsRegularizedLoss


def main():
    cfg = load_config(Path.cwd() / 'config' / 'config.yaml')
    proc_dir = Path(cfg['data']['processed_dir'])
    proc_dir.mkdir(parents=True, exist_ok=True)
    proc_file = proc_dir / 'data_daily_test.csv'

    dates = pd.date_range('2021-01-01', periods=80, freq='D')
    uid = 'loc_1'
    np.random.seed(1)
    PCR = (np.random.poisson(50, size=len(dates))).cumsum()
    CONF = np.cumsum(np.random.poisson(10, size=len(dates)))
    ACTIVE = np.abs(np.sin(np.linspace(0, 6, len(dates))) * 100) + 5
    BEDS = np.clip(np.random.normal(loc=30, scale=5, size=len(dates)), 0, None)

    df = pd.DataFrame({'ds': dates, 'unique_id': uid,
                       'PCR_TESTS': PCR,
                       'CONFIRMED.sk': CONF,
                       'ACTIVE.sk': ACTIVE,
                       'OCCUPIED_BEDS_CALCULATED': BEDS})
    extra = df.iloc[::10].copy()
    extra['PCR_TESTS'] = extra['PCR_TESTS'] // 2
    df = pd.concat([df, extra], ignore_index=True).sample(frac=1).reset_index(drop=True)

    ing = DataIngestion(cfg)
    daily = ing.to_daily(df)
    print('Daily shape:', daily.shape)
    daily.to_csv(proc_file, index=False)

    out = enrich_features(proc_file, window=7)
    print('EWS cols sample:', [c for c in out.columns if c.startswith('var_') or c.startswith('ar1_')][:8])

    targets = cfg['data'].get('targets', [])
    available_targets = [t for t in targets if t in out.columns]
    print('Available targets for modeling:', available_targets)

    seq_len = 28
    g = out[out['unique_id'] == uid].sort_values('ds').reset_index(drop=True)
    if len(g) < seq_len:
        print('Not enough data for seq_len, adjust seq_len or provide more data')
        sys.exit(3)
    start = len(g) - seq_len
    window_df = g.iloc[start: start + seq_len]

    X = np.stack([window_df[t].values for t in available_targets], axis=1).T
    X = X.T[np.newaxis, ...]

    ews_cols = []
    for t in available_targets:
        ews_cols.append(f'var_{t}_7')
        ews_cols.append(f'ar1_{t}_7')
    ews = window_df[ews_cols].fillna(0).values[np.newaxis, ...]

    print('X shape', X.shape, 'ews shape', ews.shape)

    x_t = torch.tensor(X, dtype=torch.float32)
    ews_t = torch.tensor(ews, dtype=torch.float32)
    model = PRPatchModel(seq_len=seq_len, patch_size=7, hidden=64, n_inputs=len(available_targets), n_ews=ews.shape[2])
    out_pred = model(x_t, ews_t)
    print('Model output shape:', out_pred.shape)

    loss_fn = PhysicsRegularizedLoss(lambda_base=0.05, alpha=0.5)
    y_true = torch.tensor(np.mean(X[:, -1, :], axis=0)[None, :], dtype=torch.float32)
    res = loss_fn(out_pred, y_true, inputs=x_t, ews=ews_t)
    if isinstance(res, tuple):
        total, data_l, phys_l = res
        print('loss total/data/phys:', float(total), float(data_l), float(phys_l))
    else:
        print('loss:', float(res))


if __name__ == '__main__':
    main()
