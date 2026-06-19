"""Расчёт признаков (EWS) для раннего предупреждения временных рядов.

Модуль содержит функции:
- rolling_variance: скользящая дисперсия
- rolling_ar1: оценка коэффициента AR(1) на скользящем окне

Используйте их для обогащения `data/processed/data_daily.csv` перед обучением.
"""
from __future__ import annotations

from typing import Iterable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import logging

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def rolling_variance(series: pd.Series, window: int) -> pd.Series:
    """Вычислить скользящую дисперсию для `series` с окном `window`.

    Возвращает pd.Series той же длины, первые значения NaN.
    """
    return series.rolling(window=window, min_periods=1).var()


def rolling_ar1(series: pd.Series, window: int) -> pd.Series:
    """Оценка коэффициента AR(1) на скользящем окне.

    Для каждого окна строится линейная регрессия y_t ~ y_{t-1} и возвращается
    коэффициент при лаге-1. Метод устойчив к пропускам (dropna внутри окна).
    """
    vals = []
    arr = series.values
    n = len(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        window_vals = arr[start:i + 1]
        # need at least 2 points to estimate AR(1)
        if len(window_vals) < 2 or np.all(np.isnan(window_vals)):
            vals.append(np.nan)
            continue
        # create lagged pairs
        s = pd.Series(window_vals).dropna()
        if len(s) < 2:
            vals.append(np.nan)
            continue
        y = s.values[1:]
        x = s.values[:-1]
        # solve linear reg coef = (x^T x)^{-1} x^T y
        try:
            coef = np.dot(x, y) / (np.dot(x, x) + 1e-8)
            vals.append(float(coef))
        except Exception:
            vals.append(np.nan)
    return pd.Series(vals, index=series.index)


def enrich_features(processed_csv: str | Path, window: int = 14, id_col: str = 'unique_id', date_col: str = 'ds', target_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Загрузить processed CSV и добавить колонки EWS: `var_{target}_{window}` и `ar1_{target}_{window}`.

    Расчитывает EWS отдельно для каждого целевого ряда в `target_cols`.
    Если `target_cols` не указан, попытается найти конфигированные `targets`.

    Возвращает DataFrame и сохраняет файл в том же каталоге с суффиксом `_ews`.
    """
    p = Path(processed_csv)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=[date_col])
    cfg_targets = None
    try:
        from src.config import load_config
        cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
        cfg_targets = cfg.get('data', {}).get('targets', None)
    except Exception:
        cfg_targets = None

    if target_cols is None:
        if cfg_targets:
            target_cols = [t for t in cfg_targets if t in df.columns]
        else:
            # fallback: use 'y' if present, otherwise numeric columns excluding id/date
            target_cols = ['y'] if 'y' in df.columns else [c for c in df.select_dtypes(include='number').columns if c not in (id_col, date_col)]

    out_frames = []
    for uid, g in df.groupby(id_col):
        g = g.sort_values(date_col).reset_index(drop=True)
        for t in target_cols:
            if t not in g.columns:
                continue
            g[f'var_{t}_{window}'] = rolling_variance(g[t], window)
            g[f'ar1_{t}_{window}'] = rolling_ar1(g[t], window)
        g[id_col] = uid
        out_frames.append(g)
    out = pd.concat(out_frames, ignore_index=True)
    out_path = p.parent / (p.stem + f"_ews_{window}.csv")
    out.to_csv(out_path, index=False)
    _LOG.info("EWS features saved to %s", str(out_path))
    return out


if __name__ == '__main__':
    from src.config import load_config
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    proc = Path(cfg['data']['processed_dir']) / 'data_daily.csv'
    enrich_features(proc, window=cfg.get('ews', {}).get('rolling_window', 14))
