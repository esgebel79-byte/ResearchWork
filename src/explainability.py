"""
Explainability utilities: SHAP and LIME wrappers for time-series patch models.

Features:
- `explain_shap_prpatch` : Kernel SHAP для PRPatchModel с агрегацией patch -> lag (Восстановлено!)
- `explain_lime_instance` : LIME Tabular explainer для локального анализа точек бифуркации
- `compare_attention_focus` : Сравнение распределения весов внимания (Восстановлено!)
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Callable, Any, Sequence
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import logging

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

try:
    from lime import lime_tabular
except ImportError:  # pragma: no cover - optional dependency
    lime_tabular = None

import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path('artifacts')
PLOTS_DIR = ARTIFACTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def _ensure_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def explain_shap_prpatch(
    model_or_predict_fn: Any,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    seq_len: int,
    patch_size: int,
    target_name: str,
    background_size: int = 50,
    nsamples: Any = "auto",
    device: str = 'cpu',
    feature_names: Optional[Sequence[str]] = None,
    out_basename: Optional[str] = None
) -> Tuple[np.ndarray, shap.KernelExplainer]:
    """
    Расчет SHAP-значений с деагрегацией патчей в лаги и оптимизацией фоновой выборки.
    """
    if shap is None:
        raise ImportError("SHAP is required for explain_shap_prpatch. Install the project dependencies with pip install -r requirements.txt.")

    Xb = _ensure_numpy(X_background)
    Xi = _ensure_numpy(X_instance)

    # 1. Оптимизация фоновой выборки
    bg_num = min(background_size, Xb.shape[0])
    if Xb.shape[0] > bg_num:
        indices = np.random.choice(Xb.shape[0], size=bg_num, replace=False)
        Xb_sub = Xb[indices]
    else:
        Xb_sub = Xb

    n_features = Xi.shape[2] if Xi.ndim == 3 else 1

    # 2. Универсальная обертка для предикт-функции
    def predict_fn_internal(x_2d: np.ndarray) -> np.ndarray:
        if callable(model_or_predict_fn):
            x3 = x_2d.reshape((-1, seq_len, n_features))
            preds = model_or_predict_fn(x3)
        else:
            model_or_predict_fn.eval()
            with torch.no_grad():
                if n_features > 1:
                    x_tensor = torch.tensor(x_2d.reshape(-1, seq_len, n_features), dtype=torch.float32, device=device)
                else:
                    x_tensor = torch.tensor(x_2d.reshape(-1, seq_len), dtype=torch.float32, device=device)
                preds = model_or_predict_fn(x_tensor)
        
        preds = _ensure_numpy(preds)
        if preds.ndim == 3:
            return preds.reshape(preds.shape[0], -1)[:, 0]
        elif preds.ndim == 2:
            return preds[:, 0]
        return preds.ravel()

    Xb_flat = Xb_sub.reshape((Xb_sub.shape[0], -1))
    Xi_flat = Xi.reshape((Xi.shape[0], -1))

    explainer = shap.KernelExplainer(predict_fn_internal, Xb_flat, link="identity")
    shap_vals = explainer.shap_values(Xi_flat, nsamples=nsamples)
    shap_arr = np.asarray(shap_vals)
    
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[0]

    # 3. ВОССТАНОВЛЕНО: Агрегация патч-уровня обратно к индивидуальным временным лагам
    # Проверяем, совпадает ли ширина flat-массива SHAP с ожидаемым (seq_len * n_features)
    if shap_arr.shape[1] != seq_len * n_features:
        n_patches = seq_len // patch_size
        if shap_arr.shape[1] == n_patches * n_features:
            lags_arr = np.zeros((shap_arr.shape[0], seq_len * n_features))
            for f in range(n_features):
                f_offset_patch = f * n_patches
                f_offset_lag = f * seq_len
                for p in range(n_patches):
                    start = f_offset_lag + (p * patch_size)
                    end = start + patch_size
                    lags_arr[:, start:end] += (shap_arr[:, f_offset_patch + p : f_offset_patch + p + 1] / patch_size)
            shap_arr = lags_arr

    # 4. Построение графиков
    # Sanity checks: ensure shap_arr is usable
    try:
        sa = np.asarray(shap_arr, dtype=np.float64)
    except Exception:
        logger.warning(f"SHAP array could not be converted to ndarray for {target_name}; skipping plot.")
        return shap_arr, explainer

    if sa.size == 0 or np.isnan(sa).all():
        logger.warning(f"SHAP array is empty or all-NaN for {target_name}; skipping plot.")
        return shap_arr, explainer

    fig, ax = plt.subplots(figsize=(10, 4))
    mean_abs = np.mean(np.abs(sa), axis=0)

    # Пытаемся построить summary_plot, если передан feature_names и размерности совпадают
    try:
        if feature_names and len(feature_names) == sa.shape[1]:
            shap.summary_plot(sa, Xi_flat, feature_names=feature_names, show=False)
        else:
            raise ValueError("feature_names missing or length mismatch; using fallback bar plot")
    except Exception as e:
        logger.warning(f"SHAP summary_plot failed or not suitable: {e}; drawing fallback bar chart.")
        ax.bar(np.arange(min(seq_len, len(mean_abs))), mean_abs[:seq_len], color='royalblue', edgecolor='black')
        ax.set_title(f'SHAP mean |importance| — {target_name}')
        ax.set_xlabel('lag index (0 most recent -> seq_len-1 farthest)')
        ax.grid(True, linestyle=':', alpha=0.6)

    timestamp = int(time.time())
    fname_summary = f"{out_basename or 'shap'}_{target_name}_summary_{timestamp}.png"
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    _savefig(fig, PLOTS_DIR / fname_summary)

    return shap_arr, explainer


def explain_lime_instance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    instance: np.ndarray,
    n_patches: int = 8,
    n_features = 4,
    feature_names: Optional[List[str]] = None,
    target_name: str = 'target',
    num_features: int = 10,
    save: bool = True,
    out_basename: Optional[str] = None
) -> Tuple[Any, Path]:
    
    """Локальная LIME-интерпретация для временных рядов (flat features)."""
    if lime_tabular is None:
        raise ImportError("LIME is required for explain_lime_instance. Install the project dependencies with pip install -r requirements.txt.")

    X_tr = _ensure_numpy(X_train)
    inst = _ensure_numpy(instance)

    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(1, -1)
    if inst.ndim == 1:
        inst = inst.reshape(1, -1)

    # Восстанавливаем исходную форму из X_train: (n_samples, seq_len, n_features)
    if X_tr.ndim == 3:
        n_rows, seq_len, n_features = X_tr.shape
    else:
        seq_len, n_features = X_tr.shape[1], 1 if X_tr.ndim == 1 else 1
        X_tr = X_tr.reshape((X_tr.shape[0], seq_len, n_features))

    expected_flat = seq_len * n_features
    if feature_names is not None and len(feature_names) != expected_flat:
        raise ValueError(
            f"feature_names length mismatch: expected {expected_flat} names for shape ({seq_len}, {n_features}), "
            f"got {len(feature_names)}. Use one name per flattened feature, e.g. ['col_lag_0', ...]."
        )

    Xtrain_flat = X_tr.reshape((X_tr.shape[0], -1))
    inst_flat = inst.reshape((1, -1))

    # --- make a copy for explainer, and protect against zero-variance features ---
    Xtrain_for_expl = Xtrain_flat.astype(np.float64).copy()
    inst_for_expl = inst_flat.astype(np.float64).copy()

    # Add tiny jitter to constant columns to avoid singular regressions inside LIME
    col_std = Xtrain_for_expl.std(axis=0)
    const_cols = np.where(col_std == 0)[0]
    if const_cols.size > 0:
        jitter = 1e-8 * (np.random.RandomState(0).randn(*Xtrain_for_expl[:, const_cols].shape))
        Xtrain_for_expl[:, const_cols] += jitter
        inst_for_expl[0, const_cols] += jitter[0] if jitter.shape[0] > 0 else 0.0

    explainer = lime_tabular.LimeTabularExplainer(Xtrain_for_expl, feature_names=feature_names, mode='regression')

    def predict_flat(x_flat: np.ndarray) -> np.ndarray:
        if x_flat.ndim == 1:
            x_flat = x_flat.reshape(1, -1)

        # restore the original shape from X_train's feature layout
        original_shape = (x_flat.shape[0], seq_len, n_features)
        x3 = x_flat.reshape(original_shape)
        preds = predict_fn(x3)
        preds = _ensure_numpy(preds)

        # convert to numpy float64 and preserve (N, M) shape when possible
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds.reshape(preds.shape[0], -1)
        if preds.ndim == 1:
            preds = preds.reshape(preds.shape[0], 1)
        preds = preds.astype(np.float64)

        # Debug: log output shape and sample values
        try:
            logger.debug(f"LIME predict_flat: x_flat.shape={x_flat.shape}, preds.shape={preds.shape}, sample={preds.ravel()[:8]}")
        except Exception:
            pass

        return preds

    exp = explainer.explain_instance(inst_for_expl.ravel(), predict_flat, num_features=num_features)

    # Sanity-check the explanation before plotting
    try:
        pairs = exp.as_list()
    except Exception:
        pairs = None

    if not pairs:
        logger.warning(f"LIME returned empty explanation for {target_name}; skipping LIME plot.")
        return exp, None

    # Validate weights are numeric and not NaN
    weights = [w for (_f, w) in pairs]
    try:
        w_arr = np.asarray(weights, dtype=np.float64)
    except Exception:
        logger.warning(f"LIME explanation weights are non-numeric for {target_name}; skipping LIME plot.")
        return exp, None

    if np.isnan(w_arr).all():
        logger.warning(f"LIME explanation weights are all NaN for {target_name}; skipping LIME plot.")
        return exp, None

    fig = exp.as_pyplot_figure()
    timestamp = int(time.time())
    fname = f"{out_basename or 'lime'}_{target_name}_{timestamp}_lime.png"
    out_plot = PLOTS_DIR / fname

    if save:
        Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
        _savefig(fig, out_plot)
    return exp, out_plot


def compare_attention_focus(model_attention_weights: np.ndarray, baseline_attention: np.ndarray, target_name: str) -> Path:
    """ВОССТАНОВЛЕНО: Сравнение распределения весов внимания в скрытом пространстве модели."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ma = model_attention_weights.mean(axis=0) if model_attention_weights.ndim == 2 else model_attention_weights
    ba = baseline_attention.mean(axis=0) if baseline_attention.ndim == 2 else baseline_attention
    
    ax.plot(ma, label='PRPatch attention', color='crimson', marker='o', markersize=4)
    ax.plot(ba, label='Baseline attention', color='gray', linestyle='--')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f'Attention focus comparison — {target_name}')
    
    out = PLOTS_DIR / f'attn_compare_{target_name}_{int(time.time())}.png'
    _savefig(fig, out)
    return out


if __name__ == '__main__':
    print('Модуль src.explainability полностью восстановлен и готов к работе.')