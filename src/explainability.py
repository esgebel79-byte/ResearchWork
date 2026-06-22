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
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path('artifacts')
PLOTS_DIR = ARTIFACTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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
    fig, ax = plt.subplots(figsize=(10, 4))
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    
    # Пытаемся построить summary_plot, если передан feature_names и размерности совпадают
    try:
        if feature_names and len(feature_names) == shap_arr.shape[1]:
            shap.summary_plot(shap_arr, Xi_flat, feature_names=feature_names, show=False)
        else:
            raise ValueError
    except Exception:
        # Надежный fallback из первой версии
        ax.bar(np.arange(min(seq_len, len(mean_abs))), mean_abs[:seq_len], color='royalblue', edgecolor='black')
        ax.set_title(f'SHAP mean |importance| — {target_name}')
        ax.set_xlabel('lag index (0 most recent -> seq_len-1 farthest)')
        ax.grid(True, linestyle=':', alpha=0.6)

    timestamp = int(time.time())
    fname_summary = f"{out_basename or 'shap'}_{target_name}_summary_{timestamp}.png"
    _savefig(fig, PLOTS_DIR / fname_summary)

    return shap_arr, explainer


def explain_lime_instance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    instance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_name: str = 'target',
    num_features: int = 10,
    save: bool = True,
    out_basename: Optional[str] = None
) -> Tuple[Any, Path]:
    """Локальная LIME-интерпретация для временных рядов (flat features)."""
    X_tr = _ensure_numpy(X_train)
    inst = _ensure_numpy(instance)
    
    Xtrain_flat = X_tr.reshape((X_tr.shape[0], -1))
    inst_flat = inst.reshape((1, -1))

    explainer = lime_tabular.LimeTabularExplainer(Xtrain_flat, feature_names=feature_names, mode='regression')
    
    def predict_flat(x_flat: np.ndarray) -> np.ndarray:
        seq_len = inst.shape[0] if inst.ndim == 2 else inst.shape[1]
        n_features = int(x_flat.shape[1] / seq_len)
        x3 = x_flat.reshape((-1, seq_len, n_features))
        preds = predict_fn(x3)
        preds = _ensure_numpy(preds)
        if preds.ndim == 3:
            return preds.reshape(preds.shape[0], -1)[:, 0]
        elif preds.ndim == 2:
            return preds[:, 0]
        return preds.ravel()

    exp = explainer.explain_instance(inst_flat.ravel(), predict_flat, num_features=num_features)
    fig = exp.as_pyplot_figure()
    
    timestamp = int(time.time())
    fname = f"{out_basename or 'lime'}_{target_name}_{timestamp}_lime.png"
    out_plot = PLOTS_DIR / fname
    
    if save:
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