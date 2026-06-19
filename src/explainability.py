"""
Explainability utilities: SHAP and LIME wrappers for time-series patch models.

Features:
- `explain_shap_prpatch` : run Kernel SHAP for PRPatchModel with patch -> lag aggregation
- `explain_lime_instance` : run LIME Tabular explainer on lag-features for a single instance
- `save_plot` : helper to persist matplotlib figures under artifacts/plots

The module saves artifacts into `artifacts/plots/` and names files by target and timestamp.
"""
from __future__ import annotations

from typing import Optional, Tuple, List
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

from src.models.pr_patch import PRPatchModel

ARTIFACTS_DIR = Path('artifacts')
PLOTS_DIR = ARTIFACTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def explain_shap_prpatch(
    model: torch.nn.Module,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    seq_len: int,
    patch_size: int,
    target_name: str,
    nsamples: int = 200,
    device: str = 'cpu'
) -> Tuple[np.ndarray, shap.Explainer]:
    """Run Kernel SHAP on `model` treating lags as features. Aggregates patch attributions back to lags.

    X_background/X_instance shapes: (n_bg, seq_len) and (n_inst, seq_len)
    Returns shap_values aggregated per-lag and the explainer object.
    Saves a summary plot into artifacts/plots/.
    """
    # wrapper predict function
    def predict_fn(x2d: np.ndarray):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(x2d.reshape(-1, seq_len), dtype=torch.float32, device=device)
            y = model(x)
            return y.cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_vals = explainer.shap_values(X_instance, nsamples=nsamples)
    shap_arr = np.asarray(shap_vals)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[0]

    # aggregate patch-level to lags if needed
    if shap_arr.shape[1] != seq_len:
        # assume one value per patch
        n_patches = seq_len // patch_size
        if shap_arr.shape[1] == n_patches:
            # distribute uniformly
            lags = np.zeros((shap_arr.shape[0], seq_len))
            for p in range(n_patches):
                start = p * patch_size
                end = start + patch_size
                lags[:, start:end] += (shap_arr[:, p:p+1] / patch_size)
            shap_arr = lags

    # plot summary
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(seq_len), mean_abs)
    ax.set_title(f'SHAP mean |importance| — {target_name}')
    ax.set_xlabel('lag index (0 most recent -> seq_len-1 farthest)')
    timestamp = int(time.time())
    out_path = PLOTS_DIR / f'shap_summary_{target_name}_{timestamp}.png'
    _savefig(fig, out_path)
    return shap_arr, explainer


def explain_lime_instance(
    predict_fn,
    X_train: np.ndarray,
    instance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_name: str = 'target',
    num_features: int = 10,
    save: bool = True
):
    """Run LIME Tabular explainer for a single instance (lags as features).
    Saves textual explanation and a small bar plot to artifacts.
    """
    scaler = None
    explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, mode='regression')
    exp = explainer.explain_instance(instance, predict_fn, num_features=num_features)
    txt = exp.as_list()

    # plot the local feature weights
    names = [f[0] for f in txt]
    vals = [f[1] for f in txt]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(names[::-1], vals[::-1])
    ax.set_title(f'LIME local explanation — {target_name}')
    timestamp = int(time.time())
    out_plot = PLOTS_DIR / f'lime_local_{target_name}_{timestamp}.png'
    if save:
        _savefig(fig, out_plot)
    return exp, out_plot


def compare_attention_focus(model_attention_weights: np.ndarray, baseline_attention: np.ndarray, target_name: str):
    """Simple visual comparison of attention weight distributions.
    Expects arrays of shape (n_tokens,) or (n_heads, n_tokens).
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    # reduce heads by mean if necessary
    ma = model_attention_weights.mean(axis=0) if model_attention_weights.ndim == 2 else model_attention_weights
    ba = baseline_attention.mean(axis=0) if baseline_attention.ndim == 2 else baseline_attention
    ax.plot(ma, label='PRPatch attention')
    ax.plot(ba, label='Baseline attention')
    ax.legend()
    ax.set_title(f'Attention focus comparison — {target_name}')
    out = PLOTS_DIR / f'attn_compare_{target_name}_{int(time.time())}.png'
    _savefig(fig, out)
    return out


if __name__ == '__main__':
    print('Use functions explain_shap_prpatch and explain_lime_instance from this module.')
