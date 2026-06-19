"""
Program.py

Ready-to-run script that demonstrates SHAP and LIME explanations
for two time-series models:
 - a step-wise Transformer (operates on timesteps / tokens)
 - a patch-based PR-Patch (PatchTST variant with physics regularizer)

The script contains:
 - utilities to build lag matrices and detect breakpoints (trend breaks)
 - lightweight example PyTorch models (Transformer-like and Patch-like)
 - wrapper functions adapting model inputs/outputs for `shap` and `lime`
 - SHAP and LIME explainers with mapping from patches -> original lags
 - visualizations: SHAP summary/force plots and LIME local explanation

Notes:
 - Replace the example model loading functions with your real trained
   Transformer and PR-Patch checkpoints (trained with physics loss).
 - For PR-Patch, we use a predict wrapper that accepts a flat lag vector,
   reconstructs patches, runs the model and returns scalar forecasts.

Dependencies: torch, numpy, pandas, shap, lime, matplotlib, sklearn

Author: Copilot-style assistant (adapt and integrate into your notebook)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src.config import load_config


# ------------------------ Data utilities ------------------------
def create_lag_matrix(series, window):
    """Return matrix X of shape (n_samples, window) where row i contains
    lags [t-window, ..., t-1] for target at time t.
    """
    series = np.asarray(series).astype(float)
    n = len(series)
    X = np.lib.stride_tricks.sliding_window_view(series, window + 1)
    # each row is [t-window, ..., t]
    Xlags = X[:, :-1]
    y = X[:, -1]
    return Xlags, y


def detect_breakpoints(series, win=7, slope_change_threshold=2.0):
    """Detect candidate breakpoints where trend derivative changes.

    Method: compute slope of rolling windows (linear regression) and mark
    times where the change in slope exceeds threshold*std.
    Returns indices (relative to series) corresponding to target times t
    (i.e., the last point of the window).
    """
    series = np.asarray(series).astype(float)
    n = len(series)
    if n < win * 3:
        return []
    slopes = []
    X = np.arange(win).reshape(-1, 1)
    for i in range(n - win + 1):
        y = series[i : i + win]
        lr = LinearRegression().fit(X, y)
        slopes.append(lr.coef_[0])
    slopes = np.array(slopes)
    ds = np.diff(slopes)
    thresh = slope_change_threshold * np.nanstd(ds)
    idx = np.where(np.abs(ds) > thresh)[0]
    # convert slope-change index to series index of the prediction time t
    # slope at window i is for window ending at i+win-1, diff at k is between
    # windows k and k+1, so assign target at (k+1)+win-1
    t_indices = (idx + 1) + (win - 1)
    return np.unique(t_indices).tolist()


# ------------------------ Example models ------------------------
class SmallTransformer(nn.Module):
    """Small transformer encoder model mapping sequence of lags -> scalar"""

    def __init__(self, d_model=32, nhead=4, nlayers=2, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.out = nn.Linear(d_model * seq_len, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        b, s = x.shape
        assert s == self.seq_len
        x = x.unsqueeze(-1)
        h = self.input_proj(x)  # (b, s, d)
        # Transformer expects (s, b, d)
        h = h.permute(1, 0, 2)
        h = self.transformer(h)
        h = h.permute(1, 0, 2).reshape(b, -1)
        return self.out(h).squeeze(-1)


class PRPatchDummy(nn.Module):
    """Patch-based regressor. In real PR-Patch the model is trained with a
    physics regularizer. Here we implement a patching pipeline and a small
    network that consumes patches.
    """

    def __init__(self, seq_len=30, patch_size=5, in_channels=1, hidden=64):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.encoder = nn.Sequential(
            nn.Linear(patch_size * in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.out = nn.Linear((hidden // 2) * self.n_patches, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        b, s = x.shape
        assert s == self.seq_len
        patches = x.view(b, self.n_patches, self.patch_size)
        enc = []
        for p in range(self.n_patches):
            enc.append(self.encoder(patches[:, p, :]))
        enc = torch.cat(enc, dim=1)
        return self.out(enc).squeeze(-1)


# ------------------------ Wrappers for SHAP/LIME ------------------------
def to_torch(x, device="cpu"):
    t = torch.tensor(x, dtype=torch.float32, device=device)
    return t


def transformer_predict_fn(model, X_np, device="cpu"):
    """Predict function usable by SHAP/LIME for step-wise Transformer.
    Expects X_np shape (n_samples, seq_len).
    Returns (n_samples,) predictions.
    """
    model.eval()
    with torch.no_grad():
        x = to_torch(X_np, device)
        y = model(x)
        return y.cpu().numpy()


def prpatch_predict_fn_factory(model, seq_len, patch_size, device="cpu"):
    """Return a predict_fn(x2d) where x2d shape is (n_samples, seq_len).

    For KernelExplainer, inputs will be 2D arrays where each column is a lag.
    The wrapper reconstructs patches and runs the patch-model.
    """

    def predict_fn(x2d):
        model.eval()
        with torch.no_grad():
            x = to_torch(x2d.reshape(-1, seq_len), device)
            y = model(x)
            return y.cpu().numpy()

    return predict_fn


def aggregate_patch_shap_to_lags(shap_vals, seq_len, patch_size):
    """Aggregate SHAP values produced at patch-level back to per-lag attribution.

    Approach: if shap_vals shape is (n_samples, n_patches), distribute each
    patch's attribution uniformly across its constituent lags. If shap_vals
    are per-feature (seq_len) already, return directly.
    """
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 2 and shap_vals.shape[1] == seq_len:
        return shap_vals
    # assume shape (n_samples, n_patches)
    n_patches = shap_vals.shape[1]
    assert seq_len % patch_size == 0
    lags = np.zeros((shap_vals.shape[0], seq_len))
    for p in range(n_patches):
        start = p * patch_size
        end = start + patch_size
        # distribute uniformly
        lags[:, start:end] += (shap_vals[:, p:p+1] / patch_size)
    return lags


# ------------------------ SHAP explainers ------------------------
def explain_transformer_shap(model, X_background, X_explain, device="cpu"):
    """Explain transformer using GradientExplainer if possible, else Kernel.
    X_background: (n_bg, seq_len)
    X_explain: (n_instances, seq_len)
    Returns shap_values and explainer object.
    """
    try:
        # try GradientExplainer (fast, uses model gradients)
        model.to(device)
        f = lambda x: transformer_predict_fn(model, x, device)
        explainer = shap.GradientExplainer((model, model.input_proj), to_torch(X_background, device))
        shap_values = explainer.shap_values(to_torch(X_explain, device))
        return shap_values, explainer
    except Exception:
        # fallback to KernelExplainer (model-agnostic)
        f = lambda x: transformer_predict_fn(model, x, device)
        explainer = shap.KernelExplainer(f, X_background)
        shap_values = explainer.shap_values(X_explain, nsamples=200)
        return shap_values, explainer


def explain_prpatch_shap(model, seq_len, patch_size, X_background, X_explain, device="cpu"):
    """Use KernelExplainer for patch model. We expose patches as features
    (one feature per patch), then aggregate back to per-lag importance.
    Here we define a predict function that accepts full lag vectors.
    """
    predict_fn = prpatch_predict_fn_factory(model, seq_len, patch_size, device)
    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_vals = explainer.shap_values(X_explain, nsamples=200)
    # shap_vals may be a list for multiclass; ensure array
    shap_vals = np.asarray(shap_vals)
    # If KernelExplainer returned per-lag shap (seq_len), we are done.
    if shap_vals.ndim == 2 and shap_vals.shape[1] == seq_len:
        return shap_vals, explainer
    # If Kernel gave values per feature and features are patches, aggregate
    # Here we assume X_background columns correspond to seq_len features (not patches)
    # If you used patch-level features, adapt accordingly.
    # For simplicity, try to detect patch-level shape
    if shap_vals.ndim == 2 and shap_vals.shape[1] == (seq_len // patch_size):
        shap_per_lag = aggregate_patch_shap_to_lags(shap_vals, seq_len, patch_size)
        return shap_per_lag, explainer
    # fallback: try to reshape
    return shap_vals, explainer


# ------------------------ LIME explainer ------------------------
def explain_lime(model_predict_fn, X_train, instance, feature_names=None, num_features=10):
    """LIME via LimeTabular for tabular features representing lags.
    model_predict_fn: function taking (n_samples, seq_len) -> predictions
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_scaled, feature_names=feature_names, mode="regression"
    )

    exp = explainer.explain_instance(scaler.transform(instance.reshape(1, -1))[0],
                                      lambda x: model_predict_fn(scaler.inverse_transform(x)),
                                      num_features=num_features)
    return exp


# ------------------------ Plotting helpers ------------------------
def plot_shap_summary(shap_vals, feature_names, title="SHAP summary"):
    plt.figure(figsize=(10, 4))
    # shap.summary_plot handles numpy arrays but may try to show JS; use matplotlib summary
    try:
        shap.summary_plot(shap_vals, features=None, feature_names=feature_names, show=True)
    except Exception:
        # fallback: show mean absolute importance bar
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        idx = np.argsort(mean_abs)[::-1]
        plt.bar(np.array(feature_names)[idx], mean_abs[idx])
        plt.xticks(rotation=90)
        plt.title(title)
        plt.tight_layout()


def plot_shap_force(shap_vals_single, X_instance, feature_names, matplotlib_fallback=True):
    """Plot force plot for a single instance. If JS backend unavailable,
    draw horizontal bar chart of contributions.
    """
    try:
        shap.force_plot(shap_vals_single.base_values, shap_vals_single.values, X_instance,
                        feature_names=feature_names, matplotlib=matplotlib_fallback)
    except Exception:
        vals = shap_vals_single.values
        plt.figure(figsize=(8, 4))
        idx = np.argsort(vals)
        plt.barh(np.array(feature_names)[idx], vals[idx])
        plt.title("Lags contribution (force-style)")
        plt.tight_layout()


def plot_compare_importance(imp_a, imp_b, labels, title_a="Model A", title_b="Model B"):
    """Compare aggregated importances (arrays of shape (seq_len,))."""
    seq_len = len(imp_a)
    x = np.arange(seq_len)
    width = 0.35
    plt.figure(figsize=(12, 4))
    plt.bar(x - width / 2, imp_a, width, label=title_a)
    plt.bar(x + width / 2, imp_b, width, label=title_b)
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.title("Lag importance comparison")
    plt.tight_layout()


# ------------------------ Example usage / demo ------------------------
def demo_synthetic_run():
    """Demonstration generating synthetic epidemic-like time series,
    creating two small models (Transformer & PR-Patch dummy), then running
    SHAP and LIME explanations at detected breakpoints.
    """
    # Generate synthetic series with a change in trend (wave)
    np.random.seed(0)
    t = np.arange(300)
    series = 0.05 * t + 5 * np.sin(0.2 * t) + np.random.normal(scale=1.0, size=len(t))
    # inject a surge (epidemic wave)
    series[150:180] += np.linspace(0, 20, 30)

    seq_len = 30
    patch_size = 5
    # create lag matrix
    Xlags, y = create_lag_matrix(series, seq_len)
    feature_names = [f"t-{seq_len - i}" for i in range(seq_len, 0, -1)]

    # Detect breakpoints
    bps = detect_breakpoints(series, win=7, slope_change_threshold=2.0)
    print("Detected breakpoints (indices):", bps[:10])
    if len(bps) == 0:
        print("No strong breakpoints detected; using point 160 for demo")
        bps = [160]

    # Build models
    device = "cpu"
    tr_model = SmallTransformer(d_model=32, nhead=4, nlayers=2, seq_len=seq_len)
    pr_model = PRPatchDummy(seq_len=seq_len, patch_size=patch_size)

    # Choose background dataset for SHAP (a subset)
    bg_idx = np.random.choice(len(Xlags), size=min(100, len(Xlags)), replace=False)
    X_bg = Xlags[bg_idx]

    # Choose instances to explain: those with detected breakpoint as target time
    instances = []
    for bp in bps:
        # target at time t corresponds to row index t - seq_len
        row_idx = bp - seq_len
        if 0 <= row_idx < len(Xlags):
            instances.append(row_idx)
    if len(instances) == 0:
        instances = [len(Xlags) - 1]

    X_explain = Xlags[instances]

    # SHAP Transformer
    print("Explaining Transformer via SHAP (may fallback to KernelExplainer)...")
    shap_vals_tr, expl_tr = explain_transformer_shap(tr_model, X_bg, X_explain, device=device)
    # shap_vals_tr may be a list (for some explainer formats)
    shap_arr_tr = np.asarray(shap_vals_tr)
    if shap_arr_tr.ndim == 3:
        shap_arr_tr = shap_arr_tr[0]

    plot_shap_summary(shap_arr_tr, feature_names, title="Transformer SHAP summary")

    # SHAP PR-Patch
    print("Explaining PR-Patch via SHAP KernelExplainer...")
    shap_vals_pr, expl_pr = explain_prpatch_shap(pr_model, seq_len, patch_size, X_bg, X_explain, device=device)
    shap_arr_pr = np.asarray(shap_vals_pr)
    if shap_arr_pr.ndim == 3:
        shap_arr_pr = shap_arr_pr[0]

    # If returned shap is patch-level, aggregate
    if shap_arr_pr.shape[1] != seq_len and shap_arr_pr.shape[1] == (seq_len // patch_size):
        shap_arr_pr = aggregate_patch_shap_to_lags(shap_arr_pr, seq_len, patch_size)

    plot_shap_summary(shap_arr_pr, feature_names, title="PR-Patch SHAP summary (aggregated)")

    # Compare aggregated importance (mean abs)
    imp_tr = np.mean(np.abs(shap_arr_tr), axis=0)
    imp_pr = np.mean(np.abs(shap_arr_pr), axis=0)
    labels = [f"t-{i}" for i in range(1, seq_len + 1)]
    plot_compare_importance(imp_tr, imp_pr, labels, title_a="Transformer", title_b="PR-Patch")

    # LIME for a single breakpoint instance
    inst = X_explain[0]
    print("Running LIME for Transformer (local explanation)...")
    tr_predict = lambda x: transformer_predict_fn(tr_model, x, device)
    lime_exp_tr = explain_lime(tr_predict, X_bg, inst, feature_names=feature_names, num_features=8)
    print("Transformer LIME explanation:")
    print(lime_exp_tr.as_list())

    print("Running LIME for PR-Patch (local explanation)...")
    pr_predict = prpatch_predict_fn_factory(pr_model, seq_len, patch_size, device)
    lime_exp_pr = explain_lime(pr_predict, X_bg, inst, feature_names=feature_names, num_features=8)
    print("PR-Patch LIME explanation:")
    print(lime_exp_pr.as_list())

    plt.show()


def demo_run_on_dataframe(df: pd.DataFrame, series_col: str, seq_len: int = 30, patch_size: int = 5, save_prefix: str | None = None):
    """Run the same explanation pipeline on a provided dataframe column.
    Saves plots if `save_prefix` is provided.
    """
    if series_col not in df.columns:
        print(f"Column {series_col} not found in dataframe; skipping.")
        return
    series = df[series_col].fillna(method='ffill').fillna(0).values

    Xlags, y = create_lag_matrix(series, seq_len)
    feature_names = [f"t-{seq_len - i}" for i in range(seq_len, 0, -1)]

    bps = detect_breakpoints(series, win=7, slope_change_threshold=2.0)
    if len(bps) == 0:
        bps = [len(series) - 40]
    instances = []
    for bp in bps:
        row_idx = bp - seq_len
        if 0 <= row_idx < len(Xlags):
            instances.append(row_idx)
    if len(instances) == 0:
        instances = [len(Xlags) - 1]
    X_explain = Xlags[instances]

    device = "cpu"
    tr_model = SmallTransformer(d_model=32, nhead=4, nlayers=2, seq_len=seq_len)
    pr_model = PRPatchDummy(seq_len=seq_len, patch_size=patch_size)

    bg_idx = np.random.choice(len(Xlags), size=min(100, len(Xlags)), replace=False)
    X_bg = Xlags[bg_idx]

    print(f"Running SHAP for {series_col} (Transformer)...")
    shap_vals_tr, expl_tr = explain_transformer_shap(tr_model, X_bg, X_explain, device=device)
    shap_arr_tr = np.asarray(shap_vals_tr)
    if shap_arr_tr.ndim == 3:
        shap_arr_tr = shap_arr_tr[0]
    plot_shap_summary(shap_arr_tr, feature_names, title=f"Transformer SHAP: {series_col}")
    if save_prefix:
        plt.savefig(f"{save_prefix}_{series_col}_transformer_shap.png")

    print(f"Running SHAP for {series_col} (PR-Patch)...")
    shap_vals_pr, expl_pr = explain_prpatch_shap(pr_model, seq_len, patch_size, X_bg, X_explain, device=device)
    shap_arr_pr = np.asarray(shap_vals_pr)
    if shap_arr_pr.ndim == 3:
        shap_arr_pr = shap_arr_pr[0]
    if shap_arr_pr.shape[1] != seq_len and shap_arr_pr.shape[1] == (seq_len // patch_size):
        shap_arr_pr = aggregate_patch_shap_to_lags(shap_arr_pr, seq_len, patch_size)
    plot_shap_summary(shap_arr_pr, feature_names, title=f"PR-Patch SHAP: {series_col}")
    if save_prefix:
        plt.savefig(f"{save_prefix}_{series_col}_prpatch_shap.png")


def run_configured_experiments(cfg_path: str | Path = None):
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "config" / "config.yaml"
    cfg = load_config(cfg_path)
    proc_dir = Path(cfg['data']['processed_dir'])
    # try common processed filenames
    candidates = [proc_dir / 'data_daily_ews.csv', proc_dir / 'data_daily.csv']
    proc_file = None
    for c in candidates:
        if c.exists():
            proc_file = c
            break
    if proc_file is None:
        print("No processed CSV found in data/processed — run data ingestion first.")
        return
    df = pd.read_csv(proc_file, parse_dates=[cfg.get('data', {}).get('date_col', 'ds')])
    targets = cfg.get('data', {}).get('targets', [])
    save_dir = Path(cfg.get('artifacts', {}).get('base_dir', 'artifacts')) / 'explainers'
    save_dir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        if t in df.columns:
            demo_run_on_dataframe(df, t, seq_len=cfg.get('training', {}).get('seq_len', 30),
                                  patch_size=cfg.get('patchtst', {}).get('patch_size', 7),
                                  save_prefix=str(save_dir / 'explainer'))
        else:
            print(f"Target {t} not present in {proc_file}; skipping.")


if __name__ == '__main__':
    # preserve original demo when run directly
    try:
        run_configured_experiments()
    except Exception:
        demo_synthetic_run()


if __name__ == "__main__":
    demo_synthetic_run()
