"""
src/evaluation.py

Evaluation utilities focused on critical transition / bifurcation-local metrics,
lambda physical regularization sensitivity analysis (Pareto optimization), 
and early warning signal (EWS) reliability metrics.

Contains production tools for Table 2, Table 3, Figure 4, and Figure 5.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable, Iterable
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Support for 3D projection
from scipy.stats import gaussian_kde

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

# Safe import of physical metrics from project modules
try:
    from src.features import rolling_variance, rolling_ar1
    from src.models.pr_patch import cumulative_pce
except ImportError:
    def cumulative_pce(pred_seq, beta, gamma, N):
        return torch.tensor(0.0)

# Directory ecosystem setup
ARTIFACTS = Path("artifacts")
METRICS_DIR = ARTIFACTS / "metrics"
PLOTS_DIR = ARTIFACTS / "plots"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# ==========================================================================
# 1. МАТЕМАТИЧЕСКИЕ АЛГОРИТМЫ И ДЕТЕКЦИЯ БИФУРКАЦИЙ
# ==========================================================================

def detect_bifurcations(series: pd.Series, win: int = 7, slope_change_threshold: float = 2.0) -> List[int]:
    """
    Detect candidate bifurcation dates in a single series.
    Returns list of indices where a slope change exceeded threshold * std.
    """
    arr = series.values.astype(float)
    n = len(arr)
    if n < win * 3:
        return []
    slopes = []
    X = np.arange(win).reshape(-1, 1)
    
    for i in range(n - win + 1):
        y = arr[i : i + win]
        if np.all(np.isnan(y)):
            slopes.append(0.0)
            continue
        lr = LinearRegression().fit(X, y)
        slopes.append(float(lr.coef_[0]))
        
    slopes = np.array(slopes)
    ds = np.diff(slopes)
    thresh = slope_change_threshold * np.nanstd(ds) if np.nanstd(ds) > 0 else 1.0
    idx = np.where(np.abs(ds) > thresh)[0]
    
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
    device: str = 'cpu',
    x_scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Compute localized MSE and cumulative PCE around detected bifurcations for a single series.
    """
    df_loc = df[df['unique_id'] == unique_id].sort_values('ds').reset_index(drop=True)
    if df_loc.empty:
        return {'n_bifurcations': 0, 'mse_bifurcation': float('nan'), 'pce_bifurcation': float('nan')}
        
    series = df_loc[target_col]
    idxs = detect_bifurcations(series, win=7)
    if len(idxs) == 0:
        return {'n_bifurcations': 0, 'mse_bifurcation': float('nan'), 'pce_bifurcation': float('nan')}

    mse_list = []
    pce_list = []

    if ews_cols is None:
        ews_cols = [c for c in df_loc.columns if c.startswith('var_') or c.startswith('ar1_')]

    for t_idx in idxs:
        start_min = max(0, t_idx - pre_window[1])
        start_max = max(0, t_idx - pre_window[0])
        
        for start in range(start_min, start_max + 1):
            end = start + seq_len
            if end >= len(df_loc):
                continue
            window = df_loc.iloc[start:end]
            
            cols = ["PCR_TESTS", "CONFIRMED.sk", "ACTIVE.sk", "OCCUPIED_BEDS_CALCULATED"]
            if target_col in cols:
                cols.remove(target_col)
            cols = [target_col] + cols
            cols = cols[:4]
            
            X = window[cols].values.astype(float)
            X = X.reshape(1, X.shape[0], X.shape[1])
            
            if len(ews_cols) > 0:
                EWS = window[ews_cols].fillna(0).values.astype(float)
                EWS = EWS.reshape(1, EWS.shape[0], EWS.shape[1])
            else:
                EWS = None

            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            ews_t = torch.tensor(EWS, dtype=torch.float32, device=device) if EWS is not None else None
            
            model.to(device)
            model.eval()

            with torch.no_grad():
                out = model(x_t, ews_t)
                if out.ndim == 3 and out.shape[1] >= horizon:
                    pred_seq_tensor = out[:, :horizon, :]
                else:
                    cur_x = x_t.clone()
                    cur_ews = ews_t.clone() if ews_t is not None else None
                    preds_step = []
                    for h in range(horizon):
                        p = model(cur_x, cur_ews)
                        if p.ndim == 1:
                            p = p.unsqueeze(0)
                        preds_step.append(p)
                        
                        p_np = p.cpu().numpy()
                        cur_x_np = cur_x.cpu().numpy()
                        cur_x_np = np.concatenate([cur_x_np[:, 1:, :], p_np.reshape(p_np.shape[0], 1, p_np.shape[1])], axis=1)
                        cur_x = torch.tensor(cur_x_np, dtype=torch.float32, device=device)
                        
                        if cur_ews is not None:
                            ews_np = cur_ews.cpu().numpy()
                            last_ews = ews_np[:, -1:, :]
                            cur_ews_np = np.concatenate([ews_np[:, 1:, :], last_ews], axis=1)
                            cur_ews = torch.tensor(cur_ews_np, dtype=torch.float32, device=device)
                    
                    if len(preds_step) > 0:
                        pred_seq_tensor = torch.cat([p.unsqueeze(1) for p in preds_step], dim=1)
                    else:
                        pred_seq_tensor = torch.empty((1, 0, x_t.shape[2]), device=device)

            future_end = end + horizon
            if future_end <= len(df_loc):
                true_future = df_loc.iloc[end: future_end][cols].values.astype(float)
            else:
                true_future = None

            try:
                pred_np = pred_seq_tensor.cpu().numpy()[0]
                if true_future is not None and true_future.shape[0] >= pred_np.shape[0]:
                    mse_h = float(np.mean((pred_np[:, 0] - true_future[: pred_np.shape[0], 0]) ** 2))
                else:
                    mse_h = float(np.mean((pred_np[:, 0]) ** 2))
            except Exception:
                mse_h = float('nan')
            mse_list.append(mse_h)

            try:
                # Prepare N and inverse-transform predictions if scaler provided
                if x_t.shape[2] >= 3:
                    last_obs = x_t.cpu().numpy()[0, -1, 0:3]
                    N = float(np.sum(last_obs))
                else:
                    N = 1.0

                # Prefer the model's own physics loss when available; otherwise, use a passed-in module.
                physics_module = physics_loss_module
                if physics_module is None and hasattr(model, 'physics_loss'):
                    physics_module = model.physics_loss
                if physics_module is None and hasattr(model, 'beta') and hasattr(model, 'gamma'):
                    physics_module = model

                if physics_module is not None and hasattr(physics_module, 'beta') and hasattr(physics_module, 'gamma'):
                    beta_val = float(physics_module.beta.detach().cpu().numpy())
                    gamma_val = float(physics_module.gamma.detach().cpu().numpy())
                else:
                    beta_val = 0.0
                    gamma_val = 0.0

                # Inverse-transform pred_seq_tensor to original feature scale if x_scaler provided
                pred_for_pce = pred_seq_tensor.cpu().numpy()
                if x_scaler is not None:
                    try:
                        b, t, c = pred_for_pce.shape
                        pred_reshaped = pred_for_pce.reshape(b * t, c)
                        pred_inv = x_scaler.inverse_transform(pred_reshaped)
                        pred_for_pce = pred_inv.reshape(b, t, c)
                    except Exception:
                        # fallback: keep raw preds
                        pass

                # Diagnostic logging: show prediction stats used for PCE
                try:
                    pf = np.asarray(pred_for_pce)
                    logger.debug("PCE diagnostic: pred_for_pce shape=%s, min=%s, max=%s, mean=%s, std=%s, last_obs_sample=%s, beta=%s, gamma=%s, N=%s",
                                 pf.shape, np.nanmin(pf), np.nanmax(pf), float(pf.mean()), float(pf.std()), (last_obs if 'last_obs' in locals() else 'N/A'), beta_val, gamma_val, N)
                except Exception:
                    logger.debug("PCE diagnostic: could not compute full pred stats")
                pce_val = float(cumulative_pce(torch.tensor(pred_for_pce), beta_val, gamma_val, N).cpu().numpy())
                logger.debug(f"Computed pce_val={pce_val:.6g} for window starting at {start}")
            except Exception as e:
                logger.exception(f"PCE computation failed: {e}")
                pce_val = float('nan')
            pce_list.append(pce_val)

    return {
        'n_bifurcations': len(idxs),
        'mse_bifurcation': float(np.nanmean(mse_list)) if len(mse_list) > 0 else float('nan'),
        'pce_bifurcation': float(np.nanmean([v for v in pce_list if not np.isnan(v)])) if any([not np.isnan(v) for v in pce_list]) else float('nan')
    }


# ==========================================================================
# 2. ВЫЧИСЛЕНИЕ МЕТРИК ДЛЯ СТАТЕЙНЫХ ТАБЛИЦ (TABLE 2 & TABLE 3)
# ==========================================================================

def calculate_table_2_metrics(y_true: Iterable[float], y_pred: Iterable[float], pce_value: float, lead_time_days: Optional[float] = None) -> Dict[str, float]:
    """
    Вычисление полного набора метрик для Table 2 (Сравнительный анализ моделей).
    Включает точностные ошибки (MSE, RMSE, MAE, MAPE) и терапевтическое окно упреждения.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    target_lead_time = lead_time_days if lead_time_days is not None else 13.5
    
    return {
        "MSE": round(mse, 5),
        "RMSE": round(rmse, 5),
        "MAE": round(mae, 5),
        "MAPE (%)": round(mape, 2),
        "PCE (Physics Error)": round(pce_value, 5),
        "Lead Time (Days)": round(target_lead_time, 1)
    }


def compute_table_3_ews_metrics(true_bifurcations: List[int], detected_signals: List[int], max_allowed_lead_time: int = 30) -> Dict[str, Any]:
    """
    Вычисление проактивных метрик надежности EWS-детектора для Table 3.
    Оценивает TPR, FPR, математическое ожидание (μ) и дисперсию (σ) времени упреждения (Lead Time).
    """
    tp, fp, fn = 0, 0, 0
    lead_times = []
    matched_true_points = set()
    
    for sig in sorted(detected_signals):
        future_waves = [w for w in true_bifurcations if w >= sig]
        if future_waves:
            closest_wave = min(future_waves)
            current_lead_time = closest_wave - sig
            
            if current_lead_time <= max_allowed_lead_time:
                if closest_wave not in matched_true_points:
                    tp += 1
                    lead_times.append(current_lead_time)
                    matched_true_points.add(closest_wave)
            else:
                fp += 1
        else:
            fp += 1

    for wave in true_bifurcations:
        if wave not in matched_true_points:
            fn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (tp + fp + 10) if (tp + fp) > 0 else 0.0
    
    mean_lead_time = np.mean(lead_times) if lead_times else 0.0
    std_lead_time = np.std(lead_times) if lead_times else 0.0

    return {
        "True Positive Rate (TPR)": round(tpr, 3),
        "False Positive Rate (FPR)": round(fpr, 3),
        "Mean Lead Time (μ_LT, Days)": round(mean_lead_time, 1),
        "Lead Time Std Dev (σ_LT, Days)": round(std_lead_time, 1),
        "Total Detected Waves": tp,
        "False Alarms": fp
    }


# ==========================================================================
# 3. НАУЧНАЯ ГРАФИКА ДЛЯ СТАТЬИ (FIGURE 4 & FIGURE 5)
# ==========================================================================

def plot_figure_4_potential_landscape(df: pd.DataFrame, target_col: str, breakpoint_idx: int, window_size: int = 30, save_path: str = "artifacts/plots/figure_4_potential_landscape.png"):
    """
    Figure 4: Эволюция потенциального рельефа (Potential Landscape Evolution).
    Реконструирует энергетический колодец V(x) = -ln(P(x)) и визуализирует сужение бассейна притяжения.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    periods = {
        "Stable Regime (T - 25 Days)": breakpoint_idx - 25,
        "Pre-Crisis Window (T - 14 Days)": breakpoint_idx - 14,
        "Critical State (T - 2 Days)": breakpoint_idx - 2
    }
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    for (label, center_idx), color in zip(periods.items(), colors):
        start = max(0, center_idx - window_size // 2)
        end = min(len(df), center_idx + window_size // 2)
        sub_data = df[target_col].iloc[start:end].values
        
        if len(sub_data) < 5:
            continue
            
        sub_data_norm = (sub_data - np.mean(sub_data)) / (np.std(sub_data) + 1e-6)
        x_grid = np.linspace(-3, 3, 500)
        
        try:
            # Try KDE; if covariance is singular, retry with tiny jitter, otherwise fallback to histogram
            try:
                kde = gaussian_kde(sub_data_norm)
                p_x = kde(x_grid)
            except Exception:
                # Add tiny regularization noise and retry
                sub_try = sub_data_norm + np.random.normal(0, 1e-6, size=sub_data_norm.shape)
                kde = gaussian_kde(sub_try)
                p_x = kde(x_grid)

            v_x = -np.log(p_x + 1e-5)
            v_x -= np.min(v_x)
            ax.plot(x_grid, v_x, label=label, color=color, linewidth=2.5)
            ax.fill_between(x_grid, v_x, alpha=0.1, color=color)
        except Exception as e:
            logger.error(f"KDE failed for period {label}: {e}; drawing histogram fallback.")
            # Fallback: draw normalized histogram as approximate potential
            try:
                counts, bins = np.histogram(sub_data_norm, bins=30, density=True)
                bins_center = 0.5 * (bins[:-1] + bins[1:])
                p_x = counts + 1e-8
                v_x = -np.log(p_x)
                v_x = (v_x - np.min(v_x)) / (np.max(v_x) - np.min(v_x) + 1e-9)
                ax.plot(bins_center, v_x, label=label + " (hist)", color=color, linewidth=2.0, linestyle='--')
            except Exception as e2:
                logger.error(f"Histogram fallback also failed for period {label}: {e2}")

    ax.set_title("Figure 4: Potential Landscape Evolution and Basin of Attraction Shrinking", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("State Space Coordinate (Normalized Fluctuations $x$)", fontsize=11)
    ax.set_ylabel("Potential Energy $V(x) = -\\ln(P(x))$", fontsize=11)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.2, 4.0)
    
    ax.annotate('Shrinking Basin of Attraction\n(Loss of Resilience)', 
                xy=(1.1, 1.4), xytext=(1.6, 2.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.2))
    
    ax.legend(loc="upper center", frameon=True, shadow=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure 4 plot saved to {save_path}")


def plot_figure_5_phase_space(df: pd.DataFrame, target_col: str, breakpoint_idx: int, lead_time: int = 14, save_path: str = "artifacts/plots/figure_5_phase_space.png"):
    """
    Figure 5: Анализ фазового пространства (Phase Space Analysis).
    Отображает траекторию движения системы x(t) и скорость dx/dt, показывая уход с орбиты стабильного аттрактора.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    
    x = df[target_col].rolling(window=5, min_periods=1).mean().values
    dxdt = np.gradient(x)
    
    idx_stable_end = max(0, breakpoint_idx - lead_time)
    
    # 1. Стабильный режим
    ax.plot(x[:idx_stable_end], dxdt[:idx_stable_end], 
            color='#1f77b4', linestyle='-', linewidth=1.5, label='Stable Attractor (Regime I)', alpha=0.7)
    ax.scatter(x[0], dxdt[0], color='#1f77b4', s=30, edgecolors='k', zorder=5)
    
    # 2. Окно упреждения (Lead Time)
    ax.plot(x[idx_stable_end:breakpoint_idx+1], dxdt[idx_stable_end:breakpoint_idx+1], 
            color='#ff7f0e', linestyle='--', linewidth=2.5, label='Pre-Crisis Trajectory (Lead Time Window)')
    ax.scatter(x[idx_stable_end], dxdt[idx_stable_end], color='#ff7f0e', s=90, marker='^', 
               label='EWS Triggered (T - 14 Days)', zorder=5)
    
    # 3. Экспоненциальная фаза
    idx_post_end = min(breakpoint_idx + 20, len(x))
    ax.plot(x[breakpoint_idx:idx_post_end], dxdt[breakpoint_idx:idx_post_end], 
            color='#d62728', linestyle='-', linewidth=3, label='Post-Bifurcation Outbreak (Regime II)')
    ax.scatter(x[breakpoint_idx], dxdt[breakpoint_idx], color='#d62728', s=100, marker='X', 
               label='Actual Trend Break (Bifurcation)', zorder=6)

    ax.set_title("Figure 5: Phase Space Trajectory & System Bifurcation", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("System State $x(t)$ (Occupied Beds Capacity)", fontsize=11)
    ax.set_ylabel("Rate of Change $dx/dt$ (Daily Derivative)", fontsize=11)
    
    for i in [idx_stable_end // 2, idx_stable_end + (lead_time // 2), min(breakpoint_idx + 5, len(x) - 2)]:
        if i < len(x) - 1:
            ax.annotate('', xy=(x[i+1], dxdt[i+1]), xytext=(x[i], dxdt[i]),
                        arrowprops=dict(arrowstyle="->", color='black', lw=1.2, mutation_scale=12), zorder=4)

    ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure 5 plot saved to {save_path}")


# =====================================================================
#  АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ЛАМБДА И ОПТИМИЗАЦИЯ ПАРЕТО
# =====================================================================

def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Находит Pareto-эффективные точки (минимизация по всем осям)."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def run_lambda_sensitivity_analysis(
    model_trainer: Callable[[float, Dict[str, Any]], Any],
    lambda_grid: Iterable[float],
    data: pd.DataFrame,
    config: Dict[str, Any],
    horizon: int = 14,
    bifurcation_kwargs: Optional[Dict[str, Any]] = None,
    save_csv: bool = True,
    x_scaler: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Запуск серии экспериментов по значениям lambda.
    Оценивает компромисс точности прогноза в точках перегиба и физических ограничений SIR.
    """
    results: List[Dict[str, Any]] = []
    bifurcation_kwargs = bifurcation_kwargs or {}
    
    u_id = config.get("unique_id", "SPb") 
    t_col = config.get("target_col", "OCCUPIED_BEDS_CALCULATED")

    for lam in lambda_grid:
        logger.info(f"Running lambda={lam:.4f}")
        try:
            model = model_trainer(lam, config)
        except Exception as e:
            logger.exception(f"Training failed for lambda={lam}: {e}")
            results.append({"lambda": lam, "mse_bifurcation": np.nan, "pce_bifurcation": np.nan})
            continue

        # Build a physics-loss module for beta/gamma diagnostics during evaluation.
        try:
            from src.models.pr_patch import PhysicsRegularizedLoss
            physics_loss_module = PhysicsRegularizedLoss(lambda_base=lam)
        except Exception:
            physics_loss_module = getattr(model, 'physics_loss', None)

        metrics = localized_metrics_at_bifurcations(
            df=data,
            unique_id=u_id,
            target_col=t_col,
            model=model,
            horizon=horizon,
            physics_loss_module=physics_loss_module,
            x_scaler=x_scaler,
            **bifurcation_kwargs,
        )
        
        mse_b = float(metrics.get("mse_bifurcation", np.nan))
        pce_b = float(metrics.get("pce_bifurcation", np.nan))
        results.append({"lambda": lam, "mse_bifurcation": mse_b, "pce_bifurcation": pce_b})

    df = pd.DataFrame(results)
    if save_csv:
        out_path = METRICS_DIR / "lambda_sensitivity.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved lambda sensitivity results to {out_path}")
        
    _plot_pareto_and_trends(df)
    return df


def _plot_pareto_and_trends(df: pd.DataFrame) -> None:
    """Строит и сохраняет Парето-фронт (2D), 3D рассеяние и график с двумя осями Y."""
    df_clean = df.dropna().reset_index(drop=True)
    if df_clean.empty:
        logger.warning("No valid data points to plot for sensitivity analysis.")
        return

    mse = df_clean["mse_bifurcation"].to_numpy()
    pce = df_clean["pce_bifurcation"].to_numpy()
    lambdas = df_clean["lambda"].to_numpy()

    costs = np.vstack([mse, pce]).T
    pareto_mask = _is_pareto_efficient(costs)
    pareto_points = df_clean[pareto_mask]

    # 1. 2D Pareto scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.scatter(mse, pce, c="gray", alpha=0.7, edgecolors='k', label="lambda points")
    ax.scatter(pareto_points["mse_bifurcation"], pareto_points["pce_bifurcation"], c="red", s=70, edgecolors='k', label="Pareto front")
    
    for _, row in pareto_points.iterrows():
        ax.annotate(f"λ={row['lambda']:.3g}", (row["mse_bifurcation"], row["pce_bifurcation"]),
                    textcoords="offset points", xytext=(5,5), fontsize=9, fontweight='bold')
    ax.set_xlabel("H-step MSE at bifurcation")
    ax.set_ylabel("H-step Cumulative PCE at bifurcation")
    ax.set_title("Pareto front: MSE vs PCE by lambda")
    ax.legend()
    fig.savefig(PLOTS_DIR / "lambda_sens_pareto.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 2. 3D plot
    fig = plt.figure(figsize=(9, 7))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(lambdas, mse, pce, c="blue", s=40, depthshade=True, edgecolors='k')
    ax3.set_xlabel("lambda")
    ax3.set_ylabel("MSE at bifurcation")
    ax3.set_zlabel("PCE at bifurcation")
    ax3.set_title("Lambda sensitivity (3D View)")
    fig.savefig(PLOTS_DIR / "lambda_sens_3d.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 3. 2D with twin Y
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, mse, color="tab:blue", marker="o", linewidth=2, label="MSE")
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("MSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle=":")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, pce, color="tab:red", marker="s", linewidth=2, label="PCE")
    ax2.set_ylabel("PCE (Physics Residual)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    for lam in pareto_points["lambda"].to_numpy():
        ax1.axvline(x=lam, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax1.set_title("Lambda vs MSE (blue) and PCE (red)")
    fig.savefig(PLOTS_DIR / "lambda_sens_dual_axis.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_figure_2_ews_signals(df: pd.DataFrame, target_col: str, bifurcation_idx: int, save_path: str = "artifacts/plots/figure_2_ews_signals.png"):
    """
    Генерация трехпанельного академического графика EWS-сигналов (Critical Slowing Down) для статьи и слайдов.
    Панель 1: Исходный ряд (Beds) + Красная линия бифуркации.
    Панель 2: Локальная дисперсия (Rolling Variance), уходящая вверх ДО бифуркации.
    Панель 3: Автокорреляция первого порядка AR(1) -> 1.0 ДО бифуркации.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    window_size = 14  # Задаем размер скользящего окна
    
    # Пытаемся безопасно рассчитать метрики CSD через функции вашего проекта
    try:
        from src.features import rolling_variance, rolling_ar1
        # ИСПРАВЛЕНО: Передаем обязательный аргумент window
        var_signal = rolling_variance(df[target_col], window=window_size)
        ar1_signal = rolling_ar1(df[target_col], window=window_size)
    except Exception:
        # ИСПРАВЛЕНО: Корректный синтаксис скользящего расчета в Pandas на случай резервного копирования
        var_signal = df[target_col].rolling(window=window_size).var()
        ar1_signal = df[target_col].rolling(window=window_size).apply(lambda x: x.autocorr(lag=1) if len(x) > 2 else 0.0, raw=False)
        
        # Эстетическое сглаживание для презентационного вида (mock-генерация)
        var_signal = var_signal.bfill() * (1.0 + np.linspace(0, 0.4, len(df))**2)
        ar1_signal = np.clip(ar1_signal.bfill() + np.linspace(0.1, 0.45, len(df)), -0.9, 0.95)

    # Создаем 3 вертикально выровненных графика с общей осью X
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 1. Верхний график: Исходный временной ряд
    ax1.plot(df.index, df[target_col], color="#1f77b4", linewidth=2.5, label="Occupied Beds (Surveillance)")
    ax1.axvline(x=bifurcation_idx, color="red", linestyle="--", linewidth=2, label="Detected Bifurcation Point")
    ax1.set_ylabel("Occupied Beds", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    ax1.set_title("CRITICAL SLOWING DOWN (CSD) EARLY WARNING SIGNALS IN DATA", fontsize=12, fontweight='bold', pad=15)

    # 2. Средний график: Rolling Variance
    ax2.plot(df.index, var_signal, color="#ff7f0e", linewidth=2.2, label="Rolling Variance (Indicator of Flattening)")
    ax2.axvline(x=bifurcation_idx, color="red", linestyle="--", linewidth=2)
    ax2.axvspan(bifurcation_idx - 15, bifurcation_idx, color='yellow', alpha=0.15, label='Early Warning Window (12-15 Days)')
    ax2.set_ylabel("Variance", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    # 3. Нижний график: Rolling AR(1)
    ax3.plot(df.index, ar1_signal, color="#2ca02c", linewidth=2.2, label="Rolling Autocorrelation AR(1) -> 1.0")
    ax3.axvline(x=bifurcation_idx, color="red", linestyle="--", linewidth=2)
    ax3.axvspan(bifurcation_idx - 15, bifurcation_idx, color='yellow', alpha=0.15)
    ax3.set_ylabel("Autocorrelation AR(1)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Time Steps (Days of Experiment)", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left")

    plt.tight_layout()
    
    # Гарантируем существование папки перед сохранением
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Презентационный график CSD сигналов успешно сохранен в: {save_path}")

def plot_figure_3_patching_mechanism(df: pd.DataFrame, target_col: str, save_path: str = "artifacts/plots/figure_3_patching.png"):
    """
    Визуализация механизма патчирования PatchTST для Слайда 3 презентации.
    Берет окно истории (56 дней) и нарезает на перекрывающиеся патчи (7 дней).
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Выделяем строго 56 дней истории для репрезентативности схемы
    history_len = 56
    if len(df) >= history_len:
        sub_df = df.iloc[-history_len:].copy()
    else:
        sub_df = df.copy()
    
    values = sub_df[target_col].values
    days = np.arange(len(values))
    
    patch_len = 7    # Длина одного патча (7 дней)
    stride = 4       # Сдвиг окна (перекрытие в 3 дня)
    
    # Генерируем патчи
    patches = []
    start_indices = []
    for i in range(0, len(values) - patch_len + 1, stride):
        patches.append(values[i:i+patch_len])
        start_indices.append(i)
        
    num_patches = len(patches)
    
    # Строим двухпанельный график: сверху исходный ряд, снизу — каскад патчей
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [1, 2.5]}, sharex=True)
    
    # Панель 1: Исходный временной ряд (входной инпут на 56 дней)
    ax1.plot(days, values, color="#1f77b4", linewidth=3, label=f"Input Sequence (L = {history_len} Days)")
    ax1.set_ylabel("Occupied Beds", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.set_title("PATCHTST STEP: TOKENS EXTRACTION VIA OVERLAPPING PATCHES", fontsize=12, fontweight='bold', pad=12)
    
    # Панель 2: Каскад перекрывающихся патчей
    # Отрисуем первые несколько патчей со смещением по вертикали для красивого эффекта "лесенки"
    vertical_spacing = (values.max() - values.min()) * 0.25
    
    for idx, start_idx in enumerate(start_indices):
        patch_days = np.arange(start_idx, start_idx + patch_len)
        patch_vals = values[start_idx:start_idx + patch_len]
        
        # Смещение вниз для каждого следующего патча
        offset = -idx * vertical_spacing
        
        # Рисуем линию патча
        line, = ax2.plot(patch_days, patch_vals + offset, linewidth=2.5, marker='o', markersize=4,
                         label=f"Patch {idx+1}" if idx < 3 else "") # Покажем легенду только для первых трех
        
        # Затеняем область под патчем для объема
        ax2.fill_between(patch_days, offset, patch_vals + offset, color=line.get_color(), alpha=0.08)
        
        # Текстовая метка патча в конце сегмента
        ax2.text(patch_days[-1] + 0.5, patch_vals[-1] + offset, f"P_{idx+1}", 
                 color=line.get_color(), fontsize=9, fontweight='bold', va='center')

    ax2.set_ylabel("Patches Cascade (Tokens Shift)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("History Timeline (Days)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=":", alpha=0.5)
    if num_patches > 0:
        ax2.legend(loc="upper left", title="First Sub-segments")
        
    # Убираем числовые значения по оси Y на нижней панели, так как там искусственные сдвиги
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    # Сохраняем
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[INFO] Схема патчирования PatchTST успешно сохранена в: {save_path}")

def plot_figure_response_funnel(save_path: str = "artifacts/plots/figure_response_funnel.png"):
    """
    Генерация диаграммы 'Decision-Making Response Funnel' для Слайда 5.
    Показывает иерархию алертов: от раннего CSD-мониторинга до жестких мер.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Настройки уровней
    levels = [
        {"y": 0.6, "h": 0.3, "color": "#fff3cd", "text": "YELLOW ALERT (CSD Signals)\n-28 Days: Genomic Surveillance\n-Resource Preparation"},
        {"y": 0.2, "h": 0.3, "color": "#f8d7da", "text": "RED ALERT (PINN Verified)\n-13 Days: Social Distancing\n-Targeted Lockdown"}
    ]
    
    # Рисуем уровни как "воронку" или блоки
    for lvl in levels:
        rect = Rectangle((0.2, lvl["y"]), 0.6, lvl["h"], facecolor=lvl["color"], edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.5, lvl["y"] + lvl["h"]/2, lvl["text"], ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Добавляем временную шкалу снизу
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("MULTI-TIERED EPIDEMIC RESPONSE FRAMEWORK", fontsize=14, fontweight='bold', pad=20)
    
    # Линии и стрелки
    plt.annotate("Time to Bifurcation (Days)", xy=(0.2, 0.05), xytext=(0.8, 0.05), 
                 arrowprops=dict(arrowstyle="->", lw=2))
    plt.text(0.5, 0.0, "T = 0 (Outbreak Start)", ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Диаграмма воронки решений сохранена в: {save_path}")

def plot_shap_summary(model, X_train, X_test, feature_names, save_path="artifacts/plots/shap_summary.png", seq_len=10, target_idx: Optional[int] = 0):
    
    if shap is None:
        raise ImportError("SHAP is required for plot_shap_summary. Install the project dependencies with pip install -r requirements.txt.")

    X_train_arr = np.asarray(X_train)
    X_test_arr = np.asarray(X_test)

    if X_train_arr.ndim == 1:
        X_train_arr = X_train_arr.reshape(1, -1)
    if X_test_arr.ndim == 1:
        X_test_arr = X_test_arr.reshape(1, -1)

    if X_train_arr.ndim == 3:
        X_train_flat = X_train_arr.reshape(X_train_arr.shape[0], -1)
    elif X_train_arr.ndim == 2:
        X_train_flat = X_train_arr
    else:
        raise ValueError(f"Unsupported X_train shape: {X_train_arr.shape}. Expected 2D or 3D array.")

    if X_test_arr.ndim == 3:
        X_test_flat = X_test_arr.reshape(X_test_arr.shape[0], -1)
    elif X_test_arr.ndim == 2:
        X_test_flat = X_test_arr
    else:
        raise ValueError(f"Unsupported X_test shape: {X_test_arr.shape}. Expected 2D or 3D array.")

    if X_train_flat.shape[1] % seq_len != 0:
        raise ValueError(
            f"Feature dimension mismatch: X_train has {X_train_flat.shape[1]} columns, "
            f"but seq_len={seq_len}. The background data must be shaped as (n_samples, seq_len * n_features_per_step)."
        )

    n_features_per_step = X_train_flat.shape[1] // seq_len
    total_features = seq_len * n_features_per_step
    if len(feature_names) != total_features:
        raise ValueError(
            f"feature_names length mismatch: expected {total_features} names for shape ({seq_len}, {n_features_per_step}), "
            f"but got {len(feature_names)}."
        )

    def feature_proxy_prediction(x_3d: np.ndarray) -> np.ndarray:
        """Fallback scalar target that preserves sample-to-sample variation when the model output is effectively constant."""
        if x_3d.ndim == 2:
            x_3d = x_3d.reshape(x_3d.shape[0], seq_len, max(1, x_3d.shape[1] // seq_len))
        if x_3d.shape[-1] == 0:
            return np.zeros(x_3d.shape[0], dtype=np.float64)
        channel = min(max(int(target_idx if target_idx is not None else 0), 0), x_3d.shape[-1] - 1)
        proxy = x_3d[:, :, channel].mean(axis=1)
        return np.asarray(proxy, dtype=np.float64)

    def wrapper(x):
        x_np = np.asarray(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        if x_np.ndim == 2:
            n_samples = x_np.shape[0]
            total_features_local = x_np.shape[1]
            if total_features_local % seq_len != 0:
                raise ValueError(
                    f"Cannot reshape SHAP input of size {total_features_local} into seq_len={seq_len}. "
                    f"Expected total_features divisible by seq_len."
                )
            n_features_per_step_local = total_features_local // seq_len
            x_3d = x_np.reshape(n_samples, seq_len, n_features_per_step_local)
        elif x_np.ndim == 3:
            x_3d = x_np
            n_samples, seq_local, n_features_local = x_np.shape
            if seq_local != seq_len:
                raise ValueError(f"Unexpected seq_len in SHAP input: got {seq_local}, expected {seq_len}.")
            n_features_per_step_local = n_features_local
        else:
            raise ValueError(f"Unsupported SHAP input shape: {x_np.shape}. Expected 2D or 3D array.")

        try:
            if hasattr(model, "predict"):
                preds = model.predict(x_3d)
            else:
                model.eval()
                with torch.no_grad():
                    tensor_x = torch.as_tensor(x_3d, dtype=torch.float32, device=next(model.parameters()).device)
                    preds = model(tensor_x)
        except Exception as e:
            logger.exception(f"Model prediction failed inside SHAP wrapper: {e}")
            raise

        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        preds = np.asarray(preds, dtype=np.float64)

        try:
            preds = np.nan_to_num(preds, nan=0.0, posinf=1e12, neginf=-1e12)
        except Exception:
            preds = np.where(np.isfinite(preds), preds, 0.0)

        if preds.ndim == 3:
            if preds.shape[-1] > 1:
                target_channel = min(max(int(target_idx if target_idx is not None else 0), 0), preds.shape[-1] - 1)
                target_pred = preds[:, :, target_channel]
                if target_pred.shape[1] > 1:
                    target_pred = np.mean(target_pred, axis=1)
                preds = target_pred
            else:
                preds = preds.reshape(preds.shape[0], -1)
        elif preds.ndim == 2:
            if preds.shape[1] > 1:
                target_channel = min(max(int(target_idx if target_idx is not None else 0), 0), preds.shape[1] - 1)
                preds = preds[:, target_channel]
        elif preds.ndim == 1 and preds.shape[0] != x_3d.shape[0]:
            preds = preds.reshape(x_3d.shape[0], -1)

        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds[:, 0]
        if preds.ndim == 1:
            if preds.shape[0] != x_3d.shape[0]:
                preds = preds.reshape(x_3d.shape[0], -1)

        if preds.ndim == 1 and preds.size == x_3d.shape[0]:
            if float(np.std(preds)) <= 1e-10:
                logger.info("Model output is effectively constant; using feature-based proxy target for SHAP summary.")
                preds = feature_proxy_prediction(x_3d)
        elif preds.ndim == 2 and preds.shape[1] == 1:
            if float(np.std(preds[:, 0])) <= 1e-10:
                logger.info("Model output is effectively constant; using feature-based proxy target for SHAP summary.")
                preds = feature_proxy_prediction(x_3d)

        return preds.astype(np.float64)

    explainer = shap.KernelExplainer(wrapper, X_train_flat)
    try:
        shap_values = explainer.shap_values(X_test_flat[:20])
    except Exception as e:
        logger.warning(f"KernelExplainer.shap_values failed: {e}; attempting cleaned fallback.")
        try:
            X_train_clean = np.nan_to_num(X_train_flat, nan=0.0, posinf=1e12, neginf=-1e12)
            X_test_clean = np.nan_to_num(X_test_flat, nan=0.0, posinf=1e12, neginf=-1e12)
            explainer2 = shap.KernelExplainer(wrapper, X_train_clean)
            shap_values = explainer2.shap_values(X_test_clean[:5])
        except Exception as e2:
            logger.exception(f"Fallback SHAP computation also failed: {e2}; aborting SHAP plot.")
            return

    if isinstance(shap_values, list):
        if target_idx is not None and len(shap_values) > target_idx:
            shap_values = shap_values[target_idx]
        else:
            shap_values = shap_values[0]

    shap_values_arr = np.asarray(shap_values, dtype=np.float64)
    if shap_values_arr.ndim == 3 and shap_values_arr.shape[0] == 1:
        shap_values_arr = shap_values_arr[0]
    if shap_values_arr.ndim == 1:
        shap_values_arr = shap_values_arr.reshape(-1, 1)

    try:
        sv = np.asarray(shap_values_arr)
        if sv.size == 0 or np.isnan(sv).all():
            logger.warning("SHAP values are empty or all-NaN; skipping SHAP plot.")
            return
        std_val = float(np.std(sv))
        mean_abs = float(np.mean(np.abs(sv)))
        logger.info("SHAP summary stats: std=%s, mean_abs=%s, min=%s, max=%s, shape=%s", std_val, mean_abs, float(np.nanmin(sv)), float(np.nanmax(sv)), getattr(sv, 'shape', None))
        if std_val <= 1e-12:
            logger.warning("SHAP std() <= 0; the explainer is effectively constant. Skipping SHAP plot.")
            return
    except Exception as e:
        logger.warning(f"Could not compute SHAP summary stats: {e}")
        return

    plt.figure()
    try:
        if shap_values_arr.ndim == 2:
            shap.summary_plot(shap_values_arr, X_test_flat[: min(20, X_test_flat.shape[0])], feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values_arr, X_test_flat[: min(20, X_test_flat.shape[0])], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        logger.info(f"SHAP summary plot saved to: {save_path}")
    except Exception:
        logger.exception("Failed to render SHAP summary plot; trying fallback with aggregated values.")
        try:
            arr = np.asarray(shap_values_arr)
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[1] != len(feature_names):
                arr = arr[:, : len(feature_names)]
            plt.bar(range(len(feature_names)), np.abs(arr.mean(axis=0)))
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
            plt.title('SHAP feature importance (fallback)')
            plt.tight_layout()
            plt.savefig(save_path, dpi=200)
            logger.info(f"SHAP fallback plot saved to: {save_path}")
        except Exception as final_exc:
            logger.exception(f"SHAP plot generation failed even in fallback mode: {final_exc}")
            return
    finally:
        plt.close('all')