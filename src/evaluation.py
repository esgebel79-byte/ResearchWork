"""
Evaluation utilities focused on critical transition / bifurcation-local metrics
and lambda physical regularization sensitivity analysis (Pareto optimization).

Functions:
- detect_bifurcations : detect candidate transition indices via slope change thresholds.
- localized_metrics_at_bifurcations : compute local MSE and physical constraints (PCE) near bifurcations.
- run_lambda_sensitivity_analysis : execute a grid search over lambda to generate Pareto front plots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable, Iterable
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

# Инструмент отрисовки графиков для Парето-анализа
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Явно подключаем для корректной поддержки 3D-проекции

# Безопасный импорт EWS и лосса физики из ваших модулей
try:
    from src.features import rolling_variance, rolling_ar1
    from src.models.pr_patch import cumulative_pce
except ImportError:
    # Заглушка-фолбек для изолированных тестов pipeline
    def cumulative_pce(pred_seq, beta, gamma, N):
        return torch.tensor(0.0)

# Экосистема директорий артефактов
ARTIFACTS = Path("artifacts")
METRICS_DIR = ARTIFACTS / "metrics"
PLOTS_DIR = ARTIFACTS / "plots"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


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
    
    # Сдвиг к моменту времени прогноза (последняя точка окна + 1)
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
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute localized MSE and cumulative PCE around detected bifurcations for a single series.
    Синхронизирован под имена ключей анализа чувствительности.
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
            
            # Настройка и согласование каналов с конфигурацией модели
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

            # Генерация H-шаговой траектории
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
                    
                    # Безопасная склейка и транспонирование авторегрессионного прогноза
                    if len(preds_step) > 0:
                        raw_cat = torch.cat(preds_step, dim=0)
                        if raw_cat.ndim == 2:
                            pred_seq_tensor = raw_cat.unsqueeze(-1).permute(1, 0, 2)
                        else:
                            pred_seq_tensor = raw_cat.permute(1, 0, 2)
                    else:
                        pred_seq_tensor = torch.empty((1, 0, x_t.shape[2]), device=device)

            future_end = end + horizon
            if future_end <= len(df_loc):
                true_future = df_loc.iloc[end: future_end][cols].values.astype(float)
            else:
                true_future = None

            # Расчет локального MSE
            try:
                pred_np = pred_seq_tensor.cpu().numpy()[0]
                if true_future is not None and true_future.shape[0] >= pred_np.shape[0]:
                    mse_h = float(np.mean((pred_np[:, 0] - true_future[: pred_np.shape[0], 0]) ** 2))
                else:
                    mse_h = float(np.mean((pred_np[:, 0]) ** 2))
            except Exception:
                mse_h = float('nan')
            mse_list.append(mse_h)

            # Вычисление физической невязки физического лосса (Cumulative PCE)
            try:
                if x_t.shape[2] >= 3:
                    last_obs = x_t.cpu().numpy()[0, -1, 0:3]
                    N = float(np.sum(last_obs))
                else:
                    N = 1.0
                if physics_loss_module is not None and hasattr(physics_loss_module, 'beta') and hasattr(physics_loss_module, 'gamma'):
                    beta_val = float(physics_loss_module.beta.detach().cpu().numpy())
                    gamma_val = float(physics_loss_module.gamma.detach().cpu().numpy())
                else:
                    beta_val = 0.0
                    gamma_val = 0.0
                pce_val = float(cumulative_pce(pred_seq_tensor.cpu(), beta_val, gamma_val, N).cpu().numpy())
            except Exception:
                pce_val = float('nan')
            pce_list.append(pce_val)

    return {
        'n_bifurcations': len(idxs),
        'mse_bifurcation': float(np.nanmean(mse_list)) if len(mse_list) > 0 else float('nan'),
        'pce_bifurcation': float(np.nanmean([v for v in pce_list if not np.isnan(v)])) if any([not np.isnan(v) for v in pce_list]) else float('nan')
    }


# =====================================================================
#  БЛОК АНАЛИЗА ЧУВСТВИТЕЛЬНОСТИ ЛАМБДА (МНОГОКРИТЕРИАЛЬНЫЙ ПАРЕТО-ФРОНТ)
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

        metrics = localized_metrics_at_bifurcations(
            df=data,
            unique_id=u_id,
            target_col=t_col,
            model=model,
            horizon=horizon,
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


if __name__ == '__main__':
    print('Module src.evaluation unified. Ready to support run_lambda_sensitivity_analysis inside main.py.')