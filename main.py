"""
Main Execution Pipeline for Neuro-Physical Hybrid Forecasting.
Author: Elena Sergeevna Gebel
Year: 2026

Combines Feature Ingestion, Lambda Sensitivity Analysis (Pareto Front),
and Local Explainability (SHAP/LIME) at detected bifurcation points.
"""
import os
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Принудительно отключаем GUI для matplotlib, чтобы графики корректно рендерились в фоне/серверах
import matplotlib
matplotlib.use('Agg')

# Гарантируем бесконфликтный поиск модулей внутри 'src' из корня проекта
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation import run_lambda_sensitivity_analysis, detect_bifurcations
from src.explainability import explain_shap_prpatch, explain_lime_instance

# Попытка импорта реальной модели из вашей новой структуры папок
try:
    from src.models.pr_patch import PRPatchModel
    HAS_REAL_MODEL = True
except ImportError:
    HAS_REAL_MODEL = False

# Настройка логирования для MLOps-мониторинга
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Безопасная загрузка конфигурационных параметров эксперимента с фолбеком."""
    path = project_root / config_path
    if not path.exists():
        logger.warning(f"Конфигурация {config_path} не найдена в {path}. Генерируем дефолтный конфиг.")
        return {
            "unique_id": "SPb",
            "target_col": "OCCUPIED_BEDS_CALCULATED",
            "seq_len": 56,
            "patch_size": 7,
            "horizon": 14,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dummy_model_trainer(lam: float, config: dict) -> torch.nn.Module:
    """
    Фабрика моделей. Автоматически извлекает геометрические параметры
    из вложенных секций YAML-конфига.
    """
    if HAS_REAL_MODEL:
        try:
            logger.info(f"Инициализация реальной PRPatchModel с lambda={lam}")
            
            # Извлекаем параметры из соответствующих секций
            train_cfg = config.get("training", {})
            patch_cfg = config.get("patchtst", {})
            data_cfg = config.get("data", {})
            
            seq_len = train_cfg.get("seq_len", 56)
            patch_size = patch_cfg.get("patch_size", 7)
            
            # Количество основных каналов (динамически берем из длины списка targets)
            # В вашем конфиге это 4 таргета: PCR_TESTS, CONFIRMED.sk, ACTIVE.sk, OCCUPIED_BEDS_CALCULATED
            n_inputs = len(data_cfg.get("targets", [])) or 3 
            
            return PRPatchModel(
                seq_len=seq_len,
                patch_size=patch_size,
                hidden=64,             # Базовое значение, можно также вынести в конфиг
                n_inputs=n_inputs,     # Динамически подстроится под 4 переменных
                n_ews=2                # Фиксировано (variance, ar1)
            )
        except Exception as e:
            logger.warning(f"Не удалось собрать реальную модель: {e}. Переход на эмулятор.")

        def __init__(self):
            super().__init__()
            self.beta = torch.nn.Parameter(torch.tensor(0.25))
            self.gamma = torch.nn.Parameter(torch.tensor(0.10))
            # Фиктивный линейный слой, чтобы у модели были обучаемые параметры
            self.dummy_param = torch.nn.Linear(1, 1)
            
        def forward(self, x, ews=None):
            batch_size = x.shape[0]
            horizon = config.get("horizon", 14)
            # Извлекаем количество выходных каналов (фич) динамически
            channels = x.shape[2] if x.ndim == 3 else 1
            return torch.zeros((batch_size, horizon, channels), device=x.device)
            
    return EmulatedPatchModel()


def main():
    logger.info("=== Запуск сквозного нейрофизического конвейера ===")
    
    # 0. Инициализация окружения и путей к артефактам
    config = load_config()
    device = config.get("device", "cpu")
    seq_len = config.get("seq_len", 56)
    patch_size = config.get("patch_size", 7)
    horizon = config.get("horizon", 14)
    target_col = config.get("target_col", "OCCUPIED_BEDS_CALCULATED")
    
    # Автоматическое развертывание экосистемы папок в корне
    (project_root / "artifacts/metrics").mkdir(parents=True, exist_ok=True)
    (project_root / "artifacts/plots").mkdir(parents=True, exist_ok=True)

    # 1. Загрузка данных и генерация признаков
    logger.info("Шаг 1: Подготовка мультивариантного датасета и EWS-индикаторов...")
    
    # Эмуляция пайплайна (data_ingestion + features)
    dates = pd.date_range(start="2025-01-01", periods=150, freq="D")
    np.random.seed(42)
    
    # Моделируем физический критический сдвиг (Bifurcation) в районе 80-го дня
    signal = np.sin(np.linspace(0, 10, 150)) * 100 + 200
    signal[80:] += np.linspace(0, 300, 70)  # Экспоненциальный всплеск (инфекционный взрыв)
    
    df_data = pd.DataFrame({
        "ds": dates,
        "unique_id": config.get("unique_id", "SPb"),
        target_col: signal,
        "PCR_TESTS": signal * 0.8 + np.random.normal(0, 10, 150),
        "CONFIRMED.sk": signal * 0.5 + np.random.normal(0, 5, 150),
        "ACTIVE.sk": signal * 0.3 + np.random.normal(0, 5, 150),
        # EWS индикаторы
        "var_ews": np.random.rand(150) * 0.1,  # Rolling Variance
        "ar1_ews": np.random.rand(150) * 0.9   # Rolling AR(1)
    })

    # 2. Анализируем чувствительность по Парето-компромиссу
    logger.info("Шаг 2: Запуск анализа чувствительности по сетке регуляризации lambda...")
    lambda_grid = config.get("physics", {}).get("lambda_grid", [0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
    
    df_sens = run_lambda_sensitivity_analysis(
        model_trainer=dummy_model_trainer,
        lambda_grid=lambda_grid,
        data=df_data,
        config=config,
        horizon=horizon
    )

    # 3. Интерпретация локальных патчей в окрестностях бифуркаций
    logger.info("Шаг 3: Поиск точек излома тренда и запуск SHAP/LIME интерпретации...")
    
    bif_indices = detect_bifurcations(df_data[target_col], win=7)
    if not bif_indices:
        logger.warning("Точки бифуркации математически не зафиксированы. Прогон SHAP на базовом срезе.")
        bif_indices = [85]
        
    logger.info(f"Критические временные индексы перелома тренда: {bif_indices}")
    
    # Подготовка трехмерных тензоров для анализа SHAP
    # Спецификация формы: (samples, seq_len, features)
    feature_cols = ["OCCUPIED_BEDS_CALCULATED", "PCR_TESTS", "CONFIRMED.sk", "ACTIVE.sk"]
    X_raw = df_data[feature_cols].values
    
    X_background = []
    for i in range(0, min(50, len(X_raw) - seq_len - horizon)):
        X_background.append(X_raw[i : i + seq_len])
    X_background = np.array(X_background)

    # Извлекаем окно непосредственно перед первой бифуркацией
    target_idx = max(0, min(bif_indices[0] - seq_len, len(X_raw) - seq_len))
    X_instance = X_raw[target_idx : target_idx + seq_len].reshape(1, seq_len, -1)

    # Фиксируем оптимальную модель из сетки Парето
    best_model = dummy_model_trainer(lam=0.05, config=config)
    
    def predict_fn(x3d):
        best_model.eval()
        t_x = torch.tensor(x3d, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = best_model(t_x)
            return out.cpu().numpy()

    # Спецификация имен для flat-структуры SHAP (seq_len * n_features)
    flat_feature_names = []
    for f_name in feature_cols:
        for lag in range(seq_len):
            flat_feature_names.append(f"{f_name}_lag_{lag}")

    logger.info("Генерация карт важности SHAP (с деагрегацией патч -> лаг)...")
    explain_shap_prpatch(
        model_or_predict_fn=predict_fn,
        X_background=X_background,
        X_instance=X_instance,
        seq_len=seq_len,
        patch_size=patch_size,
        target_name=target_col,
        background_size=30,
        device=device,
        feature_names=flat_feature_names
    )

    logger.info("Генерация локальных весов LIME...")
    explain_lime_instance(
        predict_fn=predict_fn,
        X_train=X_background,
        instance=X_instance[0],
        target_name=target_col,
        num_features=10,
        feature_names=flat_feature_names
    )

    logger.info("=== Пайплайн успешно завершен! Сгенерированные артефакты сохранены в папки 'artifacts/' ===")


if __name__ == "__main__":
    main()