"""
artifacts_manager.py

Модуль для управления артефактами обучения (чекпоинты, метрики, графики, логи)
и шаблоны интеграции с PyTorch / Nixtla NeuralForecast.

Ключевые классы и функции:
- ArtifactsManager: создание структуры директорий и утилиты для сохранения
  чекпоинтов, метрик и графиков.
- pce_metric: шаблон физической метрики согласованности (Physics Consistency Error).
- helpers для интеграции с PyTorch Lightning / NeuralForecast (при наличии).

Все docstrings на русском. Код кроссплатформенный (pathlib) и с обработкой ошибок.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import logging
import pandas as pd

try:
    import torch
except Exception:
    torch = None


_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class ArtifactsManager:
    """Утилита для управления директорией `artifacts/`.

    При инициализации создаёт структуру:
    artifacts/
    ├── checkpoints/
    ├── metrics/
    ├── plots/
    └── logs/

    Параметры:
    - base_dir: Path или str, корневая папка артефактов.

    Методы:
    - get_checkpoint_dir(arch): получить/создать папку для весов архитектуры.
    - save_checkpoint(state, model_name, arch, epoch): сохранить torch state_dict.
    - save_metrics(df, name): сохранить pandas.DataFrame в CSV и JSON.
    - save_plot(fig, name_prefix, model_name, horizon): сохранить matplotlib-фигуру.
    - configure_pl_callback_lightning(...): вернуть ModelCheckpoint (если есть).
    """

    base_dir: Path | str = "artifacts"

    def __post_init__(self):
        self.base = Path(self.base_dir)
        self.checkpoints = self.base / "checkpoints"
        self.metrics = self.base / "metrics"
        self.plots = self.base / "plots"
        self.logs = self.base / "logs"
        try:
            for p in (self.base, self.checkpoints, self.metrics, self.plots, self.logs):
                p.mkdir(parents=True, exist_ok=True)
            _LOG.info("Artifacts structure ensured at %s", str(self.base))
        except PermissionError as e:
            _LOG.exception("Нет прав на создание артефактов в %s", str(self.base))
            raise
        except OSError as e:
            _LOG.exception("Ошибка файловой системы при создании директорий: %s", e)
            raise

    def get_checkpoint_dir(self, arch: str) -> Path:
        """Вернуть директорию для хранения чекпоинтов конкретной архитектуры.

        Создаёт папку `artifacts/checkpoints/{arch}` если её нет.
        """
        path = self.checkpoints / arch
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            _LOG.exception("Не удалось создать директорию чекпоинтов: %s", e)
            raise
        return path

    def unique_name(self, model_name: str, suffix: str = "", horizon: Optional[int] = None) -> str:
        """Сгенерировать уникальное имя файла: дата_модель[_h{H}][_suffix]."""
        now = datetime.now().strftime("%Y%m%dT%H%M%S")
        parts = [now, model_name]
        if horizon is not None:
            parts.append(f"h{horizon}")
        if suffix:
            parts.append(suffix)
        return "_".join(parts)

    def save_checkpoint(self, state: object, model_name: str, arch: str = "default", epoch: Optional[int] = None) -> Path:
        """Сохранить чекпоинт модели (torch compatible state).

        Параметры:
        - state: обычно state_dict или словарь для torch.save
        - model_name: имя модели
        - arch: имя архитектуры (каталог внутри checkpoints)
        - epoch: необязательно, номер эпохи

        Возвращает путь сохранённого файла.
        """
        if torch is None:
            _LOG.warning("torch не найден. Сохранение чекпоинтов требует PyTorch.")
        ckpt_dir = self.get_checkpoint_dir(arch)
        suffix = f"ep{epoch}" if epoch is not None else ""
        fname = self.unique_name(model_name, suffix=suffix) + ".pth"
        path = ckpt_dir / fname
        try:
            # torch.save может принимать любой питоновский объект
            if torch is not None:
                torch.save(state, path)
            else:
                # fallback: сериализуем в json, если возможно
                try:
                    with open(path.with_suffix('.json'), 'w', encoding='utf8') as f:
                        json.dump(state, f, default=str, ensure_ascii=False, indent=2)
                    path = path.with_suffix('.json')
                except Exception:
                    raise RuntimeError("Нет PyTorch и state не сериализуем в JSON")
            _LOG.info("Checkpoint saved to %s", str(path))
            return path
        except Exception as e:
            _LOG.exception("Ошибка при сохранении чекпоинта: %s", e)
            raise

    def save_metrics(self, df: pd.DataFrame, name: str, fmt: str = "csv") -> Tuple[Path, Path]:
        """Сохранить DataFrame с метриками в CSV и JSON.

        Возвращает кортеж (csv_path, json_path).
        """
        safe_name = name.replace(' ', '_')
        base_name = self.unique_name(safe_name)
        csv_path = self.metrics / f"{base_name}.csv"
        json_path = self.metrics / f"{base_name}.json"
        try:
            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient='records', force_ascii=False)
            _LOG.info("Metrics saved: %s, %s", csv_path, json_path)
            return csv_path, json_path
        except Exception as e:
            _LOG.exception("Ошибка при сохранении метрик: %s", e)
            raise

    def save_plot(self, fig, name_prefix: str, model_name: str, horizon: Optional[int] = None, dpi: int = 150) -> Path:
        """Сохранить matplotlib-фигуру `fig` в папку `artifacts/plots`.

        - fig: matplotlib.figure.Figure или None (тогда используется plt.gcf())
        - name_prefix: краткое описание (например 'forecast_compare')
        - model_name: имя модели
        - horizon: необязательный горизонт прогноза
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise RuntimeError("matplotlib не установлен")
        if fig is None:
            fig = plt.gcf()
        suffix = name_prefix
        fname = self.unique_name(model_name, suffix=suffix, horizon=horizon) + ".png"
        path = self.plots / fname
        try:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            _LOG.info("Plot saved to %s", path)
            return path
        except Exception as e:
            _LOG.exception("Ошибка при сохранении графика: %s", e)
            raise

    # ---------------- Integration helpers ----------------
    def configure_pytorch_lightning_checkpoint_callback(self, model_name: str, arch: str = "default"):
        """Попытка создать `pytorch_lightning.callbacks.ModelCheckpoint` настроенный
        на сохранение в каталог артефактов. Возвращает callback или None.

        Это облегчает интеграцию с Nixtla/NeuralForecast, если вы используете
        PyTorch Lightning для тренировки.
        """
        try:
            from pytorch_lightning.callbacks import ModelCheckpoint

            dirpath = str(self.get_checkpoint_dir(arch))
            cb = ModelCheckpoint(dirpath=dirpath, filename=f"{model_name}-%(epoch)d")
            return cb
        except Exception:
            _LOG.warning("pytorch_lightning не доступен; callback не создан")
            return None

    def configure_neuralforecast_paths(self, nf_obj, model_name: str, arch: str = "default") -> None:
        """Шаблон настройки путей для Nixtla NeuralForecast objects.

        Nixtla использует PyTorch под капотом; здесь мы даём пример, как можно
        указать путь для сохранения чекпоинтов / логов. nf_obj — объект, который
        вы используете (например, Trainer/NeuralForecast). Реальная интеграция
        зависит от вашей конфигурации: возможно, нужно передать callback,
        или настроить `checkpoint_callback`/`logger`.

        Этот метод пытается:
        - установить `nf_obj.checkpoint_callback` если существует
        - или записать атрибут `nf_obj.artifact_dir` для вашего пайплайна
        """
        ckpt_dir = self.get_checkpoint_dir(arch)
        try:
            # Пробуем поставить callback как в Lightning
            cb = self.configure_pytorch_lightning_checkpoint_callback(model_name, arch=arch)
            if cb is not None and hasattr(nf_obj, 'trainer'):
                try:
                    nf_obj.trainer.checkpoint_callback = cb
                    _LOG.info("Установлен Lightning checkpoint callback в trainer")
                    return
                except Exception:
                    _LOG.debug("Не удалось установить callback напрямую в nf_obj.trainer")
            # Бороться с различными API: просто установить поле
            setattr(nf_obj, 'artifact_dir', str(self.base))
            _LOG.info("Установлено nf_obj.artifact_dir = %s", str(self.base))
        except Exception as e:
            _LOG.exception("Не удалось настроить пути для NeuralForecast: %s", e)
            raise


def pce_metric(predictions: pd.Series, targets: pd.Series, physics_fn: Optional[Callable] = None) -> float:
    """Вычислить Physics Consistency Error (PCE).

    Шаблон: если задан `physics_fn`, вычисляем нарушение закона:
        r_pred = physics_fn(predictions)
        r_true = physics_fn(targets)
        pce = mean_absolute_error(r_pred, r_true)

    Если physics_fn не задан, используем простую метрику несоответствия вторых
    разностей (пример: ускорение для эпидемий как proxy):
        pce = mean(|dd_pred - dd_true|)

    Возвращает скаляр.
    """
    import numpy as _np

    if physics_fn is not None:
        try:
            r_pred = physics_fn(predictions)
            r_true = physics_fn(targets)
            return float(_np.mean(_np.abs(_np.asarray(r_pred) - _np.asarray(r_true))))
        except Exception as e:
            _LOG.exception("physics_fn вызвала ошибку: %s", e)
            raise
    # fallback: second-difference consistency
    try:
        dd_pred = _np.diff(predictions, n=2)
        dd_true = _np.diff(targets, n=2)
        L = min(len(dd_pred), len(dd_true))
        return float(_np.mean(_np.abs(dd_pred[:L] - dd_true[:L])))
    except Exception as e:
        _LOG.exception("Ошибка при вычислении PCE fallback: %s", e)
        raise


def example_usage_integration():
    """Короткий пример использования ArtifactsManager в пайплайне обучения.

    Этот код демонстрирует шаблон, который можно вставить в ваш тренер.
    - Настройка callback для PyTorch Lightning
    - Сохранение метрик после валидации
    - Сохранение графика прогнозов

    Код демонстрационный и не выполняет сам тренинг.
    """
    mgr = ArtifactsManager()
    # Пример: получить callback
    cb = mgr.configure_pytorch_lightning_checkpoint_callback("PatchTST", arch="PatchTST")
    if cb is not None:
        _LOG.info("ModelCheckpoint callback готов: %s", cb)

    # Предположим, после эпохи валидации получены метрики
    df = pd.DataFrame([{"model": "PatchTST", "mse": 0.12, "mae": 0.25, "pce": 0.05}])
    mgr.save_metrics(df, name="validation_metrics_patchtst")

    # Сохранение простого графика
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [1, 2, 3], label='obs')
        ax.plot([0, 1, 2], [1.1, 1.9, 2.9], label='pred')
        ax.legend()
        mgr.save_plot(fig, name_prefix='demo_forecast', model_name='PatchTST', horizon=24)
    except Exception:
        _LOG.exception("Не удалось сохранить демо-график")


if __name__ == '__main__':
    # Локальная демонстрация. В Jupyter просто импортируйте модуль и
    # используйте `ArtifactsManager()`.
    example_usage_integration()
