"""
neuralforecast_integration.py

Класс `NFTrainingMonitor` автоматизирует интеграцию мониторинга обучения для
NeuralForecast / PyTorch-пайплайнов:

- Считает метрики MSE, MAE и кастомную PCE (Physics Consistency Error).
- Логирует метрики в TensorBoard (SummaryWriter).
- Сохраняет метрики в `artifacts/metrics/` через `ArtifactsManager`.

Файл демонстрационный: поддерживает два режима интеграции:
1) Встраивание как PyTorch Lightning Callback (если используется Lightning).
2) Вызов вручную после этапа валидации: `monitor.update_and_log(preds, targets, epoch)`.

Docstrings и сообщения на русском. Код обрабатывает ошибки создания директорий
и отсутствие optional-зависимостей.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional
import logging
import numpy as np
import pandas as pd

from artifacts_manager import ArtifactsManager, pce_metric

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import pytorch_lightning as pl
except Exception:
    pl = None

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class NFTrainingMonitor:
    """Монитор для обучения NeuralForecast/PyTorch.

    Параметры:
    - artifacts: экземпляр `ArtifactsManager` для сохранения артефактов.
    - model_name: имя модели (используется при именовании файлов).
    - horizon: прогнозный горизонт (для именования/логов).
    - physics_fn: опциональная функция, применимая к траекториям для PCE.
    - tb_log_dir: каталог логов TensorBoard (если None, будет создан в artifacts/logs).
    """

    artifacts: ArtifactsManager
    model_name: str
    horizon: Optional[int] = None
    physics_fn: Optional[Callable] = None
    tb_log_dir: Optional[Path] = None
    _writer: Optional[SummaryWriter] = field(init=False, default=None)
    _history: pd.DataFrame = field(init=False, default_factory=lambda: pd.DataFrame())

    def __post_init__(self):
        # Настроим TensorBoard writer
        try:
            if SummaryWriter is None:
                _LOG.warning("torch.utils.tensorboard SummaryWriter недоступен. Установите torch>=1.2 и tensorboard.")
            log_dir = self.tb_log_dir or (Path(self.artifacts.logs) / self.model_name)
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            if SummaryWriter is not None:
                self._writer = SummaryWriter(log_dir=str(log_dir))
                _LOG.info("TensorBoard writer -> %s", str(log_dir))
        except PermissionError as e:
            _LOG.exception("Нет прав на создание tb_log_dir: %s", e)
            raise

    # ---------------- Core metrics ----------------
    @staticmethod
    def _compute_basic_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Вычислить MSE и MAE между preds и targets.

        Ожидается, что входы — 1D или 2D массивы с одинаковой формой.
        Возвращает словарь { 'mse': ..., 'mae': ... }
        """
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        try:
            diff = preds - targets
            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))
        except Exception as e:
            _LOG.exception("Ошибка при вычислении базовых метрик: %s", e)
            raise
        return {"mse": mse, "mae": mae}

    def compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Вычислить и вернуть метрики: MSE, MAE, PCE.

        PCE вычисляется через функцию `pce_metric` из `artifacts_manager`.
        """
        base = self._compute_basic_metrics(preds, targets)
        try:
            pce = pce_metric(pd.Series(preds.flatten()), pd.Series(targets.flatten()), physics_fn=self.physics_fn)
        except Exception:
            _LOG.exception("PCE calculation failed; setting pce=nan")
            pce = float('nan')
        base.update({"pce": float(pce)})
        return base

    # ---------------- Logging & saving ----------------
    def update_and_log(self, preds: np.ndarray, targets: np.ndarray, epoch: int) -> pd.DataFrame:
        """Вычислить метрики, залогировать в TensorBoard и сохранить в artifacts.

        Возвращает DataFrame с одной строкой (epoch, mse, mae, pce).
        Этот метод можно вызывать из внешнего валидационного шага, например
        после `nf.backtest()` или внутри callback.
        """
        metrics = self.compute_metrics(preds, targets)
        metrics_row = {"epoch": int(epoch), "model": self.model_name, "horizon": self.horizon}
        metrics_row.update(metrics)
        df_row = pd.DataFrame([metrics_row])

        # лог в TensorBoard
        try:
            if self._writer is not None:
                for k, v in metrics.items():
                    self._writer.add_scalar(k, v, global_step=epoch)
                self._writer.flush()
        except Exception:
            _LOG.exception("Ошибка логирования в TensorBoard")

        # Сохраняем метрики в artifacts/metrics; аккумулируем историю
        try:
            # append to internal history and save full df snapshot
            self._history = pd.concat([self._history, df_row], ignore_index=True)
            # save snapshot and also export unique timestamped file
            self.artifacts.save_metrics(self._history, name=f"metrics_history_{self.model_name}")
        except Exception:
            _LOG.exception("Ошибка при сохранении метрик в артефактах")

        _LOG.info("Epoch %s metrics: %s", epoch, metrics)
        return df_row

    # ---------------- PyTorch Lightning callback integration ----------------
    def get_lightning_callback(self):
        """Вернуть PyTorch Lightning Callback, который вызывает `update_and_log`.

        ВАЖНО: Callback ожидает, что `pl_module` после валидации выставит
        атрибуты `val_preds` и `val_targets` — массивы numpy или torch tensors.
        В противном случае, вам нужно адаптировать callback под ваш модуль.
        """
        if pl is None:
            _LOG.warning("pytorch_lightning не установлен; callback не создан")
            return None

        monitor = self

        class _PLCallback(pl.Callback):
            def on_validation_epoch_end(self, trainer, pl_module):
                try:
                    preds = getattr(pl_module, 'val_preds', None)
                    targets = getattr(pl_module, 'val_targets', None)
                    if preds is None or targets is None:
                        _LOG.warning("pl_module не содержит val_preds/val_targets; пропуск логирования PCE")
                        return
                    # convert tensors to numpy
                    import torch as _torch
                    if _torch.is_tensor(preds):
                        preds_np = preds.detach().cpu().numpy()
                    else:
                        preds_np = np.asarray(preds)
                    if _torch.is_tensor(targets):
                        targets_np = targets.detach().cpu().numpy()
                    else:
                        targets_np = np.asarray(targets)
                    epoch = trainer.current_epoch if hasattr(trainer, 'current_epoch') else -1
                    monitor.update_and_log(preds_np, targets_np, epoch)
                except Exception:
                    _LOG.exception("Ошибка в PL callback при логировании метрик")

        return _PLCallback()

    # ---------------- NeuralForecast integration example ----------------
    def integrate_with_neuralforecast_backtest(self, nf_obj, backtest_func: Callable, epoch: int = 0):
        """Пример использования с NeuralForecast-style backtest.

        Параметры:
        - nf_obj: объект NeuralForecast или модель, для которой можно вызвать backtest/predict
        - backtest_func: calllable(nf_obj) -> (preds, targets) возращает массивы
        - epoch: номер эпохи/шага для логирования

        Метод вызывает `backtest_func`, получает preds/targets, затем вызывает
        `update_and_log`.
        """
        try:
            preds, targets = backtest_func(nf_obj)
            self.update_and_log(np.asarray(preds), np.asarray(targets), epoch)
        except Exception:
            _LOG.exception("Ошибка интеграции с NeuralForecast backtest")


if __name__ == '__main__':
    # Пример использования (демонстрация):
    from artifacts_manager import ArtifactsManager

    mgr = ArtifactsManager()
    monitor = NFTrainingMonitor(artifacts=mgr, model_name='PR-Patch', horizon=24)

    # Демонстрация: синтетические preds/targets
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.1, 1.9, 3.2, 3.8])
    monitor.update_and_log(preds, targets, epoch=1)
