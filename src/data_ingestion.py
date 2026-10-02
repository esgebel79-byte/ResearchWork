"""Модуль для загрузки и предобработки сырых данных временных рядов.

Основные классы:
- DataIngestion: читает CSV, приводит к daily frequency, аггрегирует
  значения по правилам (flow -> sum, cumulative -> last) и обрабатывает
  пропуски (ffill / interpolate).

Пример использования:
    ing = DataIngestion(cfg)
    ing.run()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import logging
import pandas as pd

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class DataIngestion:
    """Класс загрузки и нормализации данных.

    Параметры:
    - cfg: словарь конфигурации (обычно из config/config.yaml)
    - raw_path: путь к сырым данным (если None используется cfg['data']['raw_path'])
    - date_col: имя столбца с датой
    - id_col: имя столбца уникального идентификатора временного ряда

    Ожидается, что CSV содержит столбцы: `unique_id`, `ds`, `y` и дополнительные признаки.
    """

    cfg: Dict
    raw_path: Optional[Path] = None
    date_col: str = "ds"
    id_col: str = "unique_id"

    def __post_init__(self):
        data_cfg = self.cfg.get("data", {})
        configured_raw = data_cfg.get("raw_path")
        candidates = []
        if configured_raw:
            candidates.append(Path(configured_raw))

        root = Path(__file__).resolve().parents[1]
        candidates.extend([
            root / "data" / "SPb.COVID-19.united.csv",
            root / "SPb.COVID-19.united.csv",
            root / "data" / "raw" / "cases.csv",
            root / "data" / "raw" / "data.csv",
        ])

        self.raw_path = next((p for p in candidates if p.exists()), Path(configured_raw) if configured_raw else root / "data" / "SPb.COVID-19.united.csv")
        self.processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
        self.date_col = data_cfg.get("date_col", self.date_col)
        self.id_col = data_cfg.get("id_col", self.id_col)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def read_raw(self) -> pd.DataFrame:
        """Считать CSV-файл и вернуть DataFrame.

        Бросает FileNotFoundError если файл не найден.
        """
        if not self.raw_path.exists():
            _LOG.error("Raw data file not found: %s", str(self.raw_path))
            raise FileNotFoundError(f"Raw data file not found: {self.raw_path}")

        df = pd.read_csv(self.raw_path)
        if self.id_col not in df.columns:
            df[self.id_col] = 'SPb'

        if self.date_col not in df.columns:
            for candidate in ['DATE.spb', 'TIME.sk', 'ds']:
                if candidate in df.columns:
                    df[self.date_col] = df[candidate]
                    break
            else:
                df[self.date_col] = pd.date_range('2020-01-01', periods=len(df), freq='D')

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        _LOG.info("Raw data loaded: %s rows", len(df))
        return df

    def to_daily(self, df: pd.DataFrame, agg_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Привести DataFrame к ежедневной частоте.

        Параметры:
        - df: DataFrame с колонками `id_col`, `date_col` и прочими признаками.
        - agg_map: словарь колонка -> агрегация ('sum'|'last'|'mean')

        По-умолчанию агрегируем потоковые признаки суммированием, кумулятивные — last.
        Возвращает агрегированный DataFrame с MultiIndex (unique_id, ds).
        """
        df = df.copy()
        if self.id_col not in df.columns:
            df[self.id_col] = 'SPb'
        if self.date_col not in df.columns:
            for candidate in ['DATE.spb', 'TIME.sk', 'ds']:
                if candidate in df.columns:
                    df[self.date_col] = df[candidate]
                    break
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.normalize()

        if agg_map is None:
            agg_map = {}
            targets = self.cfg.get('data', {}).get('targets', None)
            val_cols = [c for c in df.columns if c not in (self.id_col, self.date_col)]
            if targets:
                val_cols = [c for c in targets if c in df.columns]
            for col in val_cols:
                name = col.lower()
                if 'test' in name or 'pcr' in name or 'tests' in name:
                    agg_map[col] = 'sum'
                elif 'confirm' in name or name.startswith('cases') or 'confirmed' in name:
                    agg_map[col] = 'last'
                elif 'active' in name or 'occup' in name or 'bed' in name:
                    agg_map[col] = 'mean'
                else:
                    agg_map[col] = 'last'

        agg_candidates = {k: v for k, v in agg_map.items() if k in df.columns}
        grouped = df.groupby([self.id_col, self.date_col]).agg(agg_candidates)
        out_frames: List[pd.DataFrame] = []
        for uid, g in grouped.groupby(level=0):
            g = g.droplevel(0).sort_index()
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq='D')
            g = g.reindex(full_idx)
            g[self.id_col] = uid
            g = g.reset_index().rename(columns={'index': self.date_col})
            out_frames.append(g)

        daily = pd.concat(out_frames, ignore_index=True)
        num_cols = daily.select_dtypes(include='number').columns.tolist()
        if num_cols:
            daily[num_cols] = daily[num_cols].ffill().interpolate()
        daily[self.id_col] = daily[self.id_col].ffill()

        _LOG.info("Converted to daily frequency: %d rows", len(daily))
        return daily

    def run(self) -> Path:
        """Полный pipeline: чтение raw -> daily -> сохранить в processed_dir.

        Возвращает путь к сохранённому CSV.
        """
        df = self.read_raw()
        daily = self.to_daily(df)
        out_path = self.processed_dir / "data_daily.csv"
        try:
            daily.to_csv(out_path, index=False)
            _LOG.info("Processed data saved to %s", str(out_path))
        except Exception as e:
            _LOG.exception("Не удалось сохранить processed data: %s", e)
            raise
        return out_path


if __name__ == "__main__":
    from src.config import load_config
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    ing = DataIngestion(cfg)
    ing.run()
