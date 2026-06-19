"""Модуль для загрузки конфигурации проекта из YAML.

Содержит функцию `load_config(path)` возвращающую словарь конфигурации.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Загрузить YAML-конфигурацию по пути `path`.

    Возвращает словарь. Бросает FileNotFoundError или yaml.YAMLError при ошибке.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    # простой тест
    cfg = load_config(Path(__file__).parents[1] / "config" / "config.yaml")
    print(cfg)
