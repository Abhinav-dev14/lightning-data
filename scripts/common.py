"""Shared helpers for the AKAM prediction and visualization pipelines."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "project_config.json"


@dataclass(frozen=True)
class IndiaBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: Any, lon: Any) -> bool:
        try:
            lat_value = float(lat)
            lon_value = float(lon)
        except (TypeError, ValueError):
            return False
        return (
            self.lat_min <= lat_value <= self.lat_max
            and self.lon_min <= lon_value <= self.lon_max
        )


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = (config_path or DEFAULT_CONFIG).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_config_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_lightning_source(config: dict[str, Any]) -> str:
    source = str(config.get("lightning_source", "synthetic")).lower()
    supported = {"synthetic", "nasa", "ildn", "nrsc"}
    if source not in supported:
        raise ValueError(f"Unsupported lightning_source '{source}'. Supported values: {sorted(supported)}")
    return source


def get_lightning_source_dir(config: dict[str, Any]) -> Path:
    source = get_lightning_source(config)
    source_paths = config.get("lightning_sources", {})
    if source not in source_paths:
        if source == "synthetic":
            return resolve_path(config, "processed_lightning")
        return resolve_path(config, f"raw_{source}")
    return resolve_config_path(str(source_paths[source]))


def ensure_project_dirs(config: dict[str, Any]) -> None:
    for value in config["paths"].values():
        path = Path(value)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)


def get_india_bounds(config: dict[str, Any]) -> IndiaBounds:
    bounds = config["india_bounds"]
    return IndiaBounds(
        lat_min=float(bounds["lat_min"]),
        lat_max=float(bounds["lat_max"]),
        lon_min=float(bounds["lon_min"]),
        lon_max=float(bounds["lon_max"]),
    )


def setup_logging(log_path: Path, logger_name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_time_index(time_index: str) -> tuple[int, int]:
    value = str(time_index).strip().zfill(4)
    return int(value[:2]), int(value[2:])
