#!/usr/bin/env python3
"""Extract India lightning events from a folder of NASA LIS NetCDF event files."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    ensure_project_dirs,
    get_india_bounds,
    load_config,
    parse_time_index,
    resolve_path,
    setup_logging,
)


REQUIRED_OUTPUT = ["timestamp", "latitude", "longitude"]
OPTIONAL_RULES = {
    "event_id": (("event", "address"), ("parent", "child", "summary")),
    "radiance": (("radiance",), ("background", "bg_", "summary", "viewtime", "one_second")),
    "quality_flag": (("alert", "flag"), ("summary", "viewtime", "one_second")),
    "cluster_id": (("cluster", "index"), ("summary", "viewtime", "one_second")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--date", type=str, default=None, help="CTBT UTC date, YYYY-MM-DD. Defaults to earliest event date.")
    return parser.parse_args()


def import_netcdf4() -> Any:
    try:
        import netCDF4  # type: ignore
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required. Run: pip install -r requirements.txt") from exc
    return netCDF4


def variable_catalog(dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, variable in dataset.variables.items():
        attrs = {attr: variable.getncattr(attr) for attr in variable.ncattrs()}
        rows.append(
            {
                "name": name,
                "shape": tuple(variable.shape),
                "dtype": str(variable.dtype),
                "attrs": attrs,
                "text": " ".join(
                    [name, *[str(attrs.get(key, "")) for key in ("standard_name", "long_name", "units", "comment")]]
                ).lower(),
            }
        )
    return rows


def select_variable(catalog: list[dict[str, Any]], include: tuple[str, ...], exclude: tuple[str, ...]) -> str | None:
    best: tuple[int, str] | None = None
    for item in catalog:
        text = item["text"]
        name = item["name"].lower()
        if any(term in text for term in exclude):
            continue
        matches = sum(1 for term in include if term in text)
        if matches == 0:
            continue
        score = matches * 10
        if all(term in name for term in include):
            score += 30
        if "lightning_event" in name:
            score += 20
        if len(item["shape"]) == 1:
            score += 5
        candidate = (score, item["name"])
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None


def get_required_variables(catalog: list[dict[str, Any]]) -> dict[str, str]:
    selected = {
        "timestamp": select_variable(catalog, ("time",), ("delta", "observe_time", "start", "end", "bounds")),
        "latitude": select_variable(catalog, ("lat",), ("bounds",)),
        "longitude": select_variable(catalog, ("lon",), ("bounds",)),
    }
    missing = [name for name, value in selected.items() if not value]
    if missing:
        raise ValueError(f"Missing required variable(s): {', '.join(missing)}")
    return {key: str(value) for key, value in selected.items()}


def convert_time(values: np.ndarray, variable: Any, netcdf4: Any) -> list[str | None]:
    units = getattr(variable, "units", "")
    calendar = getattr(variable, "calendar", "standard")
    if units and "since" in str(units).lower():
        converted = netcdf4.num2date(values, units=units, calendar=calendar)
        return [item.strftime("%Y-%m-%d %H:%M:%S") for item in converted]

    epoch = datetime(1993, 1, 1, tzinfo=timezone.utc)
    output = []
    for value in pd.to_numeric(pd.Series(values), errors="coerce"):
        if pd.isna(value):
            output.append(None)
        else:
            output.append((epoch + pd.Timedelta(seconds=float(value))).strftime("%Y-%m-%d %H:%M:%S"))
    return output


def read_1d(dataset: Any, name: str, expected_length: int | None = None) -> np.ndarray | None:
    values = np.asarray(dataset.variables[name][:]).reshape(-1)
    if expected_length is not None and len(values) != expected_length:
        return None
    return values


def add_parent_ids(dataset: Any, catalog: list[dict[str, Any]], frame: pd.DataFrame) -> None:
    event_parent = select_variable(catalog, ("event", "parent", "address"), ("child", "summary"))
    group_address = select_variable(catalog, ("group", "address"), ("parent", "child", "summary"))
    group_parent = select_variable(catalog, ("group", "parent", "address"), ("child", "summary"))

    if event_parent:
        group_ids = read_1d(dataset, event_parent, len(frame))
        if group_ids is not None:
            frame["group_id"] = group_ids
            if group_address and group_parent:
                addresses = read_1d(dataset, group_address)
                parents = read_1d(dataset, group_parent)
                if addresses is not None and parents is not None:
                    lookup = dict(zip(addresses.tolist(), parents.tolist(), strict=False))
                    frame["flash_id"] = [lookup.get(value, np.nan) for value in group_ids]


def extract_file(path: Path, netcdf4: Any, logger) -> pd.DataFrame:
    with netcdf4.Dataset(str(path)) as dataset:
        catalog = variable_catalog(dataset)
        for item in catalog:
            logger.info("Variable discovered in %s: %s shape=%s dtype=%s", path.name, item["name"], item["shape"], item["dtype"])

        selected = get_required_variables(catalog)
        for output_name, (include, exclude) in OPTIONAL_RULES.items():
            name = select_variable(catalog, include, exclude)
            if name:
                selected[output_name] = name
        logger.info("Selected variables for %s: %s", path.name, selected)

        first = read_1d(dataset, selected["timestamp"])
        if first is None:
            return pd.DataFrame()
        event_count = len(first)
        data = {
            "timestamp": convert_time(first, dataset.variables[selected["timestamp"]], netcdf4),
            "latitude": read_1d(dataset, selected["latitude"], event_count),
            "longitude": read_1d(dataset, selected["longitude"], event_count),
        }
        for output_name, variable_name in selected.items():
            if output_name in data or output_name in REQUIRED_OUTPUT:
                continue
            values = read_1d(dataset, variable_name, event_count)
            if values is not None:
                data[output_name] = values

        frame = pd.DataFrame(data)
        add_parent_ids(dataset, catalog, frame)
        frame["source_file"] = path.name
        return frame


def clean_india_events(events: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    bounds = get_india_bounds(config)
    frame = events.copy()
    total = len(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")

    invalid_time = frame["timestamp"].isna()
    invalid_coords = frame["latitude"].isna() | frame["longitude"].isna()
    outside_india = ~frame.apply(lambda row: bounds.contains(row["latitude"], row["longitude"]), axis=1)
    keep = ~(invalid_time | invalid_coords | outside_india)
    cleaned = frame.loc[keep].copy()
    cleaned.drop_duplicates(subset=["timestamp", "latitude", "longitude"], inplace=True)
    cleaned.sort_values("timestamp", inplace=True)
    cleaned["timestamp"] = cleaned["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return cleaned.reset_index(drop=True), {
        "total_events": total,
        "invalid_timestamp": int(invalid_time.sum()),
        "invalid_coordinates": int((invalid_coords & ~invalid_time).sum()),
        "outside_india_bounds": int((outside_india & ~(invalid_time | invalid_coords)).sum()),
        "duplicates_removed": int(keep.sum()) - len(cleaned),
    }


def build_windows(metadata_path: Path, event_date: str | None, events: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    metadata = pd.read_csv(metadata_path, dtype=str)
    if event_date:
        date = pd.Timestamp(event_date, tz="UTC").date()
    elif not events.empty:
        date = pd.to_datetime(events["timestamp"], utc=True).min().date()
    else:
        date = pd.Timestamp.utcnow().date()
    interval = int(config["ctbt"]["interval_minutes"])

    windows = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        time_index = str(row["time_index"]).zfill(4)
        hour, minute = parse_time_index(time_index)
        start = pd.Timestamp(datetime.combine(date, datetime.min.time()), tz="UTC") + pd.Timedelta(hours=hour, minutes=minute)
        windows.append(
            {
                "time_index": time_index,
                "filename": row["filename"],
                "frame_number": index,
                "start": start,
                "end": start + pd.Timedelta(minutes=interval),
            }
        )
    return windows


def synchronize_ctbt(events: pd.DataFrame, config: dict, event_date: str | None, logger) -> None:
    output_dir = resolve_path(config, "processed_lightning")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = resolve_path(config, "metadata") / "metadata.csv"
    windows = build_windows(metadata_path, event_date, events, config)
    event_times = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")

    for window in tqdm(windows, desc="CTBT sync", unit="frame"):
        mask = (event_times >= window["start"]) & (event_times < window["end"])
        frame = events.loc[mask].copy()
        frame["ctbt_image"] = window["filename"]
        frame["frame_number"] = window["frame_number"]
        frame["time_window"] = f"{window['start'].strftime('%H:%M')}-{window['end'].strftime('%H:%M')}"
        path = output_dir / f"lightning_{window['time_index']}.csv"
        frame.to_csv(path, index=False)
        logger.info("Wrote %s rows=%s", path, len(frame))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(resolve_path(config, "logs") / "event_processing.log", "event_processing")
    start = time.perf_counter()
    logger.info("Event processing started")
    input_dir = args.input_dir or resolve_path(config, "raw_lis_events")
    output_dir = resolve_path(config, "processed_lightning")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.nc"))
    if not files:
        logger.error("No NetCDF files found in %s", input_dir)
        return 1

    netcdf4 = import_netcdf4()
    frames = []
    for path in tqdm(files, desc="LIS events", unit="file"):
        try:
            frame = extract_file(path, netcdf4, logger)
        except Exception as exc:
            logger.exception("Failed to process %s: %s", path, exc)
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        logger.error("No events extracted from %s files.", len(files))
        return 1

    all_events = pd.concat(frames, ignore_index=True)
    india_events, stats = clean_india_events(all_events, config)
    india_path = output_dir / "india_lightning_events.csv"
    india_events.to_csv(india_path, index=False)
    synchronize_ctbt(india_events, config, args.date, logger)
    logger.info("India events saved: %s rows=%s", india_path, len(india_events))
    logger.info("Stats: %s", stats)
    logger.info("Event processing finished duration_seconds=%.2f", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
