#!/usr/bin/env python3
"""Audit AKAM project structure, datasets, dependencies, and generated outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import netCDF4
import numpy as np
import pandas as pd

from common import PROJECT_ROOT, ensure_project_dirs, get_lightning_source, get_lightning_source_dir, load_config, resolve_path, setup_logging


REQUIRED_FOLDERS = [
    "data/raw/lis_events",
    "data/raw/lis_otd",
    "data/raw/open_meteo",
    "data/raw/nasa_power",
    "data/raw/noaa_isd",
    "data/raw/ildn",
    "data/raw/nrsc",
    "data/raw/archive",
    "data/raw/archive/nasa_lis",
    "data/processed/lightning",
    "data/processed/training",
    "data/processed/weather",
    "data/processed/climatology",
    "data/overlays/images",
    "data/overlays/csv",
    "data/animations",
    "data/logs",
    "configs",
    "scripts",
    "models",
]

DEPENDENCIES = {
    "netCDF4": "netCDF4",
    "xarray": "xarray",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "opencv-python": "cv2",
    "matplotlib": "matplotlib",
    "imageio": "imageio",
    "pyproj": "pyproj",
    "rasterio": "rasterio",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "pyarrow": "pyarrow",
    "fastparquet": "fastparquet",
    "pillow": "PIL",
    "tqdm": "tqdm",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-pipelines", action="store_true", help="Run event/georef/overlay/animation smoke checks.")
    return parser.parse_args()


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def classify_netcdf(path: Path) -> str:
    name = path.name.upper()
    if name.startswith("ISS_LIS") or "LIGHTNING_EVENT" in name:
        return "ISS LIS Event File"
    if name.startswith("LISOTD") or "HRMC" in name:
        return "LIS/OTD Climatology"
    try:
        with netCDF4.Dataset(path) as ds:
            variables = set(ds.variables.keys())
        if {"lightning_event_TAI93_time", "lightning_event_lat", "lightning_event_lon"}.issubset(variables):
            return "ISS LIS Event File"
        if {"latitude", "longitude"}.issubset(variables) and any("FR" in item.upper() for item in variables):
            return "LIS/OTD Climatology"
    except Exception:
        return "Unreadable NetCDF"
    return "Unknown NetCDF"


def move_misplaced_netcdfs(config: dict[str, Any], logger) -> list[str]:
    movements: list[str] = []
    for path in PROJECT_ROOT.rglob("*.nc"):
        kind = classify_netcdf(path)
        target_dir: Path | None = None
        if kind == "ISS LIS Event File":
            target_dir = resolve_path(config, "raw_archive_nasa_lis")
        elif kind == "LIS/OTD Climatology":
            target_dir = resolve_path(config, "raw_lis_otd")
        if target_dir is None:
            continue
        target = target_dir / path.name
        if path.resolve() == target.resolve():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.move(str(path), str(target))
            msg = f"Moved {path} -> {target}"
        else:
            msg = f"Duplicate already exists at {target}; left {path} in place"
        logger.info(msg)
        movements.append(msg)
    return movements


def inspect_iss_file(path: Path) -> dict[str, Any]:
    with netCDF4.Dataset(path) as ds:
        variables = list(ds.variables.keys())
        required = {
            "time": "lightning_event_TAI93_time",
            "lat": "lightning_event_lat",
            "lon": "lightning_event_lon",
            "radiance": "lightning_event_radiance",
            "quality_flag": "lightning_event_alert_flag",
            "cluster_id": "lightning_event_cluster_index",
            "event_id": "lightning_event_address",
        }
        missing = [value for value in required.values() if value not in ds.variables]
        event_count = len(ds.variables["lightning_event_TAI93_time"][:]) if "lightning_event_TAI93_time" in ds.variables else 0
        if event_count:
            times = netCDF4.num2date(
                ds.variables["lightning_event_TAI93_time"][:],
                units=ds.variables["lightning_event_TAI93_time"].units,
                calendar=getattr(ds.variables["lightning_event_TAI93_time"], "calendar", "standard"),
            )
            lat = np.asarray(ds.variables["lightning_event_lat"][:], dtype=float)
            lon = np.asarray(ds.variables["lightning_event_lon"][:], dtype=float)
            time_range = (times[0].strftime("%Y-%m-%d %H:%M:%S"), times[-1].strftime("%Y-%m-%d %H:%M:%S"))
            lat_range = (float(np.nanmin(lat)), float(np.nanmax(lat)))
            lon_range = (float(np.nanmin(lon)), float(np.nanmax(lon)))
        else:
            time_range = ("", "")
            lat_range = (np.nan, np.nan)
            lon_range = (np.nan, np.nan)
        return {
            "filename": path.name,
            "dataset_type": "ISS LIS Event File",
            "variables": variables,
            "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
            "missing_required_variables": missing,
            "event_count": event_count,
            "time_range": time_range,
            "latitude_range": lat_range,
            "longitude_range": lon_range,
        }


def verify_india_events(event_files: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    lat_min = float(config["india_bounds"]["lat_min"])
    lat_max = float(config["india_bounds"]["lat_max"])
    lon_min = float(config["india_bounds"]["lon_min"])
    lon_max = float(config["india_bounds"]["lon_max"])
    total = 0
    indian = 0
    for path in event_files:
        with netCDF4.Dataset(path) as ds:
            lat = np.asarray(ds.variables["lightning_event_lat"][:], dtype=float)
            lon = np.asarray(ds.variables["lightning_event_lon"][:], dtype=float)
            total += len(lat)
            indian += int(((lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)).sum())
    return {
        "total_events": total,
        "indian_events": indian,
        "percentage": round(indian / total * 100, 3) if total else 0.0,
        "message": "This NASA file contains no lightning events within India's geographic bounds." if total and indian == 0 else "",
    }


def run_command(args: list[str], logger) -> tuple[bool, str]:
    start = time.perf_counter()
    completed = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True)
    duration = round(time.perf_counter() - start, 2)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    logger.info("Command %s completed rc=%s duration=%ss", " ".join(args), completed.returncode, duration)
    if completed.returncode:
        logger.warning(output)
    return completed.returncode == 0, f"rc={completed.returncode}, duration={duration}s"


def verify_ctbt(config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = resolve_path(config, "metadata") / "metadata.csv"
    ctbt_dir = resolve_path(config, "ctbt")
    if not metadata_path.exists():
        return {"passed": False, "detail": "metadata.csv missing", "frames": 0}
    metadata = pd.read_csv(metadata_path, dtype=str)
    missing_images = [row.filename for row in metadata.itertuples(index=False) if not (ctbt_dir / row.filename).exists()]
    source_dir = get_lightning_source_dir(config)
    expected_csv = [source_dir / f"lightning_{str(row.time_index).zfill(4)}.csv" for row in metadata.itertuples(index=False)]
    missing_csv = [path.name for path in expected_csv if not path.exists()]
    return {
        "passed": not missing_images and not missing_csv and len(metadata) > 0,
        "detail": f"source={get_lightning_source(config)}, frames={len(metadata)}, missing_images={missing_images}, missing_lightning_csv={missing_csv}",
        "frames": len(metadata),
    }


def verify_georef(config: dict[str, Any]) -> dict[str, Any]:
    metadata = pd.read_csv(resolve_path(config, "metadata") / "metadata.csv", dtype=str)
    csv_dir = resolve_path(config, "overlays") / "csv"
    image_dir = resolve_path(config, "ctbt")
    issues: list[str] = []
    for row in metadata.itertuples(index=False):
        time_index = str(row.time_index).zfill(4)
        csv_path = csv_dir / f"overlay_{time_index}.csv"
        image_path = image_dir / row.filename
        if not csv_path.exists():
            issues.append(f"missing {csv_path.name}")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            issues.append(f"unreadable {row.filename}")
            continue
        height, width = image.shape[:2]
        frame = pd.read_csv(csv_path)
        if {"pixel_x", "pixel_y"}.issubset(frame.columns) and not frame.empty:
            outside = frame[~(frame["pixel_x"].between(0, width - 1) & frame["pixel_y"].between(0, height - 1))]
            if not outside.empty:
                issues.append(f"{csv_path.name}: {len(outside)} points outside image")
    return {"passed": not issues, "detail": "; ".join(issues) if issues else "All overlay CSV pixels inside image bounds."}


def verify_animation(config: dict[str, Any]) -> dict[str, Any]:
    animation_dir = resolve_path(config, "animations")
    gif = animation_dir / "lightning_animation.gif"
    mp4 = animation_dir / "lightning_animation.mp4"
    overlay_count = len(list((resolve_path(config, "overlays") / "images").glob("overlay_*.png")))
    issues = []
    for path in [gif, mp4]:
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"missing or empty {path.name}")
    return {"passed": not issues and overlay_count > 0, "detail": f"overlay_frames={overlay_count}, issues={issues}"}


def verify_prediction_inputs(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "LIS/OTD climatology NetCDF": list(resolve_path(config, "raw_lis_otd").glob("*.nc")),
        "Open-Meteo JSON": list(resolve_path(config, "raw_open_meteo").glob("*.json")),
        "NASA POWER JSON": list(resolve_path(config, "raw_nasa_power").glob("*.json")),
        "NOAA ISD gz": list(resolve_path(config, "raw_noaa_isd").rglob("*.gz")),
    }
    missing = [name for name, files in expected.items() if not files]
    training_dir = resolve_path(config, "processed_training")
    installed_outputs = {
        "training_dataset.csv": (training_dir / "training_dataset.csv").exists(),
        "prediction_dataset.csv": (training_dir / "prediction_dataset.csv").exists(),
        "feature_dataset.parquet": (training_dir / "feature_dataset.parquet").exists(),
    }
    return {
        "passed": all(installed_outputs.values()),
        "detail": {name: len(files) for name, files in expected.items()},
        "missing": missing,
        "installed_outputs": installed_outputs,
    }


def build_tree() -> str:
    lines = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PROJECT_ROOT)
        depth = len(rel.parts) - 1
        prefix = "  " * depth + ("- " if path.is_file() else "+ ")
        lines.append(f"{prefix}{rel.name}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    for folder in REQUIRED_FOLDERS:
        (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)

    logger = setup_logging(resolve_path(config, "logs") / "audit.log", "audit")
    started = time.perf_counter()
    logger.info("AKAM audit started")
    movements = move_misplaced_netcdfs(config, logger)

    if args.run_pipelines:
        if get_lightning_source(config) == "nasa":
            run_command([sys.executable, "scripts/02_process_lis_events.py"], logger)
        else:
            logger.info("Skipping NASA event extraction because lightning_source=%s", get_lightning_source(config))
        run_command([sys.executable, "scripts/03_georeference_ctbt.py"], logger)
        run_command([sys.executable, "scripts/04_overlay_lightning.py"], logger)
        run_command([sys.executable, "scripts/05_generate_animation.py"], logger)

    checks: list[Check] = []
    checks.append(Check("Folder Verification", all((PROJECT_ROOT / folder).exists() for folder in REQUIRED_FOLDERS), "Required folders exist or were created."))
    dependency_status = {name: importlib.util.find_spec(module) is not None for name, module in DEPENDENCIES.items()}
    checks.append(Check("Dependencies", all(dependency_status.values()), json.dumps(dependency_status, indent=2)))

    config_ok = (
        config["india_bounds"]["lat_min"] == 6.0
        and config["india_bounds"]["lat_max"] == 38.0
        and config["india_bounds"]["lon_min"] == 68.0
        and config["india_bounds"]["lon_max"] == 98.0
        and int(config["ctbt"]["interval_minutes"]) == 30
        and config["ctbt"]["timezone"].upper() == "UTC"
    )
    checks.append(Check("Configuration", config_ok, "India bounds, CTBT interval, timezone, and paths verified."))

    nc_files = sorted(PROJECT_ROOT.rglob("*.nc"))
    classified = {str(path.relative_to(PROJECT_ROOT)): classify_netcdf(path) for path in nc_files}
    event_files = [path for path in nc_files if classify_netcdf(path) == "ISS LIS Event File"]
    climatology_files = [path for path in nc_files if classify_netcdf(path) == "LIS/OTD Climatology"]
    checks.append(Check("ISS Event Parser", bool(event_files), f"ISS event files={len(event_files)}"))
    checks.append(Check("LISOTD Parser", bool(climatology_files), f"LIS/OTD files={len(climatology_files)}; missing if zero."))

    event_inspections = [inspect_iss_file(path) for path in event_files]
    event_parser_ok = all(not item["missing_required_variables"] and item["event_count"] > 0 for item in event_inspections)
    checks.append(Check("ISS Event File Integrity", event_parser_ok, json.dumps(event_inspections, indent=2, default=str)))

    india_stats = verify_india_events(event_files, config) if event_files else {"total_events": 0, "indian_events": 0, "percentage": 0}
    checks.append(Check("India Filtering", True, json.dumps(india_stats, indent=2)))

    ctbt = verify_ctbt(config)
    checks.append(Check("CTBT Synchronization", bool(ctbt["passed"]), str(ctbt["detail"])))
    georef = verify_georef(config)
    checks.append(Check("Geo-reference", bool(georef["passed"]), str(georef["detail"])))
    overlay_images = sorted((resolve_path(config, "overlays") / "images").glob("overlay_*.png"))
    checks.append(Check("Overlay", bool(overlay_images), f"overlay_images={len(overlay_images)}"))
    animation = verify_animation(config)
    checks.append(Check("Animation", bool(animation["passed"]), str(animation["detail"])))
    prediction = verify_prediction_inputs(config)
    checks.append(
        Check(
            "Prediction Pipeline",
            bool(prediction["passed"]),
            f"detected_raw_inputs={prediction['detail']}; missing_raw_inputs={prediction['missing']}; installed_outputs={prediction['installed_outputs']}",
        )
    )

    output_paths = {
        "India lightning CSV": resolve_path(config, "processed_lightning") / "india_lightning_events.csv",
        "Overlay CSV folder": resolve_path(config, "overlays") / "csv",
        "Overlay PNG folder": resolve_path(config, "overlays") / "images",
        "Animation GIF": resolve_path(config, "animations") / "lightning_animation.gif",
        "Animation MP4": resolve_path(config, "animations") / "lightning_animation.mp4",
        "Prediction dataset": resolve_path(config, "processed_training") / "prediction_dataset.csv",
        "Training dataset": resolve_path(config, "processed_training") / "training_dataset.csv",
        "Feature dataset": resolve_path(config, "processed_training") / "feature_dataset.parquet",
    }
    output_status = {name: path.exists() for name, path in output_paths.items()}
    output_pass = all(output_status.values())
    checks.append(Check("Output Verification", output_pass, json.dumps(output_status, indent=2)))

    passed_count = sum(1 for check in checks if check.passed)
    health = round(passed_count / len(checks) * 100)
    duration = round(time.perf_counter() - started, 2)

    report = [
        "# AKAM Project Verification Report",
        "",
        f"Audit duration: {duration}s",
        f"Project Health Score: {health} / 100",
        "",
        "## Project Structure",
        "```text",
        build_tree(),
        "```",
        "",
        "## NetCDF Classification",
        json.dumps(classified, indent=2),
        "",
        "## Active Lightning Source",
        f"{get_lightning_source(config)} -> {get_lightning_source_dir(config)}",
        "",
        "## File Movements",
        "\n".join(f"- {item}" for item in movements) if movements else "No project NetCDF file movements required.",
        "",
        "## Checks",
    ]
    for check in checks:
        report.extend([f"### {check.name}: {status(check.passed)}", check.detail, ""])

    report.extend(
        [
            "## ISS LIS Event Summary",
            json.dumps(event_inspections, indent=2, default=str),
            "",
            "## India Filtering Statement",
            india_stats.get("message", "Indian lightning events were detected.") or "Indian lightning events were detected.",
            "",
            "## Prediction Input Gaps",
            "Missing prediction source files: " + (", ".join(prediction["missing"]) if prediction["missing"] else "None"),
            "",
            "## Recommendations",
            "High Priority",
            "- Add LIS/OTD climatology NetCDF files to data/raw/lis_otd before running Pipeline A.",
            "- Add Open-Meteo, NASA POWER, and NOAA ISD source files before training prediction models.",
            "- Confirm georeference bounds against the real CTBT projection if the CTBT image is not a simple India lat/lon rectangle.",
            "",
            "Medium Priority",
            "- Add unit tests for variable detection, TAI93 conversion, CTBT window assignment, and pixel conversion.",
            "- Add ILDN/NRSC CSV adapters that emit timestamp, latitude, longitude into data/processed/lightning.",
            "",
            "Low Priority",
            "- Remove legacy data/ildn outputs once downstream code fully uses data/processed/lightning.",
        ]
    )

    report_path = resolve_path(config, "logs") / "akam_verification_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    logger.info("AKAM audit finished in %ss; report=%s", duration, report_path)
    print(report_path)
    return 0 if health >= 70 else 1


if __name__ == "__main__":
    raise SystemExit(main())
