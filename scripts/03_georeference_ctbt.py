#!/usr/bin/env python3
"""Convert lightning latitude/longitude CSVs into CTBT pixel-coordinate CSVs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from common import ensure_project_dirs, get_lightning_source, get_lightning_source_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def latlon_to_pixel(lat: float, lon: float, width: int, height: int, bounds: dict) -> tuple[int, int]:
    x_ratio = (lon - float(bounds["lon_min"])) / (float(bounds["lon_max"]) - float(bounds["lon_min"]))
    y_ratio = (float(bounds["lat_max"]) - lat) / (float(bounds["lat_max"]) - float(bounds["lat_min"]))
    x = int(round(x_ratio * (width - 1)))
    y = int(round(y_ratio * (height - 1)))
    return x, y


def georeference_frame(lightning_csv: Path, ctbt_image: Path, output_csv: Path, bounds: dict) -> int:
    image = cv2.imread(str(ctbt_image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read CTBT image: {ctbt_image}")
    height, width = image.shape[:2]
    frame = pd.read_csv(lightning_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame(columns=["timestamp", "latitude", "longitude", "pixel_x", "pixel_y"]).to_csv(output_csv, index=False)
        return 0

    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["latitude", "longitude"]).copy()
    pixels = frame.apply(
        lambda row: latlon_to_pixel(float(row["latitude"]), float(row["longitude"]), width, height, bounds),
        axis=1,
    )
    frame["pixel_x"] = [item[0] for item in pixels]
    frame["pixel_y"] = [item[1] for item in pixels]
    frame = frame[
        frame["pixel_x"].between(0, width - 1) & frame["pixel_y"].between(0, height - 1)
    ].copy()
    frame.to_csv(output_csv, index=False)
    return len(frame)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(resolve_path(config, "logs") / "georeference.log", "georeference")
    start = time.perf_counter()
    logger.info("Georeference started")
    lightning_dir = get_lightning_source_dir(config)
    logger.info("Lightning source=%s directory=%s", get_lightning_source(config), lightning_dir)
    output_dir = resolve_path(config, "overlays") / "csv"
    ctbt_dir = resolve_path(config, "ctbt")
    metadata = pd.read_csv(resolve_path(config, "metadata") / "metadata.csv", dtype=str)
    bounds = config["ctbt"]["georef_bounds"]

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Georeference", unit="frame"):
        time_index = str(row.time_index).zfill(4)
        lightning_csv = lightning_dir / f"lightning_{time_index}.csv"
        ctbt_image = ctbt_dir / row.filename
        output_csv = output_dir / f"overlay_{time_index}.csv"
        if not lightning_csv.exists():
            logger.warning("Missing lightning CSV: %s", lightning_csv)
            continue
        count = georeference_frame(lightning_csv, ctbt_image, output_csv, bounds)
        logger.info("Wrote %s rows=%s", output_csv, count)
    logger.info("Georeference finished duration_seconds=%.2f", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
