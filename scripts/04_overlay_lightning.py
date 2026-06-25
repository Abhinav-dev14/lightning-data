#!/usr/bin/env python3
"""Draw georeferenced lightning points on CTBT images."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from common import ensure_project_dirs, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--radius", type=int, default=5)
    return parser.parse_args()


def draw_overlay(ctbt_image: Path, overlay_csv: Path, output_image: Path, radius: int) -> int:
    image = cv2.imread(str(ctbt_image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read CTBT image: {ctbt_image}")
    frame = pd.read_csv(overlay_csv)
    count = 0
    if not frame.empty:
        for row in frame.itertuples(index=False):
            x = int(getattr(row, "pixel_x"))
            y = int(getattr(row, "pixel_y"))
            cv2.circle(image, (x, y), radius, (0, 255, 255), thickness=-1)
            cv2.circle(image, (x, y), radius + 2, (0, 0, 255), thickness=1)
            count += 1
    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), image)
    return count


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(resolve_path(config, "logs") / "overlay.log", "overlay")
    start = time.perf_counter()
    logger.info("Overlay generation started")
    metadata = pd.read_csv(resolve_path(config, "metadata") / "metadata.csv", dtype=str)
    ctbt_dir = resolve_path(config, "ctbt")
    overlay_csv_dir = resolve_path(config, "overlays") / "csv"
    output_dir = resolve_path(config, "overlays") / "images"

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Overlay", unit="frame"):
        time_index = str(row.time_index).zfill(4)
        overlay_csv = overlay_csv_dir / f"overlay_{time_index}.csv"
        if not overlay_csv.exists():
            logger.warning("Missing overlay CSV: %s", overlay_csv)
            continue
        output_image = output_dir / f"overlay_{time_index}.png"
        count = draw_overlay(ctbt_dir / row.filename, overlay_csv, output_image, args.radius)
        logger.info("Wrote %s lightning_points=%s", output_image, count)
    logger.info("Overlay generation finished duration_seconds=%.2f", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
