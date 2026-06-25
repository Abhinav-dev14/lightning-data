#!/usr/bin/env python3
"""Create GIF and MP4 time-lapse animations from CTBT overlay images."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from common import ensure_project_dirs, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser.parse_args()


def write_mp4(images: list[Path], output_path: Path, fps: float) -> None:
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Cannot read first frame: {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(resolve_path(config, "logs") / "animation.log", "animation")
    start = time.perf_counter()
    logger.info("Animation generation started")
    image_dir = resolve_path(config, "overlays") / "images"
    output_dir = resolve_path(config, "animations")
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(image_dir.glob("overlay_*.png"))
    if not images:
        logger.error("No overlay images found in %s", image_dir)
        return 1

    gif_path = output_dir / "lightning_animation.gif"
    mp4_path = output_dir / "lightning_animation.mp4"
    frames = [Image.open(path).convert("P") for path in tqdm(images, desc="GIF frames", unit="frame")]
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    write_mp4(images, mp4_path, args.fps)
    logger.info("Wrote %s and %s", gif_path, mp4_path)
    logger.info("Animation generation finished duration_seconds=%.2f", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
