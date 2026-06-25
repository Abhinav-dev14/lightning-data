"""Organize IMD CTBT satellite images for ILDN lightning overlays.

The script creates the project data folders, converts chronologically ordered
source images to PNG files named by 30-minute offsets, writes metadata, and
validates the generated dataset.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd


EXPECTED_IMAGE_COUNT = 16
TIME_STEP_MINUTES = 30
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DATA_SUBDIRECTORIES = (
    "ctbt",
    "ildn",
    "overlays",
    "animations",
    "logs",
    "metadata",
)


@dataclass(frozen=True)
class OrganizedImage:
    source_path: Path
    output_path: Path
    time_index: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize IMD CTBT images into a validated PNG time series."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"c:\Users\Abhinav Mehta\Downloads\akactbt"),
        help="Directory containing the downloaded CTBT source images.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root where the data/ directory should be created.",
    )
    parser.add_argument(
        "--move-source",
        action="store_true",
        help=(
            "Delete source files after successful PNG conversion. By default the "
            "originals are preserved."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing ctbt_*.png outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail if an output PNG already exists.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> list[object]:
    """Sort filenames in a human-friendly way while preserving download order."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def time_index_for_position(position: int) -> str:
    minutes = position * TIME_STEP_MINUTES
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}{mins:02d}"


def create_project_structure(project_root: Path) -> dict[str, Path]:
    data_root = project_root / "data"
    folders = {"data": data_root}
    for name in DATA_SUBDIRECTORIES:
        folders[name] = data_root / name
        folders[name].mkdir(parents=True, exist_ok=True)
    return folders


def configure_logging(log_dir: Path) -> Path:
    log_path = log_dir / "ctbt_organizer.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def discover_source_images(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    images = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_sort_key)


def convert_to_png(source_path: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"OpenCV could not write PNG: {output_path}")


def organize_images(
    source_images: Iterable[Path],
    ctbt_dir: Path,
    overwrite: bool,
    move_source: bool,
) -> list[OrganizedImage]:
    organized: list[OrganizedImage] = []
    source_images = list(source_images)

    if len(source_images) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} source images, found {len(source_images)}."
        )

    for position, source_path in enumerate(source_images):
        time_index = time_index_for_position(position)
        output_path = ctbt_dir / f"ctbt_{time_index}.png"
        logging.info("Converting %s -> %s", source_path.name, output_path.name)
        convert_to_png(source_path, output_path, overwrite=overwrite)
        organized.append(
            OrganizedImage(
                source_path=source_path,
                output_path=output_path,
                time_index=time_index,
            )
        )

    if move_source:
        for item in organized:
            item.source_path.unlink()

    return organized


def write_metadata(organized: list[OrganizedImage], metadata_dir: Path) -> Path:
    metadata_path = metadata_dir / "metadata.csv"
    rows = [
        {"filename": item.output_path.name, "time_index": item.time_index}
        for item in organized
    ]
    pd.DataFrame(rows, columns=["filename", "time_index"]).to_csv(
        metadata_path, index=False
    )
    return metadata_path


def validate_outputs(ctbt_dir: Path, metadata_path: Path) -> list[str]:
    errors: list[str] = []
    png_files = sorted(ctbt_dir.glob("ctbt_*.png"), key=natural_sort_key)
    filenames = [path.name for path in png_files]
    expected_names = [
        f"ctbt_{time_index_for_position(position)}.png"
        for position in range(EXPECTED_IMAGE_COUNT)
    ]

    if len(png_files) != EXPECTED_IMAGE_COUNT:
        errors.append(f"Expected 16 PNG files, found {len(png_files)}.")

    if len(filenames) != len(set(filenames)):
        errors.append("Duplicate filenames found in data/ctbt.")

    non_png = [path.name for path in ctbt_dir.iterdir() if path.is_file() and path.suffix.lower() != ".png"]
    if non_png:
        errors.append(f"Non-PNG files found in data/ctbt: {', '.join(non_png)}")

    if filenames != expected_names:
        errors.append(
            "PNG filenames are not in the expected 30-minute chronological sequence."
        )

    for path in png_files:
        if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
            errors.append(f"Unreadable PNG output: {path.name}")

    metadata = pd.read_csv(metadata_path, dtype=str)
    expected_metadata = pd.DataFrame(
        {
            "filename": expected_names,
            "time_index": [
                time_index_for_position(position)
                for position in range(EXPECTED_IMAGE_COUNT)
            ],
        }
    )
    if not metadata.equals(expected_metadata):
        errors.append("metadata.csv does not match the expected image sequence.")

    return errors


def build_tree(root: Path) -> str:
    lines = ["data/"]
    for folder in DATA_SUBDIRECTORIES:
        lines.append(f"+-- {folder}/")
    return "\n".join(lines)


def write_validation_report(
    log_dir: Path,
    organized: list[OrganizedImage],
    metadata_path: Path,
    errors: list[str],
) -> Path:
    report_path = log_dir / "validation_report.txt"
    metadata = pd.read_csv(metadata_path, dtype=str)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Final folder structure\n")
        handle.write(build_tree(log_dir.parents[1]))
        handle.write("\n\nRenamed file list\n")
        for item in organized:
            handle.write(f"{item.output_path.name}\n")
        handle.write("\nMetadata table\n")
        handle.write(metadata.to_string(index=False))
        handle.write("\n\nErrors found\n")
        handle.write("\n".join(errors) if errors else "None")
        handle.write("\n")
    return report_path


def print_output(
    organized: list[OrganizedImage],
    metadata_path: Path,
    errors: list[str],
    report_path: Path,
) -> None:
    metadata = pd.read_csv(metadata_path, dtype=str)

    print("\n1. Final folder structure")
    print(build_tree(metadata_path.parents[1]))

    print("\n2. Renamed file list")
    for item in organized:
        print(item.output_path.name)

    print("\n3. Metadata table")
    print(metadata.to_string(index=False))

    print("\n4. Any errors found")
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("None")

    print(f"\nValidation report: {report_path}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    folders = create_project_structure(project_root)
    log_path = configure_logging(folders["logs"])
    logging.info("Log file: %s", log_path)

    try:
        source_images = discover_source_images(args.source_dir.resolve())
        organized = organize_images(
            source_images=source_images,
            ctbt_dir=folders["ctbt"],
            overwrite=args.overwrite,
            move_source=args.move_source,
        )
        metadata_path = write_metadata(organized, folders["metadata"])
        errors = validate_outputs(folders["ctbt"], metadata_path)
        report_path = write_validation_report(
            folders["logs"], organized, metadata_path, errors
        )
        print_output(organized, metadata_path, errors, report_path)
        return 1 if errors else 0
    except Exception as exc:
        logging.exception("CTBT organization failed")
        report_path = folders["logs"] / "validation_report.txt"
        report_path.write_text(f"Errors found\n{exc}\n", encoding="utf-8")
        print(f"\nERROR: {exc}")
        print(f"See log: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
