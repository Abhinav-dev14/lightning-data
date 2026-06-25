"""Convert NASA LIS NetCDF lightning events into CTBT-synchronized CSV files."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import PillowWriter
from tqdm import tqdm


INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 98.0
CTBT_WINDOW_MINUTES = 30
REQUIRED_COLUMNS = ["timestamp", "latitude", "longitude"]
OPTIONAL_FIELD_RULES = {
    "radiance": {
        "include": ("radiance",),
        "prefer": ("lightning_event",),
        "exclude": ("background", "bg_", "summary", "viewtime", "one_second"),
    },
    "quality_flag": {
        "include": ("flag",),
        "prefer": ("lightning_event", "alert"),
        "exclude": ("background", "summary", "viewtime", "one_second"),
    },
    "energy": {
        "include": ("energy",),
        "prefer": ("lightning_event",),
        "exclude": ("summary", "viewtime", "one_second"),
    },
    "confidence": {
        "include": ("cluster", "index"),
        "prefer": ("lightning_event", "probability"),
        "exclude": ("summary", "viewtime", "one_second"),
    },
}


@dataclass(frozen=True)
class VariableInfo:
    """Metadata for one NetCDF variable."""

    path: str
    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        pieces = [self.path, self.name]
        for key in ("standard_name", "long_name", "units", "description"):
            value = self.attributes.get(key)
            if value is not None:
                pieces.append(str(value))
        return " ".join(pieces).lower()


@dataclass(frozen=True)
class ExtractionResult:
    """Lightning dataframe plus extraction diagnostics."""

    dataframe: pd.DataFrame
    variables: list[VariableInfo]
    selected_variables: dict[str, str]


@dataclass(frozen=True)
class FrameWindow:
    """One CTBT image and its UTC time window."""

    filename: str
    time_index: str
    frame_number: int
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract NASA LIS lightning events and synchronize them with CTBT frames."
    )
    parser.add_argument(
        "netcdf_files",
        nargs="+",
        type=Path,
        help="NASA LIS NetCDF files to process.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="AKAM project root containing data/metadata/metadata.csv.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to CTBT metadata.csv.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="UTC date for CTBT frame windows, formatted YYYY-MM-DD. Defaults to earliest LIS event date.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite generated CSVs and figures. Enabled by default.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail if any generated output already exists.",
    )
    return parser.parse_args()


def create_project_structure(project_root: Path) -> dict[str, Path]:
    data_root = project_root / "data"
    folders = {
        "data": data_root,
        "ildn": data_root / "ildn",
        "overlays": data_root / "overlays",
        "animations": data_root / "animations",
        "logs": data_root / "logs",
        "metadata": data_root / "metadata",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def configure_logging(log_dir: Path) -> Path:
    log_path = log_dir / "extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def import_netcdf4() -> Any:
    try:
        import netCDF4  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NetCDF4 support is required for NASA LIS HDF5/NetCDF4 files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return netCDF4


class LightningSource:
    """Interface for source-specific lightning readers."""

    def extract(self, paths: Iterable[Path]) -> ExtractionResult:
        raise NotImplementedError


class NasaLisNetcdfSource(LightningSource):
    """Read all lightning events from NASA LIS NetCDF files."""

    def __init__(self) -> None:
        self.netcdf4 = import_netcdf4()

    def extract(self, paths: Iterable[Path]) -> ExtractionResult:
        frames: list[pd.DataFrame] = []
        all_variables: list[VariableInfo] = []
        selected: dict[str, str] = {}

        for path in tqdm(list(paths), desc="Reading NetCDF files", unit="file"):
            logging.info("Inspecting NetCDF file: %s", path)
            with self.netcdf4.Dataset(path, "r") as dataset:
                variables = self._discover_variables(dataset)
                all_variables.extend(variables)
                for variable in variables:
                    logging.info(
                        "Variable discovered: %s shape=%s dtype=%s attrs=%s",
                        variable.path,
                        variable.shape,
                        variable.dtype,
                        variable.attributes,
                    )

                field_map = self._select_fields(variables)
                selected.update(field_map)
                logging.info("Selected variables for %s: %s", path.name, field_map)
                frames.append(self._build_dataframe(dataset, field_map, path.name))

        if not frames:
            raise ValueError("No NetCDF files were supplied.")

        dataframe = pd.concat(frames, ignore_index=True)
        return ExtractionResult(dataframe=dataframe, variables=all_variables, selected_variables=selected)

    def _discover_variables(self, dataset: Any) -> list[VariableInfo]:
        discovered: list[VariableInfo] = []

        def walk(group: Any, prefix: str = "") -> None:
            for name, variable in group.variables.items():
                attrs = {
                    attr: self._normalize_attr(variable.getncattr(attr))
                    for attr in variable.ncattrs()
                }
                discovered.append(
                    VariableInfo(
                        path=f"{prefix}/{name}" if prefix else name,
                        name=name,
                        dimensions=tuple(str(dim) for dim in variable.dimensions),
                        shape=tuple(int(size) for size in variable.shape),
                        dtype=str(variable.dtype),
                        attributes=attrs,
                    )
                )
            for group_name, child in group.groups.items():
                walk(child, f"{prefix}/{group_name}" if prefix else group_name)

        walk(dataset)
        return discovered

    def _select_fields(self, variables: list[VariableInfo]) -> dict[str, str]:
        fields = {
            "timestamp": self._best_variable(
                variables,
                include=("time",),
                prefer=("lightning_event", "tai93", "since"),
                exclude=("bounds", "delta", "observe_time", "start", "end"),
            ),
            "latitude": self._best_variable(
                variables,
                include=("lat",),
                prefer=("lightning_event", "standard_name"),
                exclude=("bounds",),
            ),
            "longitude": self._best_variable(
                variables,
                include=("lon",),
                prefer=("lightning_event", "standard_name"),
                exclude=("bounds",),
            ),
        }

        missing = [field for field, value in fields.items() if value is None]
        if missing:
            raise ValueError(f"Could not identify required LIS variable(s): {', '.join(missing)}")

        selected = {field: str(path) for field, path in fields.items()}
        for output_name, rule in OPTIONAL_FIELD_RULES.items():
            path = self._best_variable(
                variables,
                include=rule["include"],
                prefer=rule["prefer"],
                exclude=("bounds", *rule["exclude"]),
            )
            if path:
                selected[output_name] = path
        return selected

    def _best_variable(
        self,
        variables: list[VariableInfo],
        include: tuple[str, ...],
        prefer: tuple[str, ...] = (),
        exclude: tuple[str, ...] = (),
    ) -> str | None:
        best: tuple[int, str] | None = None
        for variable in variables:
            text = variable.searchable_text
            if any(term in text for term in exclude):
                continue
            include_matches = sum(1 for term in include if term in text)
            if include_matches == 0:
                continue
            score = include_matches * 5
            score += sum(4 for term in prefer if term in text)
            name_text = variable.name.lower()
            if all(term in name_text for term in include):
                score += 20
            if variable.shape and len(variable.shape) == 1:
                score += 2
            if "event" in text:
                score += 2
            if "lightning" in text:
                score += 1
            if score <= 0:
                continue
            candidate = (score, variable.path)
            if best is None or candidate > best:
                best = candidate
        return best[1] if best else None

    def _build_dataframe(
        self, dataset: Any, field_map: dict[str, str], source_name: str
    ) -> pd.DataFrame:
        raw_columns: dict[str, np.ndarray] = {}
        expected_length: int | None = None

        for output_name, variable_path in field_map.items():
            variable = self._get_variable(dataset, variable_path)
            values = np.asarray(variable[:]).reshape(-1)
            if expected_length is None and output_name in REQUIRED_COLUMNS:
                expected_length = len(values)
            if expected_length is not None and len(values) != expected_length:
                logging.warning(
                    "Skipping %s from %s because length %s does not match event length %s.",
                    variable_path,
                    source_name,
                    len(values),
                    expected_length,
                )
                continue
            if output_name == "timestamp":
                raw_columns[output_name] = self._convert_time(values, variable)
            else:
                raw_columns[output_name] = values

        dataframe = pd.DataFrame(raw_columns)
        self._add_lis_hierarchy_fields(dataset, dataframe)
        dataframe["source_file"] = source_name
        return dataframe

    def _add_lis_hierarchy_fields(self, dataset: Any, dataframe: pd.DataFrame) -> None:
        """Attach LIS event/group/flash record addresses when available."""
        event_address = self._read_named_1d(
            dataset,
            terms=("lightning", "event", "address"),
            expected_length=len(dataframe),
            exclude=("parent", "child", "grandchild", "greatgrandchild", "summary"),
        )
        event_parent = self._read_named_1d(
            dataset,
            terms=("lightning", "event", "parent", "address"),
            expected_length=len(dataframe),
            exclude=("child", "grandchild", "greatgrandchild", "summary"),
        )
        group_address = self._read_named_1d(
            dataset,
            terms=("lightning", "group", "address"),
            expected_length=None,
            exclude=("parent", "child", "grandchild", "greatgrandchild", "summary"),
        )
        group_parent = self._read_named_1d(
            dataset,
            terms=("lightning", "group", "parent", "address"),
            expected_length=None,
            exclude=("child", "grandchild", "greatgrandchild", "summary"),
        )

        if event_address is not None:
            dataframe["event_id"] = event_address
        if event_parent is not None:
            dataframe["group_id"] = event_parent
        if event_parent is not None and group_address is not None and group_parent is not None:
            lookup = dict(zip(group_address.tolist(), group_parent.tolist(), strict=False))
            dataframe["flash_id"] = [lookup.get(group_id, np.nan) for group_id in event_parent]

    def _read_named_1d(
        self,
        dataset: Any,
        terms: tuple[str, ...],
        expected_length: int | None,
        exclude: tuple[str, ...],
    ) -> np.ndarray | None:
        variables = self._discover_variables(dataset)
        path = self._best_variable(
            variables,
            include=terms,
            prefer=("lightning",),
            exclude=exclude,
        )
        if not path:
            return None
        values = np.asarray(self._get_variable(dataset, path)[:]).reshape(-1)
        if expected_length is not None and len(values) != expected_length:
            return None
        return values

    def _get_variable(self, dataset: Any, path: str) -> Any:
        current = dataset
        parts = path.split("/")
        for group in parts[:-1]:
            current = current.groups[group]
        return current.variables[parts[-1]]

    def _convert_time(self, values: np.ndarray, variable: Any) -> np.ndarray:
        units = getattr(variable, "units", None)
        calendar = getattr(variable, "calendar", "standard")
        if units and "since" in str(units).lower():
            dates = self.netcdf4.num2date(values, units=units, calendar=calendar)
            return np.array([self._format_datetime(item) for item in dates], dtype=object)

        name_text = " ".join(
            [str(getattr(variable, "_name", "")), str(getattr(variable, "long_name", ""))]
        ).lower()
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        has_1993_units = bool(units and "1993" in str(units).lower())
        if "tai93" in name_text or has_1993_units:
            epoch = datetime(1993, 1, 1, tzinfo=timezone.utc)
        elif numeric.dropna().median() > 1_000_000_000:
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            epoch = datetime(1993, 1, 1, tzinfo=timezone.utc)

        converted: list[str | None] = []
        for value in numeric:
            if pd.isna(value):
                converted.append(None)
            else:
                converted.append((epoch + timedelta(seconds=float(value))).strftime("%Y-%m-%d %H:%M:%S"))
        return np.array(converted, dtype=object)

    @staticmethod
    def _format_datetime(value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _normalize_attr(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


class LightningPipeline:
    """Source-independent CTBT synchronization pipeline."""

    def __init__(
        self,
        project_root: Path,
        metadata_path: Path,
        source: LightningSource,
        ctbt_date: str | None,
        overwrite: bool,
    ) -> None:
        self.project_root = project_root
        self.folders = create_project_structure(project_root)
        self.metadata_path = metadata_path
        self.source = source
        self.ctbt_date = ctbt_date
        self.overwrite = overwrite

    def run(self, netcdf_files: list[Path]) -> dict[str, Any]:
        start_time = time.perf_counter()
        extraction = self.source.extract(netcdf_files)
        raw = extraction.dataframe
        cleaned, discard_counts = self._clean_and_filter_india(raw)
        windows = self._load_frame_windows(cleaned)
        matched, per_frame = self._write_frame_csvs(cleaned, windows)
        self._write_overlay_csv(cleaned)
        self._write_training_dataset(matched, windows, per_frame)
        self._create_visualizations(cleaned, per_frame)
        stats = self._build_stats(raw, cleaned, matched, discard_counts, per_frame, start_time)
        self._print_stats(stats)
        return stats

    def _clean_and_filter_india(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        dataframe = raw.copy()
        total = len(dataframe)
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True, errors="coerce")
        dataframe["latitude"] = pd.to_numeric(dataframe["latitude"], errors="coerce")
        dataframe["longitude"] = pd.to_numeric(dataframe["longitude"], errors="coerce")

        invalid_time = dataframe["timestamp"].isna()
        invalid_latlon = dataframe["latitude"].isna() | dataframe["longitude"].isna()
        invalid_coords = ~dataframe["latitude"].between(-90, 90) | ~dataframe["longitude"].between(-180, 180)
        outside_india = ~dataframe["latitude"].between(INDIA_LAT_MIN, INDIA_LAT_MAX) | ~dataframe["longitude"].between(INDIA_LON_MIN, INDIA_LON_MAX)

        keep = ~(invalid_time | invalid_latlon | invalid_coords | outside_india)
        discard_counts = {
            "invalid_timestamp": int(invalid_time.sum()),
            "nan_or_non_numeric_coordinates": int((invalid_latlon & ~invalid_time).sum()),
            "invalid_coordinate_range": int((invalid_coords & ~(invalid_time | invalid_latlon)).sum()),
            "outside_india_bounds": int((outside_india & ~(invalid_time | invalid_latlon | invalid_coords)).sum()),
        }
        discard_counts["total_discarded"] = total - int(keep.sum())

        cleaned = dataframe.loc[keep].copy()
        cleaned.sort_values("timestamp", inplace=True)
        cleaned["timestamp"] = cleaned["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return cleaned.reset_index(drop=True), discard_counts

    def _load_frame_windows(self, india_events: pd.DataFrame) -> list[FrameWindow]:
        metadata = pd.read_csv(self.metadata_path, dtype=str)
        required = {"filename", "time_index"}
        if not required.issubset(metadata.columns):
            raise ValueError(f"{self.metadata_path} must contain columns: filename,time_index")

        if self.ctbt_date:
            date = datetime.strptime(self.ctbt_date, "%Y-%m-%d").date()
        elif not india_events.empty:
            date = pd.to_datetime(india_events["timestamp"], utc=True).min().date()
        else:
            date = datetime.now(timezone.utc).date()
            logging.warning("No Indian events found; using today's UTC date for empty frame windows.")

        windows: list[FrameWindow] = []
        for frame_number, row in metadata.reset_index(drop=True).iterrows():
            time_index = str(row["time_index"]).zfill(4)
            hour = int(time_index[:2])
            minute = int(time_index[2:])
            start = pd.Timestamp(datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc)) + pd.Timedelta(hours=hour, minutes=minute)
            windows.append(
                FrameWindow(
                    filename=str(row["filename"]),
                    time_index=time_index,
                    frame_number=int(frame_number),
                    start=start,
                    end=start + pd.Timedelta(minutes=CTBT_WINDOW_MINUTES),
                )
            )
        return windows

    def _write_frame_csvs(
        self, events: pd.DataFrame, windows: list[FrameWindow]
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        event_times = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        matched_rows: list[pd.DataFrame] = []
        per_frame: dict[str, int] = {}

        for window in tqdm(windows, desc="Writing CTBT CSVs", unit="frame"):
            mask = (event_times >= window.start) & (event_times < window.end)
            frame_events = events.loc[mask].copy()
            frame_events["ctbt_image"] = window.filename
            frame_events["time_window"] = f"{window.start.strftime('%H:%M')}-{window.end.strftime('%H:%M')}"
            frame_events["frame_number"] = window.frame_number
            output_path = self.folders["ildn"] / f"lightning_{window.time_index}.csv"
            self._write_csv(frame_events, output_path)
            per_frame[window.time_index] = len(frame_events)
            if not frame_events.empty:
                matched_rows.append(frame_events)
            logging.info("CSV created: %s rows=%s", output_path, len(frame_events))

        matched = pd.concat(matched_rows, ignore_index=True) if matched_rows else events.iloc[0:0].copy()
        return matched, per_frame

    def _write_overlay_csv(self, events: pd.DataFrame) -> None:
        overlay_path = self.folders["ildn"] / "lightning_overlay_ready.csv"
        self._write_csv(events[REQUIRED_COLUMNS], overlay_path)
        logging.info("Overlay-ready CSV created: %s rows=%s", overlay_path, len(events))

    def _write_training_dataset(
        self,
        matched: pd.DataFrame,
        windows: list[FrameWindow],
        per_frame: dict[str, int],
    ) -> None:
        training_path = self.folders["metadata"] / "lightning_training.csv"
        if matched.empty:
            columns = [
                "timestamp",
                "hour",
                "minute",
                "latitude",
                "longitude",
                "flash_count",
                "time_window",
                "ctbt_image",
                "frame_number",
                "future_frame",
            ]
            self._write_csv(pd.DataFrame(columns=columns), training_path)
            return

        frame_lookup = {window.frame_number: window for window in windows}
        timestamp = pd.to_datetime(matched["timestamp"], utc=True)
        training = pd.DataFrame(
            {
                "timestamp": matched["timestamp"],
                "hour": timestamp.dt.hour,
                "minute": timestamp.dt.minute,
                "latitude": matched["latitude"],
                "longitude": matched["longitude"],
                "flash_count": matched["time_window"].map(matched.groupby("time_window").size()),
                "time_window": matched["time_window"],
                "ctbt_image": matched["ctbt_image"],
                "frame_number": matched["frame_number"],
            }
        )
        training["future_frame"] = training["frame_number"].apply(
            lambda number: frame_lookup[number + 1].filename if number + 1 in frame_lookup else ""
        )
        self._write_csv(training, training_path)
        logging.info("Training dataset created: %s rows=%s", training_path, len(training))

    def _create_visualizations(self, events: pd.DataFrame, per_frame: dict[str, int]) -> None:
        map_path = self.folders["overlays"] / "india_lightning_map.png"
        density_path = self.folders["overlays"] / "lightning_density.png"
        timeline_path = self.folders["overlays"] / "lightning_timeline.png"
        animation_path = self.folders["animations"] / "lightning_animation.gif"

        if events.empty:
            self._empty_plot(map_path, "No Indian lightning events")
            self._empty_plot(density_path, "No Indian lightning events")
            self._empty_plot(timeline_path, "No Indian lightning events")
            self._empty_gif(animation_path, "No Indian lightning events")
            return

        plt.figure(figsize=(8, 9))
        plt.scatter(events["longitude"], events["latitude"], s=10, alpha=0.65)
        plt.xlim(INDIA_LON_MIN, INDIA_LON_MAX)
        plt.ylim(INDIA_LAT_MIN, INDIA_LAT_MAX)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("India Lightning Events")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        self._save_current_figure(map_path)

        plt.figure(figsize=(8, 9))
        plt.hist2d(
            events["longitude"],
            events["latitude"],
            bins=[40, 40],
            range=[[INDIA_LON_MIN, INDIA_LON_MAX], [INDIA_LAT_MIN, INDIA_LAT_MAX]],
            cmap="inferno",
        )
        plt.colorbar(label="Event count")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Lightning Density")
        plt.tight_layout()
        self._save_current_figure(density_path)

        frame = pd.DataFrame({"time_index": list(per_frame.keys()), "events": list(per_frame.values())})
        plt.figure(figsize=(10, 4))
        plt.bar(frame["time_index"], frame["events"])
        plt.xlabel("CTBT frame")
        plt.ylabel("Lightning events")
        plt.title("Lightning Timeline")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_current_figure(timeline_path)

        self._create_animation(events, animation_path)

    def _create_animation(self, events: pd.DataFrame, output_path: Path) -> None:
        times = pd.to_datetime(events["timestamp"], utc=True)
        ordered = events.assign(_time=times).sort_values("_time")
        chunks = np.array_split(ordered, max(1, min(20, len(ordered))))
        fig, ax = plt.subplots(figsize=(8, 9))

        def draw(index: int) -> list[Any]:
            ax.clear()
            chunk = pd.concat(chunks[: index + 1])
            ax.scatter(chunk["longitude"], chunk["latitude"], s=10, alpha=0.65)
            ax.set_xlim(INDIA_LON_MIN, INDIA_LON_MAX)
            ax.set_ylim(INDIA_LAT_MIN, INDIA_LAT_MAX)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"Lightning accumulation: {chunk['_time'].max().strftime('%H:%M:%S UTC')}")
            ax.grid(True, alpha=0.25)
            return []

        from matplotlib.animation import FuncAnimation

        animation = FuncAnimation(fig, draw, frames=len(chunks), interval=300, blit=False)
        self._ensure_can_write(output_path)
        animation.save(output_path, writer=PillowWriter(fps=3))
        plt.close(fig)
        logging.info("Animation created: %s", output_path)

    def _build_stats(
        self,
        raw: pd.DataFrame,
        india: pd.DataFrame,
        matched: pd.DataFrame,
        discard_counts: dict[str, int],
        per_frame: dict[str, int],
        start_time: float,
    ) -> dict[str, Any]:
        total = len(raw)
        indian = len(india)
        matched_count = len(matched)
        return {
            "processing_seconds": round(time.perf_counter() - start_time, 2),
            "total_events": total,
            "indian_events": indian,
            "events_matched_to_ctbt": matched_count,
            "events_discarded": discard_counts["total_discarded"],
            "discard_reason": discard_counts,
            "coverage_percentage": round((matched_count / indian * 100.0), 2) if indian else 0.0,
            "earliest_timestamp": india["timestamp"].min() if indian else "",
            "latest_timestamp": india["timestamp"].max() if indian else "",
            "latitude_range": (
                float(india["latitude"].min()) if indian else math.nan,
                float(india["latitude"].max()) if indian else math.nan,
            ),
            "longitude_range": (
                float(india["longitude"].min()) if indian else math.nan,
                float(india["longitude"].max()) if indian else math.nan,
            ),
            "events_per_ctbt_frame": per_frame,
        }

    def _print_stats(self, stats: dict[str, Any]) -> None:
        print("\nLightning extraction statistics")
        for key, value in stats.items():
            if key == "events_per_ctbt_frame":
                print("Events per CTBT frame:")
                for frame, count in value.items():
                    print(f"  {frame}: {count}")
            else:
                print(f"{key}: {value}")
        logging.info("Processing complete: %s", stats)

    def _write_csv(self, dataframe: pd.DataFrame, path: Path) -> None:
        self._ensure_can_write(path)
        dataframe.to_csv(path, index=False)

    def _ensure_can_write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {path}")

    def _save_current_figure(self, path: Path) -> None:
        self._ensure_can_write(path)
        plt.savefig(path, dpi=180)
        plt.close()
        logging.info("Figure created: %s", path)

    def _empty_plot(self, path: Path, title: str) -> None:
        plt.figure(figsize=(6, 4))
        plt.title(title)
        plt.axis("off")
        self._save_current_figure(path)

    def _empty_gif(self, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title)
        ax.axis("off")
        self._ensure_can_write(path)
        from matplotlib.animation import FuncAnimation

        animation = FuncAnimation(fig, lambda _: [], frames=1, interval=500, blit=False)
        animation.save(path, writer=PillowWriter(fps=1))
        plt.close(fig)
        logging.info("Animation created: %s", path)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    folders = create_project_structure(project_root)
    log_path = configure_logging(folders["logs"])
    logging.info("Log file: %s", log_path)

    metadata_path = args.metadata.resolve() if args.metadata else folders["metadata"] / "metadata.csv"
    try:
        source = NasaLisNetcdfSource()
        pipeline = LightningPipeline(
            project_root=project_root,
            metadata_path=metadata_path,
            source=source,
            ctbt_date=args.date,
            overwrite=args.overwrite,
        )
        pipeline.run([path.resolve() for path in args.netcdf_files])
        return 0
    except Exception as exc:
        logging.exception("LIS lightning extraction failed")
        print(f"\nERROR: {exc}")
        print(f"See log: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
