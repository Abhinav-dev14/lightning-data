#!/usr/bin/env python3
"""Build prediction-ready thunderstorm datasets from climatology and weather data."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    PROJECT_ROOT,
    ensure_project_dirs,
    get_india_bounds,
    load_config,
    resolve_path,
    setup_logging,
)


INDIA_STATES = {
    "Andhra Pradesh": (12.6, 19.1, 76.7, 84.7),
    "Arunachal Pradesh": (26.7, 29.5, 91.5, 97.4),
    "Assam": (24.1, 27.9, 89.7, 96.0),
    "Bihar": (24.3, 27.5, 83.3, 88.3),
    "Chhattisgarh": (17.8, 24.1, 80.2, 84.4),
    "Goa": (14.9, 15.8, 73.7, 74.4),
    "Gujarat": (20.1, 24.7, 68.2, 74.5),
    "Haryana": (27.7, 30.9, 74.5, 77.6),
    "Himachal Pradesh": (30.4, 33.2, 75.6, 79.0),
    "Jharkhand": (21.9, 25.3, 83.3, 87.9),
    "Karnataka": (11.6, 18.5, 74.1, 78.6),
    "Kerala": (8.1, 12.8, 74.9, 77.4),
    "Madhya Pradesh": (21.1, 26.9, 74.0, 82.8),
    "Maharashtra": (15.6, 22.0, 72.6, 80.9),
    "Manipur": (23.8, 25.7, 93.0, 94.8),
    "Meghalaya": (25.0, 26.1, 89.8, 92.8),
    "Mizoram": (21.9, 24.5, 92.3, 93.5),
    "Nagaland": (25.2, 27.1, 93.3, 95.3),
    "Odisha": (17.8, 22.6, 81.4, 87.5),
    "Punjab": (29.5, 32.5, 73.9, 76.9),
    "Rajasthan": (23.1, 30.2, 69.5, 78.3),
    "Sikkim": (27.1, 28.1, 88.0, 88.9),
    "Tamil Nadu": (8.1, 13.6, 76.2, 80.3),
    "Telangana": (15.8, 19.9, 77.2, 81.3),
    "Tripura": (22.9, 24.5, 91.2, 92.3),
    "Uttar Pradesh": (23.9, 30.4, 77.1, 84.6),
    "Uttarakhand": (28.7, 31.5, 77.6, 81.1),
    "West Bengal": (21.4, 27.3, 85.8, 89.9),
    "Andaman & Nicobar": (6.0, 13.7, 92.2, 93.9),
    "Chandigarh": (30.6, 30.8, 76.7, 76.9),
    "Delhi": (28.4, 28.9, 76.8, 77.3),
    "Jammu & Kashmir": (32.3, 37.1, 73.7, 80.4),
    "Ladakh": (32.0, 36.4, 76.0, 79.9),
    "Lakshadweep": (8.0, 12.4, 71.7, 74.1),
    "Puducherry": (11.9, 12.1, 79.7, 79.9),
}

ISD_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "temperature_c",
    "dew_point_c",
    "pressure_hpa",
    "wind_direction_deg",
    "wind_speed_ms",
    "sky_condition",
    "precip_1h_mm",
    "precip_6h_mm",
]
ISD_MISSING = -9999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def assign_state(lat: float, lon: float) -> str:
    candidates = []
    for state, (lat_min, lat_max, lon_min, lon_max) in INDIA_STATES.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            candidates.append(state)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        areas = {
            state: (INDIA_STATES[state][1] - INDIA_STATES[state][0])
            * (INDIA_STATES[state][3] - INDIA_STATES[state][2])
            for state in candidates
        }
        return min(areas, key=areas.get)
    return "Unknown"


def assign_district(lat: float, lon: float, state: str) -> str:
    lat_r = round(lat * 2) / 2
    lon_r = round(lon * 2) / 2
    return f"{state[:3].upper()}_{lat_r:.1f}N_{lon_r:.1f}E"


def season_for_month(month: int | float) -> str:
    if pd.isna(month):
        return "unknown"
    value = int(month)
    if value in (12, 1, 2):
        return "winter"
    if value in (3, 4, 5):
        return "pre_monsoon"
    if value in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def process_lis_otd_netcdf(nc_path: Path, config: dict, logger) -> pd.DataFrame:
    try:
        import netCDF4 as nc
    except ImportError:
        logger.error("netCDF4 not installed. Run: pip install -r requirements.txt")
        return pd.DataFrame()

    bounds = get_india_bounds(config)
    logger.info("Processing LIS/OTD climatology: %s", nc_path.name)
    records = []

    with nc.Dataset(str(nc_path)) as ds:
        lat_var = ds.variables.get("latitude") or ds.variables.get("lat")
        lon_var = ds.variables.get("longitude") or ds.variables.get("lon")
        if lat_var is None or lon_var is None:
            logger.warning("Skipping %s: latitude/longitude variables not found", nc_path)
            return pd.DataFrame()

        flash_var = next(
            (
                name
                for name in ["HRMC_COM_FR", "flash_rate", "fr", "combined_flash_rate"]
                if name in ds.variables
            ),
            None,
        )
        if flash_var is None:
            logger.warning("Skipping %s: flash-rate variable not found", nc_path)
            return pd.DataFrame()

        lat_arr = np.asarray(lat_var[:])
        lon_arr = np.asarray(lon_var[:])
        flash_data = np.asarray(ds.variables[flash_var][:])
        is_monthly = flash_data.ndim == 3 and flash_data.shape[0] == 12

        for i, lat in enumerate(tqdm(lat_arr, desc=f"LIS/OTD {nc_path.name}", leave=False)):
            for j, lon in enumerate(lon_arr):
                if not bounds.contains(lat, lon):
                    continue
                if is_monthly:
                    for month in range(12):
                        rate = float(flash_data[month, i, j])
                        if np.isfinite(rate) and rate >= 0:
                            records.append(
                                {
                                    "latitude": round(float(lat), 2),
                                    "longitude": round(float(lon), 2),
                                    "month": month + 1,
                                    "flash_rate_monthly": round(rate, 4),
                                }
                            )
                else:
                    rate = float(flash_data[i, j] if flash_data.ndim == 2 else flash_data[0, i, j])
                    if np.isfinite(rate) and rate >= 0:
                        records.append(
                            {
                                "latitude": round(float(lat), 2),
                                "longitude": round(float(lon), 2),
                                "flash_rate_annual": round(rate, 4),
                            }
                        )

    return pd.DataFrame(records)


def parse_open_meteo_json(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"datetime_utc": pd.to_datetime(times, utc=True, errors="coerce")})
    df["latitude"] = data.get("latitude")
    df["longitude"] = data.get("longitude")
    col_map = {
        "temperature_2m": "temperature_c",
        "relativehumidity_2m": "relative_humidity_pct",
        "relative_humidity_2m": "relative_humidity_pct",
        "dewpoint_2m": "dew_point_c",
        "dew_point_2m": "dew_point_c",
        "precipitation": "precipitation_mm",
        "surface_pressure": "pressure_hpa",
        "windspeed_10m": "wind_speed_ms",
        "wind_speed_10m": "wind_speed_ms",
        "winddirection_10m": "wind_direction_deg",
        "wind_direction_10m": "wind_direction_deg",
        "cloudcover": "cloud_cover_pct",
        "cloud_cover": "cloud_cover_pct",
        "cape": "cape_j_kg",
        "lifted_index": "lifted_index",
    }
    for source, target in col_map.items():
        if source in hourly:
            df[target] = hourly[source]
    return df


def load_open_meteo(raw_dir: Path, logger) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.json"))
    logger.info("Open-Meteo files: %s", len(files))
    frames = []
    for path in tqdm(files, desc="Open-Meteo", unit="file"):
        try:
            parsed = parse_open_meteo_json(path)
        except Exception as exc:
            logger.warning("Open-Meteo parse failed for %s: %s", path.name, exc)
            continue
        if not parsed.empty:
            parsed["data_source"] = "open_meteo"
            frames.append(parsed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_nasa_power_json(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    props = data.get("properties", {}).get("parameter", {})
    coords = data.get("geometry", {}).get("coordinates", [None, None])
    t2m = props.get("T2M", {})
    if not t2m:
        return pd.DataFrame()

    df = pd.DataFrame({"date_str": list(t2m.keys())})
    df["datetime_utc"] = pd.to_datetime(df["date_str"], format="%Y%m%d", utc=True, errors="coerce")
    df["latitude"] = coords[1]
    df["longitude"] = coords[0]
    col_map = {
        "T2M": "temperature_c",
        "T2MDEW": "dew_point_c",
        "RH2M": "relative_humidity_pct",
        "PRECTOTCORR": "precipitation_mm",
        "PS": "pressure_hpa",
        "WS10M": "wind_speed_ms",
        "WD10M": "wind_direction_deg",
        "CLOUD_AMT": "cloud_cover_pct",
        "T2M_MAX": "temperature_max_c",
        "T2M_MIN": "temperature_min_c",
    }
    for source, target in col_map.items():
        if source in props:
            df[target] = df["date_str"].map(props[source]).astype(float)
    if "pressure_hpa" in df:
        df["pressure_hpa"] = df["pressure_hpa"] * 10.0
    return df.drop(columns=["date_str"])


def load_nasa_power(raw_dir: Path, logger) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.json"))
    logger.info("NASA POWER files: %s", len(files))
    frames = []
    for path in tqdm(files, desc="NASA POWER", unit="file"):
        try:
            parsed = parse_nasa_power_json(path)
        except Exception as exc:
            logger.warning("NASA POWER parse failed for %s: %s", path.name, exc)
            continue
        if not parsed.empty:
            parsed["data_source"] = "nasa_power"
            frames.append(parsed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_isd_lite_gz(gz_path: Path) -> pd.DataFrame:
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        df = pd.read_csv(handle, sep=r"\s+", header=None, names=ISD_COLUMNS, na_values=[ISD_MISSING])

    for col in ["temperature_c", "dew_point_c", "pressure_hpa", "wind_speed_ms", "precip_1h_mm", "precip_6h_mm"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 10.0

    df["datetime_utc"] = pd.to_datetime(df[["year", "month", "day", "hour"]], utc=True, errors="coerce")
    df["relative_humidity_pct"] = 100 * np.exp(
        (17.625 * df["dew_point_c"] / (243.04 + df["dew_point_c"]))
        - (17.625 * df["temperature_c"] / (243.04 + df["temperature_c"]))
    )
    df["station_id"] = gz_path.stem.split("-")[0]
    return df


def load_isd(raw_dir: Path, logger) -> pd.DataFrame:
    station_meta = raw_dir / "stations.csv"
    station_lookup = pd.DataFrame()
    if station_meta.exists():
        station_lookup = pd.read_csv(station_meta, dtype={"station_id": str})

    files = sorted(raw_dir.rglob("*.gz"))
    logger.info("NOAA ISD files: %s", len(files))
    frames = []
    for path in tqdm(files, desc="NOAA ISD", unit="file"):
        try:
            parsed = parse_isd_lite_gz(path)
        except Exception as exc:
            logger.warning("ISD parse failed for %s: %s", path.name, exc)
            continue
        if not station_lookup.empty and "station_id" in station_lookup:
            parsed = parsed.merge(station_lookup, on="station_id", how="left", suffixes=("", "_station"))
        if {"latitude", "longitude"}.issubset(parsed.columns):
            parsed["data_source"] = "noaa_isd"
            frames.append(parsed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assign_flash_data(met_df: pd.DataFrame, annual_df: pd.DataFrame, monthly_df: pd.DataFrame, logger) -> pd.DataFrame:
    if met_df.empty:
        return met_df
    if annual_df.empty and monthly_df.empty:
        logger.warning("No LIS/OTD climatology available; flash-rate features will be NaN.")
        met_df["flash_rate_annual"] = np.nan
        met_df["flash_rate_monthly"] = np.nan
        met_df["lightning_flash_count_est"] = np.nan
        return met_df

    from scipy.spatial import cKDTree

    def nearest(source_df: pd.DataFrame, value_col: str) -> np.ndarray:
        tree = cKDTree(source_df[["latitude", "longitude"]].to_numpy())
        _, idx = tree.query(met_df[["latitude", "longitude"]].to_numpy(), k=1)
        return source_df[value_col].iloc[idx].to_numpy()

    met_df["flash_rate_annual"] = nearest(annual_df, "flash_rate_annual") if not annual_df.empty else np.nan
    if not monthly_df.empty:
        monthly_rates = []
        for _, row in met_df.iterrows():
            sub = monthly_df[monthly_df["month"] == int(row.get("month", 1) or 1)]
            if sub.empty:
                monthly_rates.append(np.nan)
                continue
            tree = cKDTree(sub[["latitude", "longitude"]].to_numpy())
            _, idx = tree.query([[row["latitude"], row["longitude"]]], k=1)
            monthly_rates.append(float(sub["flash_rate_monthly"].iloc[idx[0]]))
        met_df["flash_rate_monthly"] = monthly_rates
    else:
        met_df["flash_rate_monthly"] = met_df["flash_rate_annual"]

    cell_area_km2 = 55.0 * 55.0
    met_df["lightning_flash_count_est"] = (
        met_df["flash_rate_monthly"].fillna(met_df["flash_rate_annual"]) * cell_area_km2 / 8760.0
    ).round(4)
    return met_df


def add_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["datetime_utc"], utc=True, errors="coerce")
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["season"] = out["month"].apply(season_for_month)
    out["state"] = out.apply(lambda row: assign_state(float(row["latitude"]), float(row["longitude"])), axis=1)
    out["district"] = out.apply(
        lambda row: assign_district(float(row["latitude"]), float(row["longitude"]), row["state"]),
        axis=1,
    )
    out["thunderstorm_risk_label"] = (out["lightning_flash_count_est"].fillna(0) > 0).astype(int)
    return out


def write_reports(df: pd.DataFrame, out_dir: Path) -> None:
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    df.isna().sum().sort_values(ascending=False).to_csv(report_dir / "missing_value_report.csv", header=["missing_count"])
    df.describe(include="all").transpose().to_csv(report_dir / "summary_statistics.csv")

    numeric = df.select_dtypes(include=[np.number])
    if "thunderstorm_risk_label" in numeric:
        importance = numeric.corr(numeric_only=True)["thunderstorm_risk_label"].abs().sort_values(ascending=False)
        importance.to_csv(report_dir / "feature_importance_proxy.csv", header=["abs_correlation_to_label"])


def build_prediction_dataset(config: dict, logger) -> pd.DataFrame:
    bounds = get_india_bounds(config)
    lis_dir = resolve_path(config, "raw_lis_otd")
    annual_df = pd.DataFrame()
    monthly_df = pd.DataFrame()

    for path in sorted(lis_dir.glob("*.nc")):
        parsed = process_lis_otd_netcdf(path, config, logger)
        if parsed.empty:
            continue
        if "flash_rate_monthly" in parsed:
            monthly_df = pd.concat([monthly_df, parsed], ignore_index=True)
        else:
            annual_df = pd.concat([annual_df, parsed], ignore_index=True)

    met_frames = [
        load_open_meteo(resolve_path(config, "raw_open_meteo"), logger),
        load_nasa_power(resolve_path(config, "raw_nasa_power"), logger),
        load_isd(resolve_path(config, "raw_noaa_isd"), logger),
    ]
    met_frames = [frame for frame in met_frames if not frame.empty]
    if not met_frames:
        logger.error("No meteorological data found in raw input folders.")
        return pd.DataFrame()

    met_df = pd.concat(met_frames, ignore_index=True)
    met_df = met_df[pd.to_numeric(met_df["latitude"], errors="coerce").notna()]
    met_df = met_df[pd.to_numeric(met_df["longitude"], errors="coerce").notna()]
    met_df["latitude"] = met_df["latitude"].astype(float)
    met_df["longitude"] = met_df["longitude"].astype(float)
    met_df = met_df[met_df.apply(lambda row: bounds.contains(row["latitude"], row["longitude"]), axis=1)].copy()
    met_df["month"] = pd.to_datetime(met_df["datetime_utc"], utc=True, errors="coerce").dt.month
    met_df = assign_flash_data(met_df, annual_df, monthly_df, logger)
    return add_prediction_features(met_df)


def write_outputs(df: pd.DataFrame, config: dict, logger) -> None:
    training_dir = resolve_path(config, "processed_training")
    climatology_dir = resolve_path(config, "processed_climatology")
    weather_dir = resolve_path(config, "processed_weather")
    training_dir.mkdir(parents=True, exist_ok=True)
    climatology_dir.mkdir(parents=True, exist_ok=True)
    weather_dir.mkdir(parents=True, exist_ok=True)

    prediction_csv = training_dir / "prediction_dataset.csv"
    training_csv = training_dir / "training_dataset.csv"
    prediction_parquet = training_dir / "prediction_dataset.parquet"
    feature_parquet = training_dir / "feature_dataset.parquet"

    df.to_csv(prediction_csv, index=False)
    df.to_csv(training_csv, index=False)
    try:
        df.to_parquet(prediction_parquet, index=False)
        df.to_parquet(feature_parquet, index=False)
    except Exception as exc:
        logger.warning("Parquet output skipped: %s", exc)

    write_reports(df, training_dir)
    logger.info("Prediction dataset rows: %s", len(df))
    logger.info("Wrote %s", prediction_csv)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(resolve_path(config, "logs") / "prediction.log", "prediction")
    start = time.perf_counter()
    logger.info("Prediction dataset build started")
    logger.info("Project root: %s", PROJECT_ROOT)

    dataset = build_prediction_dataset(config, logger)
    if dataset.empty:
        logger.error("Prediction dataset is empty.")
        logger.info("Prediction dataset build finished duration_seconds=%.2f", time.perf_counter() - start)
        return 1
    write_outputs(dataset, config, logger)
    logger.info("Prediction dataset build finished duration_seconds=%.2f", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
