#!/usr/bin/env python3
"""Train a baseline thunderstorm-risk model from processed prediction features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from common import ensure_project_dirs, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--target", default="thunderstorm_risk_label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs(config)
    logger = setup_logging(resolve_path(config, "logs") / "prediction_model.log", "prediction_model")
    start = time.perf_counter()
    logger.info("Prediction model training started")
    training_path = resolve_path(config, "processed_training") / "training_dataset.csv"
    if not training_path.exists():
        logger.error("Missing training dataset: %s", training_path)
        logger.info("Prediction model training finished duration_seconds=%.2f", time.perf_counter() - start)
        return 1

    data = pd.read_csv(training_path)
    if args.target not in data:
        logger.error("Target column not found: %s", args.target)
        logger.info("Prediction model training finished duration_seconds=%.2f", time.perf_counter() - start)
        return 1

    numeric = data.select_dtypes(include=["number"]).dropna(axis=1, how="all")
    if args.target not in numeric:
        numeric[args.target] = data[args.target]
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    x = numeric.drop(columns=[args.target], errors="ignore")
    y = numeric[args.target].astype(int)
    if x.empty or y.nunique() < 2:
        logger.error("Not enough numeric features or target classes to train a model.")
        logger.info("Prediction model training finished duration_seconds=%.2f", time.perf_counter() - start)
        return 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    report = classification_report(y_test, model.predict(x_test), output_dict=True)

    model_dir = resolve_path(config, "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "thunderstorm_risk_random_forest.joblib")
    (model_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False).to_csv(
        model_dir / "feature_importance.csv", header=["importance"]
    )
    logger.info("Model and reports written to %s", model_dir)
    logger.info("Prediction model training finished duration_seconds=%.2f", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
