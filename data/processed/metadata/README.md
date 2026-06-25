# AKAM Synthetic Lightning Dataset (Proof of Concept)

## IMPORTANT DISCLAIMER

This dataset contains synthetic lightning events generated around real Indian geographic locations for software development, visualization, and machine learning testing. It does not represent actual lightning observations.

## Generation parameters

- Start date: 2026-06-18
- Number of days: 3
- Total events generated: 13268
- Total CTBT frames (30-min) across window: 144
- Random seed: 1234
- Geography source: hand-curated reference table of real Indian
  states/districts/cities (`src/geo_reference.py`), compiled without live
  GADM/OSM/GeoNames access in this environment. Coverage is best-effort
  and weighted toward thunderstorm-prone regions as specified by the user;
  it is NOT an exhaustive list of all Indian districts.
- Lightning events are placed near these real reference points using
  synthetic storm-cluster physics (see `src/storm_engine.py`) and are not
  derived from any real lightning detection network.

## Output structure

```
data/processed/
  lightning/
    india_lightning_events.csv      # master event file
    frames/lightning_HHMM.csv       # one file per 30-min CTBT slot (48 files)
    storm_tracks.csv                # per-storm per-frame position/intensity
    storm_summary.csv               # one row per storm (birth/death/kinematics)
    daily_summary.csv
    hourly_summary.csv
    cluster_statistics.csv
    lightning_density_grid.csv      # 0.5-degree grid aggregation
  training/
    training_dataset.csv            # ML-ready feature table
  metadata/
    README.md                       # this file
```

## Known limitations of this proof-of-concept

- Scale: ~13268 events over 3 days (vs. the full 100k-250k /
  30-day target). Re-run `generate_lightning_dataset.py` with
  `N_DAYS=30` and `TARGET_TOTAL_EVENTS` in the 100k-250k range to scale up.
- District coverage: ~136 curated real locations, not all ~750 Indian
  districts. Skewed toward the high-thunderstorm states requested.
- Storm movement is a simplified linear-drift model with a
  developing/dissipating intensity curve -- not a physical NWP simulation.
- CTBT image files (ctbt_HHMM.png) are referenced by filename only; no
  actual satellite imagery is generated or included.
