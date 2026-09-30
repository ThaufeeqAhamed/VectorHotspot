#!/usr/bin/env python3
"""
VectorHotspot — Feature Metadata Generator

Reads the actual feature columns from the processed feature parquets and
produces a structured feature dictionary (CSV + Markdown) with:
  - Feature name
  - Feature group (disease/weather/environment/spatial/seasonality/target)
  - Human-readable description
  - Data type
  - Value range (min/max from training set)
  - Missing value count

Output (new):
  outputs/reports/feature_metadata.csv
  outputs/reports/feature_metadata.md

Usage:
  py -3 src/data_prep/generate_feature_metadata.py

Notes:
  - Reads only features_dengue_train.parquet (Malaria uses same schema)
  - Does NOT modify any existing file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ---------------------------------------------------------------------------
# Feature dictionary — name → (group, description)
# ---------------------------------------------------------------------------
FEATURE_DICT = {
    # Disease History
    "cases_lag_1":          ("Disease History", "Normalized case rate 1 week before prediction point"),
    "cases_lag_2":          ("Disease History", "Normalized case rate 2 weeks before prediction point"),
    "cases_lag_3":          ("Disease History", "Normalized case rate 3 weeks before prediction point"),
    "cases_lag_4":          ("Disease History", "Normalized case rate 4 weeks before prediction point"),
    "cases_lag_8":          ("Disease History", "Normalized case rate 8 weeks before prediction point"),
    "cases_roll_mean_4w":   ("Disease History", "Rolling mean of normalized cases over past 4 weeks"),
    "cases_roll_std_4w":    ("Disease History", "Rolling std dev of normalized cases over past 4 weeks"),
    "cases_roll_mean_12w":  ("Disease History", "Rolling mean of normalized cases over past 12 weeks"),
    "cases_momentum_4w":    ("Disease History", "Change in cases over past 4 weeks (cases_lag_1 - cases_lag_4)"),
    "case_rate_lag_1":      ("Disease History", "Cases per population rate 1 week prior (incidence rate)"),

    # Spatial Neighbor
    "neighbor_cases_k1_lag1": ("Spatial Neighbor", "Mean case rate of ring-1 H3 neighbors, lag 1 week"),
    "neighbor_cases_k1_lag2": ("Spatial Neighbor", "Mean case rate of ring-1 H3 neighbors, lag 2 weeks"),
    "neighbor_cases_k2_lag1": ("Spatial Neighbor", "Mean case rate of ring-2 H3 neighbors, lag 1 week"),

    # Temperature
    "tmax_lag_1":  ("Weather — Temperature", "Maximum daily temperature (°C), 1 week prior"),
    "tmax_lag_2":  ("Weather — Temperature", "Maximum daily temperature (°C), 2 weeks prior"),
    "tmax_lag_4":  ("Weather — Temperature", "Maximum daily temperature (°C), 4 weeks prior"),
    "tmin_lag_1":  ("Weather — Temperature", "Minimum daily temperature (°C), 1 week prior"),
    "tmin_lag_2":  ("Weather — Temperature", "Minimum daily temperature (°C), 2 weeks prior"),
    "tmin_lag_4":  ("Weather — Temperature", "Minimum daily temperature (°C), 4 weeks prior"),
    "tmean_lag_1": ("Weather — Temperature", "Mean daily temperature (°C), 1 week prior"),
    "tmean_lag_2": ("Weather — Temperature", "Mean daily temperature (°C), 2 weeks prior"),
    "tmean_lag_4": ("Weather — Temperature", "Mean daily temperature (°C), 4 weeks prior"),
    "dtr_lag_1":   ("Weather — Temperature", "Diurnal temperature range (Tmax - Tmin), 1 week prior"),
    "dtr_lag_2":   ("Weather — Temperature", "Diurnal temperature range (Tmax - Tmin), 2 weeks prior"),

    # Rainfall
    "rain_lag_1":       ("Weather — Rainfall", "Total weekly rainfall (mm), 1 week prior"),
    "rain_lag_2":       ("Weather — Rainfall", "Total weekly rainfall (mm), 2 weeks prior"),
    "rain_lag_4":       ("Weather — Rainfall", "Total weekly rainfall (mm), 4 weeks prior"),
    "rain_lag_6":       ("Weather — Rainfall", "Total weekly rainfall (mm), 6 weeks prior"),
    "rain_roll_sum_2w": ("Weather — Rainfall", "Cumulative rainfall over past 2 weeks (mm)"),
    "rain_roll_sum_4w": ("Weather — Rainfall", "Cumulative rainfall over past 4 weeks (mm)"),

    # Environmental
    "suitability_lag_1": ("Environmental", "Vector habitat suitability index, 1 week prior (composite environmental score)"),
    "suitability_lag_2": ("Environmental", "Vector habitat suitability index, 2 weeks prior"),
    "suitability_lag_4": ("Environmental", "Vector habitat suitability index, 4 weeks prior"),
    "ndvi_mean":         ("Environmental", "Mean NDVI (Normalized Difference Vegetation Index) for H3 cell (MODIS, static)"),
    "frac_trees":        ("Environmental", "Fraction of H3 cell covered by trees (ESA WorldCover 2021)"),
    "frac_water":        ("Environmental", "Fraction of H3 cell covered by water bodies (ESA WorldCover 2021)"),
    "frac_built":        ("Environmental", "Fraction of H3 cell covered by built-up area (ESA WorldCover 2021)"),
    "frac_shrub":        ("Environmental", "Fraction of H3 cell covered by shrubland (ESA WorldCover 2021)"),
    "jrc_occurrence":    ("Environmental", "Water occurrence frequency 0–100 (JRC Global Surface Water)"),

    # Population
    "log_population":  ("Population", "Log-transformed population count for H3 cell (WorldPop 1km, interpolated)"),
    "pop_density":     ("Population", "Population density per km² for H3 cell"),

    # Geography
    "center_lat": ("Geography", "Latitude of H3 cell centroid (decimal degrees)"),
    "center_lon": ("Geography", "Longitude of H3 cell centroid (decimal degrees)"),

    # Seasonality
    "sin_week": ("Seasonality", "Sine of week-of-year (2π × week / 52) — captures annual seasonality"),
    "cos_week": ("Seasonality", "Cosine of week-of-year (2π × week / 52) — captures annual seasonality"),

    # Targets (documented but excluded from model input)
    "target_lead_1": ("Target", "Normalized case rate 1 week ahead (prediction target for t+1 horizon)"),
    "target_lead_2": ("Target", "Normalized case rate 2 weeks ahead (prediction target for t+2 horizon)"),
    "target_lead_3": ("Target", "Normalized case rate 3 weeks ahead (prediction target for t+3 horizon)"),
    "target_lead_4": ("Target", "Normalized case rate 4 weeks ahead (prediction target for t+4 horizon)"),

    # Metadata columns (not features)
    "h3_index":          ("Metadata", "H3 hexagonal cell index at resolution 7 (~5.2 km²)"),
    "state":             ("Metadata", "Indian state name for the H3 cell"),
    "district":          ("Metadata", "Indian district name for the H3 cell"),
    "year":              ("Metadata", "Calendar year of the observation"),
    "week":              ("Metadata", "ISO week of year (1–52/53)"),
    "is_spatial_holdout":("Metadata", "Boolean flag: True if district is held out from model training"),
    "outbreak_lead_1":   ("Target", "Binary outbreak flag 1 week ahead (1 = high risk threshold exceeded)"),
    "outbreak_lead_2":   ("Target", "Binary outbreak flag 2 weeks ahead"),
    "outbreak_lead_4":   ("Target", "Binary outbreak flag 4 weeks ahead"),
}


def generate_feature_metadata():
    """Generate feature metadata CSV and Markdown from training parquet."""
    print("=" * 65)
    print(" VECTORHOTSPOT — FEATURE METADATA GENERATOR")
    print("=" * 65)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training set for statistics
    train_path = PROCESSED_DIR / "features_dengue_train.parquet"
    print(f"\nLoading {train_path.name} ...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_parquet(train_path)
    print(f"done ({len(df):,} rows, {time.time()-t0:.1f}s)", flush=True)

    # Build metadata records
    records = []
    for col in df.columns:
        group, desc = FEATURE_DICT.get(col, ("Unknown", f"Column '{col}' — no description available"))

        if col in df.select_dtypes(include="number").columns:
            s = df[col]
            dtype = str(s.dtype)
            val_min = round(float(s.min()), 6)
            val_max = round(float(s.max()), 6)
            val_mean = round(float(s.mean()), 6)
            null_count = int(s.isnull().sum())
        else:
            dtype = str(df[col].dtype)
            val_min = val_max = val_mean = None
            null_count = int(df[col].isnull().sum())

        records.append({
            "feature": col,
            "group": group,
            "description": desc,
            "dtype": dtype,
            "val_min": val_min,
            "val_max": val_max,
            "val_mean": val_mean,
            "null_count_train": null_count,
            "is_model_input": group not in ("Metadata", "Target"),
        })

    df_meta = pd.DataFrame(records)

    # Save CSV
    csv_path = REPORTS_DIR / "feature_metadata.csv"
    df_meta.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path.name}")

    # Save Markdown
    md_lines = [
        "# VectorHotspot — Feature Metadata Dictionary\n",
        "> Auto-generated from `features_dengue_train.parquet`.",
        "> Value ranges are from the training split (2000–~2021).",
        "> Malaria models use the identical feature schema.\n",
        "---\n",
    ]

    groups_order = [
        "Disease History", "Spatial Neighbor",
        "Weather — Temperature", "Weather — Rainfall",
        "Environmental", "Population", "Geography", "Seasonality",
        "Target", "Metadata",
    ]

    for group in groups_order:
        group_rows = df_meta[df_meta["group"] == group]
        if group_rows.empty:
            continue
        md_lines.append(f"## {group}\n")
        md_lines.append("| Feature | Description | Min | Max | Mean | Null Count |")
        md_lines.append("|---|---|---|---|---|---|")
        for _, row in group_rows.iterrows():
            mn = f"{row['val_min']:.4f}" if row["val_min"] is not None else "—"
            mx = f"{row['val_max']:.4f}" if row["val_max"] is not None else "—"
            me = f"{row['val_mean']:.4f}" if row["val_mean"] is not None else "—"
            md_lines.append(
                f"| `{row['feature']}` | {row['description']} | {mn} | {mx} | {me} | {row['null_count_train']} |"
            )
        md_lines.append("")

    md_lines.append("---\n")
    md_lines.append(f"*Total features (model inputs): {df_meta[df_meta['is_model_input']]['feature'].nunique()}*\n")

    md_path = REPORTS_DIR / "feature_metadata.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown: {md_path.name}")

    # Print summary
    print(f"\n{'='*65}")
    print(" FEATURE SUMMARY BY GROUP")
    print(f"{'='*65}")
    for group in groups_order:
        n = len(df_meta[df_meta["group"] == group])
        if n > 0:
            print(f"  {group:<30} {n:>3} features")
    print(f"  {'─'*34}")
    total_inputs = df_meta[df_meta["is_model_input"]]["feature"].nunique()
    print(f"  {'Total model inputs':<30} {total_inputs:>3}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    generate_feature_metadata()
