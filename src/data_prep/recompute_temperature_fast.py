"""
Recompute ONLY tmax/tmin (rainfall from your existing CSV is already
correct and is reused as-is) -- fixing both the correctness bug and the
severe slowness of the original approach.

WHAT WAS WRONG:
1. CORRECTNESS: rasterstats.zonal_stats() by default only counts a pixel
   if its CENTER falls inside a district polygon. Temperature's grid is
   1x1 degree (~100km) -- far coarser than most Indian districts -- so
   most districts had ZERO qualifying pixels and silently got null
   forever. Fixed here with all_touched=True (any overlap counts).
2. SPEED: the original script called zonal_stats() separately for each
   of ~9,125 days, re-rasterizing the district/pixel relationship from
   scratch every single call, even though that relationship never
   changes day to day. This is why it took 8 hours.

THE FIX: rasterize the district-to-pixel mapping ONCE (not per day), then
for each day just do a fast vectorized scipy.ndimage lookup against that
fixed mapping. This should take minutes, not hours.

SETUP (you should already have these from before):
    pip install imdlib xarray pandas geopandas rasterio shapely scipy

BEFORE RUNNING:
    - india_districts_clean.geojson in the same folder
    - Your existing imd_district_weekly_weather_2000_2024.csv in the same
      folder (its correct Rainfall_mm column is reused directly)
    - Your already-downloaded imd_raw_data/tmax and imd_raw_data/tmin
      folders in the same folder (no re-download needed)

OUTPUT:
    imd_district_weekly_weather_2000_2024_FIXED.csv
"""

import imdlib as imd
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from rasterio.features import rasterize
from scipy import ndimage
import warnings
import os

warnings.filterwarnings("ignore")

START_YEAR = 2000
END_YEAR = 2024
from pathlib import Path as FilePath

SCRIPT_DIR = FilePath(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DISTRICTS_GEOJSON = PROJECT_ROOT / "data" / "boundaries" / "india_districts_clean.geojson"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EXISTING_CSV = OUTPUT_DIR / "imd_district_weekly_weather_2000_2024.csv"
OUTPUT_CSV = OUTPUT_DIR / "imd_district_weekly_weather_2000_2024_FIXED.csv"
DATA_DIR = PROJECT_ROOT / "imd_raw_data"

# ----------------------------------------------------------------------
# 1. Load districts and existing (correct) rainfall data
# ----------------------------------------------------------------------

print("Loading district boundaries...")
districts = gpd.read_file(DISTRICTS_GEOJSON).to_crs("EPSG:4326").reset_index(drop=True)
n_districts = len(districts)
print(f"  {n_districts} districts")

print("Loading existing weekly CSV (reusing its Rainfall_mm as-is)...")
existing = pd.read_csv(EXISTING_CSV)
rain_weekly = existing[["State", "District", "Year", "Week", "Rainfall_mm"]].copy()


def raster_transform_from_grid(lats, lons):
    res_lat = abs(lats[1] - lats[0])
    res_lon = abs(lons[1] - lons[0])
    return from_origin(lons.min() - res_lon / 2, lats.max() + res_lat / 2, res_lon, res_lat)


def build_district_pixel_membership(districts, transform, out_shape):
    """
    For each district, find every pixel it touches -- computed
    INDEPENDENTLY per district (not via a single combined rasterize()
    call), so that a pixel shared by several small districts correctly
    counts toward ALL of them, not just whichever one happened to be
    drawn last. This is the fix for the "winner-take-all" bug: a plain
    rasterize() call assigns each pixel exactly ONE label, silently
    dropping every other district that legitimately overlaps that same
    (large, ~100km) temperature grid cell.

    Returns two parallel arrays: membership_district_idx and
    membership_flat_pixel_idx, where each (i, j) pair means "pixel j
    (flat index) belongs to district i". A pixel can appear multiple
    times for different districts.
    """
    n_rows, n_cols = out_shape
    district_idx_list = []
    pixel_idx_list = []
    for idx, geom in enumerate(districts.geometry):
        mask = rasterize(
            [(geom, 1)], out_shape=out_shape, transform=transform,
            fill=0, all_touched=True, dtype="uint8",
        )
        rows, cols = np.nonzero(mask)
        flat_idx = rows * n_cols + cols
        district_idx_list.append(np.full(len(flat_idx), idx, dtype=np.int32))
        pixel_idx_list.append(flat_idx)
    return np.concatenate(district_idx_list), np.concatenate(pixel_idx_list)


def fast_daily_to_district(ds, varname, districts, nodata_sentinels=(-999.0, 99.9, -99.9)):
    lats = ds["lat"].values
    lons = ds["lon"].values
    transform = raster_transform_from_grid(lats, lons)
    n_rows, n_cols = len(lats), len(lons)

    print("  Building per-district pixel membership (one-time cost, allows shared pixels)...")
    membership_district_idx, membership_pixel_idx = build_district_pixel_membership(
        districts, transform, (n_rows, n_cols)
    )
    print(f"  {len(membership_district_idx)} district-pixel membership pairs "
          f"across {n_districts} districts "
          f"(avg {len(membership_district_idx)/n_districts:.1f} pixels/district)")

    times = ds["time"].values
    n_times = len(times)
    records = []

    for i, t in enumerate(times):
        day_slice = ds[varname].isel(time=i).values
        if lats[0] < lats[-1]:
            day_slice = np.flipud(day_slice)

        invalid = np.zeros(day_slice.shape, dtype=bool)
        for sentinel in nodata_sentinels:
            invalid |= np.isclose(day_slice, sentinel)
        invalid |= np.isnan(day_slice)

        day_flat = day_slice.ravel()
        invalid_flat = invalid.ravel()

        values_at_membership = day_flat[membership_pixel_idx]
        valid_membership = ~invalid_flat[membership_pixel_idx]

        valid_district_idx = membership_district_idx[valid_membership]
        valid_values = values_at_membership[valid_membership]

        sums = np.bincount(valid_district_idx, weights=valid_values, minlength=n_districts)
        counts = np.bincount(valid_district_idx, minlength=n_districts)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)

        date = pd.Timestamp(t)
        for zid in range(n_districts):
            records.append({
                "State": districts.iloc[zid]["st_nm"],
                "District": districts.iloc[zid]["district"],
                "date": date,
                "value": means[zid],
            })

        if (i + 1) % 500 == 0:
            print(f"    {varname}: {i+1}/{n_times} days processed")

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 2. Open and process tmax / tmin only
# ----------------------------------------------------------------------

print("\nOpening tmax data...")
tmax_data = imd.open_data("tmax", START_YEAR, END_YEAR, "yearwise", file_dir=os.path.join(DATA_DIR, "tmax"))
tmax_ds = tmax_data.get_xarray()
print("Computing district-level daily tmax (fast method)...")
tmax_district = fast_daily_to_district(tmax_ds, "tmax", districts)
tmax_district = tmax_district.rename(columns={"value": "Tmax_C"})

print("\nOpening tmin data...")
tmin_data = imd.open_data("tmin", START_YEAR, END_YEAR, "yearwise", file_dir=os.path.join(DATA_DIR, "tmin"))
tmin_ds = tmin_data.get_xarray()
print("Computing district-level daily tmin (fast method)...")
tmin_district = fast_daily_to_district(tmin_ds, "tmin", districts)
tmin_district = tmin_district.rename(columns={"value": "Tmin_C"})

# ----------------------------------------------------------------------
# 3. Aggregate to weekly and merge with existing rainfall
# ----------------------------------------------------------------------

print("\nAggregating temperature to weekly...")
for df_ in (tmax_district, tmin_district):
    df_["Year"] = df_["date"].dt.isocalendar().year
    df_["Week"] = df_["date"].dt.isocalendar().week

tmax_weekly = tmax_district.groupby(["State", "District", "Year", "Week"]).agg(
    Tmax_C=("Tmax_C", "mean")
).reset_index()
tmin_weekly = tmin_district.groupby(["State", "District", "Year", "Week"]).agg(
    Tmin_C=("Tmin_C", "mean")
).reset_index()

final = rain_weekly.merge(
    tmax_weekly, on=["State", "District", "Year", "Week"], how="left"
).merge(
    tmin_weekly, on=["State", "District", "Year", "Week"], how="left"
)

final.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Saved {len(final)} rows to {OUTPUT_CSV}")
print(f"Null Tmax_C: {final['Tmax_C'].isna().sum()} ({final['Tmax_C'].isna().mean()*100:.1f}%)")
print(f"Null Tmin_C: {final['Tmin_C'].isna().sum()} ({final['Tmin_C'].isna().mean()*100:.1f}%)")
print("\nPlease upload this CSV back to continue.")