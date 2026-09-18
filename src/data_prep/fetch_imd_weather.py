"""
Fetch IMD gridded rainfall + temperature data (2000-2024) and aggregate it
to district-level weekly averages, ready to join with our dengue/malaria
case data and district boundaries.

WHY THIS SCRIPT EXISTS / RUN THIS YOURSELF:
IMD's data portal serves files via a POST-form submission (not a plain
downloadable link), so it can't be fetched automatically by Claude's tools.
This script uses `imdlib` (open-source, does the same POST requests IMD's
own website does) to download the data on YOUR machine, then immediately
aggregates ~25 years of daily gridded data down to a single compact CSV
(a few MB instead of several GB of raw grids) that you can upload back.

IMD's server is known to be flaky/slow and will intermittently refuse
connections under repeated requests. This version downloads ONE YEAR AT A
TIME with automatic retries (with backoff) per year, and SKIPS any year
that has already been downloaded successfully -- so if the script dies
partway through (as it likely will at least once), you can just rerun it
and it will pick up where it left off instead of starting over.

SETUP (run once):
    pip install imdlib xarray pandas geopandas rasterstats netCDF4 shapely

BEFORE RUNNING:
    Place india_districts_clean.geojson (from our earlier session) in the
    same folder as this script, or update DISTRICTS_GEOJSON below.

OUTPUT:
    imd_district_weekly_weather_2000_2024.csv
    Columns: State, District, Year, Week, Rainfall_mm, Tmax_C, Tmin_C
"""

import imdlib as imd
import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import rasterio
from rasterio.transform import from_origin
import warnings
import time
import glob
import os
warnings.filterwarnings("ignore")

START_YEAR = 2000
END_YEAR = 2024
DISTRICTS_GEOJSON = "india_districts_clean.geojson"
OUTPUT_CSV = "imd_district_weekly_weather_2000_2024.csv"
DATA_DIR = "imd_raw_data"  # NEW downloads go here
# Older runs may have saved files directly to ./rain, ./tmax, ./tmin instead --
# check both locations so already-downloaded years are never redownloaded.
CANDIDATE_DATA_DIRS = ["imd_raw_data", "."]

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 45  # IMD's server needs a breather between failures


def year_already_downloaded(variable, year):
    """Check if this variable+year is already downloaded and readable, by
    asking imdlib itself to open it (avoids guessing imdlib's internal
    filename convention, which was unreliable)."""
    for base in CANDIDATE_DATA_DIRS:
        folder = os.path.join(base, variable)
        if not os.path.isdir(folder):
            continue
        try:
            imd.open_data(variable, year, year, 'yearwise', file_dir=folder)
            return True
        except Exception:
            continue
    return False


def download_variable_year(variable, year):
    """Download a single variable/year with retries and backoff."""
    if year_already_downloaded(variable, year):
        print(f"  {variable} {year}: already downloaded, skipping")
        return True

    folder = os.path.join(DATA_DIR, variable)
    os.makedirs(folder, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            imd.get_data(variable, year, year, fn_format='yearwise', file_dir=folder)
            print(f"  {variable} {year}: downloaded (attempt {attempt})")
            return True
        except Exception as e:
            print(f"  {variable} {year}: attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}: {e})")
            if attempt < MAX_RETRIES:
                print(f"    waiting {RETRY_WAIT_SECONDS}s before retry...")
                time.sleep(RETRY_WAIT_SECONDS)
    print(f"  {variable} {year}: GAVE UP after {MAX_RETRIES} attempts -- will need a manual rerun later")
    return False


# ----------------------------------------------------------------------
# 1. Download the raw IMD grids, one year at a time, with retries
# ----------------------------------------------------------------------

failed = []
for variable in ['rain', 'tmax', 'tmin']:
    print(f"\nDownloading {variable} data, {START_YEAR}-{END_YEAR}...")
    for year in range(START_YEAR, END_YEAR + 1):
        ok = download_variable_year(variable, year)
        if not ok:
            failed.append((variable, year))

if failed:
    print(f"\n{'='*60}")
    print(f"WARNING: {len(failed)} variable/year combos failed after all retries:")
    for v, y in failed:
        print(f"  - {v} {y}")
    print("Just rerun this script again later -- it will skip everything")
    print("that already succeeded and only retry the failed ones.")
    print(f"{'='*60}\n")
    print("Stopping here so you can rerun for the missing years.")
    print("(Or comment out this exit() to proceed with partial data.)")
    import sys
    sys.exit(1)

print("\nAll years downloaded successfully!")

# ----------------------------------------------------------------------
# 1b. Consolidate: if any files ended up in the old default location
#     (./rain, ./tmax, ./tmin from an earlier run), copy them into
#     imd_raw_data/ so everything downstream reads from one place.
# ----------------------------------------------------------------------

import shutil

for variable in ['rain', 'tmax', 'tmin']:
    old_folder = os.path.join(".", variable)
    new_folder = os.path.join(DATA_DIR, variable)
    if os.path.isdir(old_folder) and os.path.abspath(old_folder) != os.path.abspath(new_folder):
        os.makedirs(new_folder, exist_ok=True)
        for fname in os.listdir(old_folder):
            src = os.path.join(old_folder, fname)
            dst = os.path.join(new_folder, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  consolidated {src} -> {dst}")

# ----------------------------------------------------------------------
# 2. Open as xarray datasets
# ----------------------------------------------------------------------

print("Opening downloaded data...")
rain_data = imd.open_data('rain', START_YEAR, END_YEAR, 'yearwise', file_dir=os.path.join(DATA_DIR, 'rain'))
tmax_data = imd.open_data('tmax', START_YEAR, END_YEAR, 'yearwise', file_dir=os.path.join(DATA_DIR, 'tmax'))
tmin_data = imd.open_data('tmin', START_YEAR, END_YEAR, 'yearwise', file_dir=os.path.join(DATA_DIR, 'tmin'))

rain_ds = rain_data.get_xarray()
tmax_ds = tmax_data.get_xarray()
tmin_ds = tmin_data.get_xarray()

# ----------------------------------------------------------------------
# 3. Load district boundaries
# ----------------------------------------------------------------------

print("Loading district boundaries...")
districts = gpd.read_file(DISTRICTS_GEOJSON)
districts = districts.to_crs("EPSG:4326")


def raster_transform_from_grid(lats, lons):
    """Build an affine transform for a regular lat/lon grid (IMD is 0.25deg)."""
    res_lat = abs(lats[1] - lats[0])
    res_lon = abs(lons[1] - lons[0])
    # origin = top-left corner (max lat, min lon)
    return from_origin(lons.min() - res_lon / 2, lats.max() + res_lat / 2, res_lon, res_lat)


def daily_to_district(ds, varname, agg="mean", nodata_sentinels=(-999.0,)):
    """
    For each day in the dataset, compute the district-level zonal
    statistic (mean rainfall, mean tmax, etc.) and return a long-format
    DataFrame: District, State, date, value.

    nodata_sentinels: values in the raw grid that mean "missing", which
    are masked out before computing stats. Kept per-variable (rather than
    one universal list) because a broad list applied to rainfall would
    incorrectly null out legitimate values like a real 99.9mm rain day.
    """
    lats = ds['lat'].values
    lons = ds['lon'].values
    transform = raster_transform_from_grid(lats, lons)

    records = []
    times = ds['time'].values
    arr = ds[varname].values  # shape depends on lat/lon/time ordering

    for i, t in enumerate(times):
        day_slice = ds[varname].isel(time=i).values
        # flip vertically if lat is ascending (rasterio expects north-up)
        if lats[0] < lats[-1]:
            day_slice = np.flipud(day_slice)
        # IMD's nodata/missing-value sentinel varies by product/version --
        # mask out this variable's known sentinel(s). Kept per-variable
        # (see nodata_sentinels docstring above) so rainfall's real values
        # near common sentinel numbers (e.g. a genuine 99.9mm rain day)
        # are never wrongly treated as missing.
        for sentinel in nodata_sentinels:
            day_slice = np.where(np.isclose(day_slice, sentinel), np.nan, day_slice)

        stats = zonal_stats(
            districts, day_slice, affine=transform, stats=[agg], nodata=np.nan,
            all_touched=True,  # CRITICAL: without this, small districts that
            # don't fully contain any single pixel CENTER get silently
            # skipped -- this is exactly what caused 478/724 districts to
            # come back 100% null for tmax/tmin (whose 1x1 degree grid is
            # much coarser than rainfall's 0.25x0.25 degree grid). With
            # all_touched=True, any pixel the polygon overlaps at all counts.
        )
        for j, s in enumerate(stats):
            records.append({
                "State": districts.iloc[j]["st_nm"],
                "District": districts.iloc[j]["district"],
                "date": pd.Timestamp(t),
                "value": s[agg],
            })

        if (i + 1) % 100 == 0:
            print(f"  {varname}: processed {i+1}/{len(times)} days")

    return pd.DataFrame(records)


print("Computing district-level daily rainfall (this is the slow step)...")
rain_district = daily_to_district(rain_ds, "rain", agg="mean", nodata_sentinels=(-999.0,))
rain_district = rain_district.rename(columns={"value": "Rainfall_mm"})

print("Computing district-level daily tmax...")
tmax_district = daily_to_district(tmax_ds, "tmax", agg="mean", nodata_sentinels=(-999.0, 99.9, -99.9))
tmax_district = tmax_district.rename(columns={"value": "Tmax_C"})

print("Computing district-level daily tmin...")
tmin_district = daily_to_district(tmin_ds, "tmin", agg="mean", nodata_sentinels=(-999.0, 99.9, -99.9))
tmin_district = tmin_district.rename(columns={"value": "Tmin_C"})

# ----------------------------------------------------------------------
# 4. Merge and aggregate to weekly
# ----------------------------------------------------------------------

print("Merging and aggregating to weekly...")
merged = rain_district.merge(
    tmax_district, on=["State", "District", "date"]
).merge(
    tmin_district, on=["State", "District", "date"]
)

merged["Year"] = merged["date"].dt.isocalendar().year
merged["Week"] = merged["date"].dt.isocalendar().week

weekly = merged.groupby(["State", "District", "Year", "Week"]).agg(
    Rainfall_mm=("Rainfall_mm", "sum"),   # sum of daily rainfall over the week
    Tmax_C=("Tmax_C", "mean"),
    Tmin_C=("Tmin_C", "mean"),
).reset_index()

weekly.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Saved {len(weekly)} district-week rows to {OUTPUT_CSV}")
print("Please upload this CSV back to continue.")