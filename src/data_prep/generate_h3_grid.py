"""
Generate an H3 hexagon grid at resolution 7 (~5.2 km^2 per cell, ~635,000
hexagons over India) covering the whole country, with each hexagon tagged
with its district and state -- the base spatial unit everything else
(population, weather, disaggregated case counts) will eventually join
against.

WHY RESOLUTION 7:
Coarser resolutions (6 and below) don't meaningfully beat district-level
granularity for our purposes. Finer (8+) starts to outrun what our real
covariates can actually distinguish (population is 1km resolution,
rainfall ~25km, temperature ~100km) -- going finer would just multiply
row count without adding genuine spatial information. Resolution 7 sits
close to the practical ceiling of what our data can support.

SETUP:
    pip install h3 geopandas shapely pandas

PROJECT STRUCTURE ASSUMED (paths are resolved relative to this script's
own location, not the current working directory, so this runs correctly
regardless of where you launch python from):
    VectorHotspot/
    ├── data/
    │   ├── boundaries/india_districts_clean.geojson   <- INPUT (must exist)
    │   └── processed/india_h3_grid_res7.csv           <- OUTPUT (created)
    └── src/
        └── data_prep/generate_h3_grid.py              <- THIS FILE

OUTPUT:
    data/processed/india_h3_grid_res7.csv
    Columns: h3_index, center_lat, center_lon, district, state
"""

import h3
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

RESOLUTION = 7

# Resolve paths relative to THIS FILE's location (src/data_prep/), not the
# current working directory -- so it works whether you run it from the
# project root, from src/data_prep/, or anywhere else.
SCRIPT_DIR = Path(__file__).resolve().parent          # .../src/data_prep
PROJECT_ROOT = SCRIPT_DIR.parent.parent                # .../VectorHotspot
DISTRICTS_GEOJSON = PROJECT_ROOT / "data" / "boundaries" / "india_districts_clean.geojson"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "india_h3_grid_res7.csv"

if not DISTRICTS_GEOJSON.exists():
    raise FileNotFoundError(
        f"Could not find {DISTRICTS_GEOJSON}\n"
        f"Expected it at data/boundaries/india_districts_clean.geojson "
        f"relative to the project root ({PROJECT_ROOT}).\n"
        f"Make sure the file structure matches what this script expects."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load districts, build a single unioned India boundary
# ----------------------------------------------------------------------

print("Loading district boundaries...")
districts = gpd.read_file(DISTRICTS_GEOJSON).to_crs("EPSG:4326").reset_index(drop=True)
print(f"  {len(districts)} districts loaded")

print("Computing India-wide union geometry (may take a moment)...")
try:
    india_union = districts.geometry.union_all()  # newer geopandas
except AttributeError:
    india_union = districts.geometry.unary_union  # older geopandas


def polygon_to_h3_set(geom, resolution):
    """Convert a shapely Polygon/MultiPolygon to a set of H3 cell indices
    using the h3-py v4 API. Handles MultiPolygon by processing each part
    and holes (interior rings, e.g. enclaves) separately."""
    cells = set()
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        exterior = [(lat, lon) for lon, lat in poly.exterior.coords]
        holes = [[(lat, lon) for lon, lat in interior.coords] for interior in poly.interiors]
        h3poly = h3.LatLngPoly(exterior, *holes)
        cells.update(h3.polygon_to_cells(h3poly, resolution))
    return cells


print(f"Generating resolution-{RESOLUTION} hexagons over all of India "
      f"(this may take a few minutes for a geometry this complex)...")
all_cells = polygon_to_h3_set(india_union, RESOLUTION)
print(f"  Generated {len(all_cells)} hexagons")

# ----------------------------------------------------------------------
# 2. Compute hexagon centers
# ----------------------------------------------------------------------

print("Computing hexagon centers...")
records = []
for cell in all_cells:
    lat, lon = h3.cell_to_latlng(cell)
    records.append({"h3_index": cell, "center_lat": lat, "center_lon": lon})
hex_df = pd.DataFrame(records)

# ----------------------------------------------------------------------
# 3. Assign each hexagon to a district via point-in-polygon spatial join
# ----------------------------------------------------------------------

print("Assigning hexagons to districts (spatial join)...")
hex_gdf = gpd.GeoDataFrame(
    hex_df,
    geometry=gpd.points_from_xy(hex_df["center_lon"], hex_df["center_lat"]),
    crs="EPSG:4326",
)
joined = gpd.sjoin(
    hex_gdf, districts[["st_nm", "district", "geometry"]],
    how="left", predicate="within",
)
joined = joined.drop(columns=["index_right"], errors="ignore")

n_unassigned = joined["district"].isna().sum()
print(f"  Hexagons with no direct district match "
      f"(coastal/boundary edge cases): {n_unassigned}")

# Fallback: nearest district centroid for hexagons whose center fell
# just outside every polygon (coastline simplification, etc.)
if n_unassigned > 0:
    print("  Assigning fallback district via nearest centroid...")
    district_centroids = districts.copy()
    district_centroids["centroid"] = district_centroids.geometry.centroid
    unassigned_idx = joined[joined["district"].isna()].index
    for idx in unassigned_idx:
        pt = joined.loc[idx, "geometry"]
        dists = district_centroids["centroid"].distance(pt)
        nearest = district_centroids.loc[dists.idxmin()]
        joined.loc[idx, "district"] = nearest["district"]
        joined.loc[idx, "st_nm"] = nearest["st_nm"]

joined = joined.rename(columns={"st_nm": "state"})
result = joined[["h3_index", "center_lat", "center_lon", "district", "state"]].copy()

# ----------------------------------------------------------------------
# 4. Ensure every district has at least 1 hexagon (tiny UTs etc.)
# ----------------------------------------------------------------------

districts_with_hex = set(result["district"].unique())
all_districts = set(districts["district"].unique())
missing_districts = all_districts - districts_with_hex
print(f"\nDistricts with ZERO hexagons "
      f"(smaller than a single resolution-{RESOLUTION} cell): {len(missing_districts)}")

if missing_districts:
    print(f"  Adding a fallback centroid-snapped hexagon for: {sorted(missing_districts)}")
    extra_records = []
    for dname in missing_districts:
        drow = districts[districts["district"] == dname].iloc[0]
        centroid = drow.geometry.centroid
        cell = h3.latlng_to_cell(centroid.y, centroid.x, RESOLUTION)
        extra_records.append({
            "h3_index": cell, "center_lat": centroid.y, "center_lon": centroid.x,
            "district": dname, "state": drow["st_nm"],
        })
    result = pd.concat([result, pd.DataFrame(extra_records)], ignore_index=True)

result = result.drop_duplicates(subset="h3_index", keep="first")

result.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(result)} hexagons to {OUTPUT_CSV}")

# ----------------------------------------------------------------------
# 5. Validation summary
# ----------------------------------------------------------------------

print("\n=== VALIDATION ===")
print(f"Total hexagons: {len(result)}")
print(f"Distinct districts covered: {result['district'].nunique()} / {len(districts)}")
print(f"Distinct states covered: {result['state'].nunique()} / {districts['st_nm'].nunique()}")
print(f"Duplicate h3_index: {result['h3_index'].duplicated().sum()}")
print(f"Hexagons per district (min/median/max): "
      f"{result.groupby('district').size().min()} / "
      f"{result.groupby('district').size().median()} / "
      f"{result.groupby('district').size().max()}")
print("\nPlease upload india_h3_grid_res7.csv back to continue.")