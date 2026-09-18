"""
Compute district-level population totals from WorldPop 1km GeoTIFFs,
using a pure-Python zonal-statistics approach (no geopandas/rasterio/GDAL
available in this environment).

Method: for each district polygon, find the raster pixels whose centers
fall inside it (vectorized point-in-polygon test via matplotlib.path.Path),
and sum their population values.
"""

import json
import numpy as np
import tifffile
from matplotlib.path import Path
import pandas as pd

DISTRICTS_GEOJSON = "/mnt/user-data/outputs/india_districts_clean.geojson"
YEARS = [2000, 2005, 2010, 2015, 2020]
TIF_TEMPLATE = "/mnt/user-data/uploads/ind_ppp_{year}_1km_Aggregated.tif"

# ----------------------------------------------------------------------
# Load district boundaries
# ----------------------------------------------------------------------

with open(DISTRICTS_GEOJSON) as f:
    districts_gj = json.load(f)

print(f"Loaded {len(districts_gj['features'])} district features")


def get_rings(geometry):
    """
    Return a list of (exterior_ring, [hole_rings]) tuples for a
    Polygon or MultiPolygon geometry, as lists of (lon, lat) tuples.
    """
    polys = []
    if geometry["type"] == "Polygon":
        coords = geometry["coordinates"]
        polys.append((coords[0], coords[1:]))
    elif geometry["type"] == "MultiPolygon":
        for coords in geometry["coordinates"]:
            polys.append((coords[0], coords[1:]))
    return polys


# ----------------------------------------------------------------------
# Process each year's raster
# ----------------------------------------------------------------------

all_results = {}  # year -> list of dicts

for year in YEARS:
    tif_path = TIF_TEMPLATE.format(year=year)
    print(f"\nProcessing {year}: {tif_path}")

    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        scale = page.tags["ModelPixelScaleTag"].value
        tiepoint = page.tags["ModelTiepointTag"].value
        nodata = float(page.tags["GDAL_NODATA"].value)

    from PIL import Image
    arr = np.array(Image.open(tif_path))

    px_scale_x, px_scale_y = scale[0], scale[1]
    origin_lon, origin_lat = tiepoint[3], tiepoint[4]
    n_rows, n_cols = arr.shape

    arr = np.where(arr == nodata, 0.0, arr)  # treat nodata as 0 population

    def lonlat_to_colrow(lon, lat):
        col = (lon - origin_lon) / px_scale_x
        row = (origin_lat - lat) / px_scale_y
        return col, row

    year_records = []
    for i, feat in enumerate(districts_gj["features"]):
        props = feat["properties"]
        state = props["st_nm"]
        district = props["district"]
        polys = get_rings(feat["geometry"])

        total_pop = 0.0
        for exterior, holes in polys:
            lons = [pt[0] for pt in exterior]
            lats = [pt[1] for pt in exterior]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)

            c0, r0 = lonlat_to_colrow(min_lon, max_lat)
            c1, r1 = lonlat_to_colrow(max_lon, min_lat)
            col_start = max(0, int(np.floor(c0)) - 1)
            col_end = min(n_cols, int(np.ceil(c1)) + 1)
            row_start = max(0, int(np.floor(r0)) - 1)
            row_end = min(n_rows, int(np.ceil(r1)) + 1)

            if col_end <= col_start or row_end <= row_start:
                continue

            sub = arr[row_start:row_end, col_start:col_end]
            rows_idx = np.arange(row_start, row_end)
            cols_idx = np.arange(col_start, col_end)
            col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)

            lon_grid = origin_lon + (col_grid + 0.5) * px_scale_x
            lat_grid = origin_lat - (row_grid + 0.5) * px_scale_y
            points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

            ext_path = Path(exterior)
            mask = ext_path.contains_points(points)

            for hole in holes:
                hole_path = Path(hole)
                mask &= ~hole_path.contains_points(points)

            mask = mask.reshape(sub.shape)
            total_pop += sub[mask].sum()

        year_records.append({
            "State": state,
            "District": district,
            "Year": year,
            "Population": round(float(total_pop), 1),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(districts_gj['features'])} districts processed")

    all_results[year] = year_records
    year_total = sum(r["Population"] for r in year_records)
    print(f"  Year {year} total population across all districts: {year_total:,.0f}")

# ----------------------------------------------------------------------
# Combine into final long-format CSV
# ----------------------------------------------------------------------

all_records = [r for recs in all_results.values() for r in recs]
df = pd.DataFrame(all_records)
df.to_csv("/home/claude/district_population_2000_2020.csv", index=False)
print(f"\nSaved {len(df)} rows to district_population_2000_2020.csv")
