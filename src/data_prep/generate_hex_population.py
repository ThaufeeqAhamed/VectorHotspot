#!/usr/bin/env python3
"""
Generate per-hexagon population estimates from WorldPop 1km resolution rasters.

Uses a fast bulk rasterization approach (rasterize all hexagons at once, then
bincount) rather than per-hexagon zonal_stats, which would take 50+ minutes
per year at 620k+ hexagons.

IMPORTANT CONTEXT (from handover debugging history):
The initial version used rasterstats.zonal_stats() per hexagon, which took
52 minutes for year 2000 alone (projected 4+ hours total). This version uses
bulk rasterize + np.bincount, completing all 5 years in minutes.

Known limitation: The fast method assigns each raster pixel to exactly one
hexagon ("winner takes all" for boundary pixels). This causes a small
systematic edge bias, but cross-validation showed it's acceptable (0.15%
national error, 75% of districts within 1.2%).

SETUP:
    pip install rasterio geopandas pandas numpy shapely h3

INPUT:
    - data/processed/india_h3_grid_res7.csv (620,742 hexagons with geometries)
    - data/raw/population/ind_ppp_2000_1km_Aggregated.tif
    - data/raw/population/ind_ppp_2005_1km_Aggregated.tif
    - data/raw/population/ind_ppp_2010_1km_Aggregated.tif
    - data/raw/population/ind_ppp_2015_1km_Aggregated.tif
    - data/raw/population/ind_ppp_2020_1km_Aggregated.tif

OUTPUT:
    - data/processed/hex_population_2000_2020.csv
      Columns: h3_index, population_2000, population_2005, ..., population_2020

WORLDPOP DATA SOURCE:
    Website: https://hub.worldpop.org/geodata/listing?id=29
    Dataset: "India - Population - 1km resolution"
    Direct links (1km Aggregated, Constrained):
    - 2000: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2000/IND/ind_ppp_2000_1km_Aggregated.tif
    - 2005: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2005/IND/ind_ppp_2005_1km_Aggregated.tif
    - 2010: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2010/IND/ind_ppp_2010_1km_Aggregated.tif
    - 2015: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2015/IND/ind_ppp_2015_1km_Aggregated.tif
    - 2020: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif

    Files are ~18MB each. Download and place in data/raw/population/ before running.

NOTE ON 100m vs 1km RESOLUTION:
WorldPop offers both 100m (~1.7GB/year) and 1km (~18MB/year) products.
We use 1km because:
1. Population doesn't vary meaningfully week-to-week for our use case
2. 1km is already finer than most of our other covariates
3. The 100x smaller file size makes the pipeline more practical
4. Our H3 hexagons are ~2.3km across, so 1km input is appropriate
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from pathlib import Path
import h3
from shapely.geometry import Polygon

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# WorldPop file patterns
POPULATION_YEARS = [2000, 2005, 2010, 2015, 2020]
POPULATION_DIR = PROJECT_ROOT / 'data' / 'raw' / 'population'


def hex_to_polygon(h3_index):
    """Convert H3 index to Shapely Polygon."""
    boundary = h3.cell_to_boundary(h3_index)
    # h3.cell_to_boundary returns (lat, lng) tuples, but Shapely needs (lng, lat)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def extract_hex_population_fast(hex_gdf, raster_path, year):
    """
    Fast bulk extraction of population for all hexagons using rasterize + bincount.

    Args:
        hex_gdf: GeoDataFrame with h3_index and geometry (polygons in WGS84)
        raster_path: Path to WorldPop GeoTIFF
        year: Year for logging

    Returns:
        Series with h3_index as index and population as values
    """
    print(f"  Processing year {year}...")

    with rasterio.open(raster_path) as src:
        print(f"    Raster shape: {src.shape}, CRS: {src.crs}")

        # Reproject hexagons to match raster CRS if needed
        if hex_gdf.crs != src.crs:
            hex_gdf_proj = hex_gdf.to_crs(src.crs)
        else:
            hex_gdf_proj = hex_gdf

        # Create a mapping: hexagon_id -> integer index (for rasterization)
        hex_gdf_proj = hex_gdf_proj.reset_index(drop=True)
        hex_gdf_proj['hex_id'] = range(len(hex_gdf_proj))

        print(f"    Rasterizing {len(hex_gdf_proj):,} hexagon polygons...")

        # Bulk rasterize all hexagons at once
        # Each pixel gets assigned to one hexagon (the one that covers it)
        # all_touched=True means pixels touching hexagon boundary are included
        shapes = [(geom, hex_id) for geom, hex_id in
                  zip(hex_gdf_proj.geometry, hex_gdf_proj['hex_id'])]

        hex_raster = features.rasterize(
            shapes=shapes,
            out_shape=src.shape,
            transform=src.transform,
            fill=-1,  # Pixels not in any hexagon
            all_touched=True,
            dtype=np.int32
        )

        print(f"    Reading population raster...")
        pop_data = src.read(1)  # Read band 1

        # Mask out nodata values
        nodata = src.nodata
        if nodata is not None:
            valid_mask = pop_data != nodata
        else:
            valid_mask = np.ones_like(pop_data, dtype=bool)

        # Also mask out negative populations (if any)
        valid_mask &= pop_data >= 0

        # Flatten arrays
        hex_ids_flat = hex_raster[valid_mask].astype(np.int32)
        pop_flat = pop_data[valid_mask]

        # Filter out pixels not in any hexagon (hex_id == -1)
        in_hex_mask = hex_ids_flat >= 0
        hex_ids_flat = hex_ids_flat[in_hex_mask]
        pop_flat = pop_flat[in_hex_mask]

        print(f"    Aggregating population via bincount...")
        # Sum population per hexagon using bincount
        # bincount requires non-negative integer indices
        hex_population = np.bincount(hex_ids_flat, weights=pop_flat, minlength=len(hex_gdf_proj))

        # Create result series
        result = pd.Series(hex_population, index=hex_gdf_proj['h3_index'])
        result.name = f'population_{year}'

        total_pop = result.sum()
        print(f"    Year {year} total population: {total_pop:,.0f}")
        print(f"    Hexagons with population > 0: {(result > 0).sum():,} / {len(result):,}")

        return result


def validate_against_district_totals(hex_pop_df, hex_grid_df, district_pop_df, year):
    """
    Cross-validate hexagon population aggregates against known district totals.

    Args:
        hex_pop_df: DataFrame with h3_index and population_YYYY
        hex_grid_df: DataFrame with h3_index, district, state
        district_pop_df: Long-format DataFrame with State, District, Year, Population
        year: Year to validate

    Returns:
        DataFrame with validation results per district
    """
    pop_col = f'population_{year}'

    # Join hex population with district assignments
    hex_with_district = hex_grid_df[['h3_index', 'district', 'state']].merge(
        hex_pop_df[['h3_index', pop_col]], on='h3_index', how='inner'
    )

    # Aggregate hexagon population to district level
    hex_district_totals = (hex_with_district
                           .groupby(['state', 'district'])
                           [pop_col]
                           .sum()
                           .reset_index()
                           .rename(columns={pop_col: 'hex_total'}))

    # Filter district pop to the target year and normalize column names
    # district_pop_df is long-format: State, District, Year, Population
    year_district_pop = (district_pop_df[district_pop_df['Year'] == year]
                         [['State', 'District', 'Population']]
                         .rename(columns={'State': 'state', 'District': 'district',
                                          'Population': 'known_total'})
                         .copy())

    # Merge with known district totals
    validation = hex_district_totals.merge(
        year_district_pop,
        on=['state', 'district'],
        how='inner'
    )

    # Calculate errors
    validation['abs_error'] = abs(validation['hex_total'] - validation['known_total'])
    validation['pct_error'] = 100 * validation['abs_error'] / validation['known_total'].replace(0, np.nan)

    return validation


def main():
    """Main execution function."""
    print("=" * 60)
    print("VectorHotspot: Generate Hexagon-Level Population")
    print("=" * 60)

    # Check for required input files
    print("\nChecking for required files...")

    # H3 grid
    h3_grid_path = PROJECT_ROOT / 'data' / 'processed' / 'india_h3_grid_res7.csv'
    if not h3_grid_path.exists():
        raise FileNotFoundError(f"H3 grid file not found: {h3_grid_path}")
    print(f"  ✓ H3 grid: {h3_grid_path}")

    # District population (for validation)
    district_pop_path = PROJECT_ROOT / 'data' / 'processed' / 'district_population_2000_2020.csv'
    has_district_pop = district_pop_path.exists()
    if has_district_pop:
        print(f"  ✓ District population: {district_pop_path}")
    else:
        print(f"  ⚠ District population not found (validation will be skipped): {district_pop_path}")

    # WorldPop rasters
    missing_rasters = []
    for year in POPULATION_YEARS:
        raster_path = POPULATION_DIR / f'ind_ppp_{year}_1km_Aggregated.tif'
        if raster_path.exists():
            print(f"  ✓ {year}: {raster_path}")
        else:
            print(f"  ✗ {year}: {raster_path} NOT FOUND")
            missing_rasters.append(year)

    if missing_rasters:
        print("\n" + "!" * 60)
        print("ERROR: Missing WorldPop raster files for years:", missing_rasters)
        print("\nPlease download the missing files from:")
        print("https://hub.worldpop.org/geodata/listing?id=29")
        print("\nDirect download URLs:")
        for year in missing_rasters:
            print(f"  {year}: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/{year}/IND/ind_ppp_{year}_1km_Aggregated.tif")
        print(f"\nPlace downloaded .tif files in: {POPULATION_DIR}")
        print("!" * 60)
        return

    # Load H3 grid
    print(f"\nLoading H3 grid from {h3_grid_path}...")
    h3_grid = pd.read_csv(h3_grid_path, encoding='utf-8-sig')
    h3_grid.columns = h3_grid.columns.str.strip()
    print(f"  Loaded {len(h3_grid):,} hexagons")

    # Convert to GeoDataFrame with geometries
    print("  Converting H3 indices to polygons...")
    h3_grid['geometry'] = h3_grid['h3_index'].apply(hex_to_polygon)
    hex_gdf = gpd.GeoDataFrame(h3_grid, geometry='geometry', crs='EPSG:4326')

    # Extract population for each year
    print("\nExtracting population from WorldPop rasters...")
    population_series = {}

    for year in POPULATION_YEARS:
        raster_path = POPULATION_DIR / f'ind_ppp_{year}_1km_Aggregated.tif'
        pop_series = extract_hex_population_fast(hex_gdf, raster_path, year)
        population_series[year] = pop_series

    # Combine all years into one DataFrame
    print("\nCombining results...")
    result_df = pd.DataFrame({'h3_index': h3_grid['h3_index']})

    for year, pop_series in population_series.items():
        result_df = result_df.merge(
            pop_series.reset_index().rename(columns={'h3_index': 'h3_index', 0: f'population_{year}'}),
            on='h3_index',
            how='left'
        )

    # Fill any NaNs with 0 (hexagons with no population)
    pop_cols = [f'population_{year}' for year in POPULATION_YEARS]
    result_df[pop_cols] = result_df[pop_cols].fillna(0)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for year in POPULATION_YEARS:
        col = f'population_{year}'
        total = result_df[col].sum()
        nonzero = (result_df[col] > 0).sum()
        print(f"{year}: {total:>15,.0f} total population, {nonzero:>7,} hexagons with pop > 0")

    # Validation against district totals (if available)
    if has_district_pop:
        print("\n" + "=" * 60)
        print("VALIDATION: Hexagon Totals vs. Known District Totals")
        print("=" * 60)

        district_pop = pd.read_csv(district_pop_path, encoding='utf-8-sig')
        district_pop.columns = district_pop.columns.str.strip()

        for year in POPULATION_YEARS:
            print(f"\n{year}:")
            validation = validate_against_district_totals(
                result_df, h3_grid, district_pop, year
            )

            # National-level comparison
            hex_national = result_df[f'population_{year}'].sum()
            known_national = district_pop[district_pop['Year'] == year]['Population'].sum()
            national_error = 100 * abs(hex_national - known_national) / known_national

            print(f"  National: hex={hex_national:,.0f}, known={known_national:,.0f}, error={national_error:.2f}%")

            # District-level statistics
            print(f"  Districts matched: {len(validation):,}")
            print(f"  Mean % error: {validation['pct_error'].mean():.2f}%")
            print(f"  Median % error: {validation['pct_error'].median():.2f}%")
            print(f"  75th percentile: {validation['pct_error'].quantile(0.75):.2f}%")
            print(f"  95th percentile: {validation['pct_error'].quantile(0.95):.2f}%")
            print(f"  Max % error: {validation['pct_error'].max():.2f}%")

            # Districts with largest errors
            worst = validation.nlargest(5, 'pct_error')[['state', 'district', 'pct_error']]
            if len(worst) > 0:
                print(f"  Top 5 largest % errors:")
                for _, row in worst.iterrows():
                    print(f"    {row['state']}, {row['district']}: {row['pct_error']:.2f}%")

    # Save results
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'hex_population_2000_2020.csv'
    print(f"\nSaving results to {output_path}...")
    result_df.to_csv(output_path, index=False, encoding='utf-8')

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(result_df):,}")
    print(f"  Columns: {len(result_df.columns)}")

    print("\n" + "=" * 60)
    print("SUCCESS: Hexagon population generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
