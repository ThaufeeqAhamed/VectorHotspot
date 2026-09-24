#!/usr/bin/env python3
"""
Setup verification script for VectorHotspot Phase 5.

Checks that all required files and dependencies are in place before running
the disaggregation pipeline.
"""

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

def check_file(path, description, required=True):
    """Check if a file exists and report status."""
    exists = path.exists()
    size_mb = path.stat().st_size / (1024 * 1024) if exists else 0

    if exists:
        print(f"  ✓ {description}")
        if size_mb > 1:
            print(f"    ({size_mb:.1f} MB)")
        return True
    else:
        symbol = "✗" if required else "⚠"
        status = "MISSING (REQUIRED)" if required else "MISSING (optional)"
        print(f"  {symbol} {description}: {status}")
        print(f"    Expected at: {path}")
        return False

def check_dependency(module_name, import_name=None):
    """Check if a Python module is installed."""
    if import_name is None:
        import_name = module_name

    try:
        __import__(import_name)
        print(f"  ✓ {module_name}")
        return True
    except ImportError:
        print(f"  ✗ {module_name}: NOT INSTALLED")
        return False

def main():
    """Main verification function."""
    print("=" * 70)
    print("VectorHotspot Phase 5 Setup Verification")
    print("=" * 70)

    all_good = True

    # Check Python dependencies
    print("\n1. Checking Python dependencies...")
    deps = [
        ('pandas', None),
        ('numpy', None),
        ('scipy', None),
        ('geopandas', None),
        ('shapely', None),
        ('h3', None),
        ('rasterio', None),
        ('requests', None),
        ('tqdm', None)
    ]

    for module_name, import_name in deps:
        if not check_dependency(module_name, import_name):
            all_good = False

    # Check required data files
    print("\n2. Checking required data files...")

    required_files = [
        (PROJECT_ROOT / 'data' / 'processed' / 'dengue_state_2010_2024.csv',
         'Dengue state-level data'),
        (PROJECT_ROOT / 'data' / 'processed' / 'malaria_district_2000_2024.csv',
         'Malaria district-level data (corrected)'),
        (PROJECT_ROOT / 'data' / 'processed' / 'india_h3_grid_res7.csv',
         'H3 hexagon grid (620,742 hexagons)'),
        (PROJECT_ROOT / 'data' / 'processed' / 'district_population_2000_2020.csv',
         'District-level population'),
        (PROJECT_ROOT / 'data' / 'boundaries' / 'india_districts_clean.geojson',
         'District boundaries'),
    ]

    for path, desc in required_files:
        if not check_file(path, desc, required=True):
            all_good = False

    # Check WorldPop rasters
    print("\n3. Checking WorldPop population rasters...")
    worldpop_dir = PROJECT_ROOT / 'data' / 'raw' / 'population'
    years = [2000, 2005, 2010, 2015, 2020]

    worldpop_missing = []
    for year in years:
        path = worldpop_dir / f'ind_ppp_{year}_1km_Aggregated.tif'
        if not check_file(path, f'{year} population raster', required=True):
            worldpop_missing.append(year)
            all_good = False

    # Check for hex population (generated file)
    print("\n4. Checking generated files...")
    hex_pop_path = PROJECT_ROOT / 'data' / 'processed' / 'hex_population_2000_2020.csv'
    hex_pop_exists = check_file(hex_pop_path, 'Hexagon-level population', required=False)

    disagg_report_path = PROJECT_ROOT / 'data' / 'processed' / 'disaggregation_fit_report.txt'
    disagg_report_exists = check_file(disagg_report_path, 'Disaggregation fit report', required=False)

    # Summary and next steps
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    if all_good:
        print("\n✓ All required files and dependencies are present!")

        if not hex_pop_exists:
            print("\n→ NEXT STEP: Generate hexagon-level population")
            print("  Run: python src/data_prep/generate_hex_population.py")
        elif not disagg_report_exists:
            print("\n→ NEXT STEP: Run spatial disaggregation model")
            print("  Run: python src/disaggregation/wire_disaggregation_model.py")
        else:
            print("\n✓ Phase 5 appears to be complete!")
            print("  All expected outputs are present.")
            print("\n→ NEXT STEP: Decide on Phase 6 direction")
            print("  Option A: Pull Tier 2 covariates (NDVI, land cover, water)")
            print("  Option B: Design temporal disaggregation (annual → weekly)")
    else:
        print("\n✗ Some required files or dependencies are missing.")

        if worldpop_missing:
            print("\n→ NEXT STEP: Download WorldPop rasters")
            print("  Run: python src/data_prep/download_worldpop.py")
            print("\n  Or download manually from:")
            for year in worldpop_missing:
                url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/{year}/IND/ind_ppp_{year}_1km_Aggregated.tif"
                print(f"    {year}: {url}")
            print(f"\n  Place files in: {worldpop_dir}")

        print("\nMissing Python dependencies can be installed with:")
        print("  pip install pandas numpy scipy geopandas shapely h3 rasterio requests tqdm")

    print("=" * 70)

    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
