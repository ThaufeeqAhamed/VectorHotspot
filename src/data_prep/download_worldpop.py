#!/usr/bin/env python3
"""
Download WorldPop 1km resolution population rasters for India.

This script downloads the 5 required WorldPop GeoTIFF files (2000, 2005, 2010,
2015, 2020) from the WorldPop FTP server.

Each file is ~18MB. Total download: ~90MB.

SETUP:
    pip install requests tqdm

OUTPUT:
    Files saved to: data/raw/population/
    - ind_ppp_2000_1km_Aggregated.tif
    - ind_ppp_2005_1km_Aggregated.tif
    - ind_ppp_2010_1km_Aggregated.tif
    - ind_ppp_2015_1km_Aggregated.tif
    - ind_ppp_2020_1km_Aggregated.tif
"""

import requests
from pathlib import Path
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'population'

# WorldPop URLs
WORLDPOP_BASE = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km"
YEARS = [2000, 2005, 2010, 2015, 2020]


def download_file(url, dest_path, desc):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        dest_path: Path to save file to
        desc: Description for progress bar
    """
    # Check if file already exists
    if dest_path.exists():
        file_size = dest_path.stat().st_size
        print(f"  ✓ {dest_path.name} already exists ({file_size / (1024*1024):.1f} MB)")
        return True

    print(f"  Downloading {desc}...")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"    {dest_path.name}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"  ✓ Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False


def main():
    """Main download function."""
    print("=" * 60)
    print("WorldPop Data Download for VectorHotspot")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Download each year
    print(f"\nDownloading {len(YEARS)} WorldPop raster files (~18 MB each)...")
    print("This may take several minutes depending on your connection.\n")

    success_count = 0
    failed_years = []

    for year in YEARS:
        filename = f"ind_ppp_{year}_1km_Aggregated.tif"
        url = f"{WORLDPOP_BASE}/{year}/IND/{filename}"
        dest_path = OUTPUT_DIR / filename

        if download_file(url, dest_path, f"{year}"):
            success_count += 1
        else:
            failed_years.append(year)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successfully downloaded/verified: {success_count}/{len(YEARS)} files")

    if failed_years:
        print(f"\n⚠ Failed to download years: {failed_years}")
        print("\nYou can manually download these files from:")
        for year in failed_years:
            filename = f"ind_ppp_{year}_1km_Aggregated.tif"
            url = f"{WORLDPOP_BASE}/{year}/IND/{filename}"
            print(f"  {year}: {url}")
        print(f"\nPlace downloaded files in: {OUTPUT_DIR}")
    else:
        print("\n✓ All files ready!")
        print(f"\nNext step: Run generate_hex_population.py to extract per-hexagon population")

    print("=" * 60)


if __name__ == '__main__':
    main()
