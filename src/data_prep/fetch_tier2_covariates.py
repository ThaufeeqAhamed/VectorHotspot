#!/usr/bin/env python3
"""
Fetch and process Tier 2 environmental covariates for VectorHotspot Phase 6.

OPTIMIZED VERSION: uses checkpointing, parallel downloads, and a lightweight
MODIS approach (annual mean GeoTIFF via AppEEARS REST API instead of 483 HDF
granules). Survives shutdowns and resumes from where it left off.

Downloads and extracts per-hexagon values for:
  1. MODIS MOD13A2 NDVI (mean 2018-2020, 1km) -- via AppEEARS annual mean GeoTIFF
  2. ESA WorldCover 2021 (land cover fractions, 10m) -- public AWS, parallel
  3. JRC Global Surface Water Occurrence (1984-2021, 30m) -- public HTTPS

OUTPUT: data/processed/hex_tier2_covariates.csv
Columns: h3_index, ndvi_mean, frac_water, frac_trees, frac_built, frac_shrub, jrc_occurrence

SETUP:
    pip install earthaccess rasterio geopandas h3 numpy pandas requests tqdm

AUTHENTICATION:
  MODIS via AppEEARS requires a free NASA Earthdata account.
  Credentials must be in C:\\Users\\<you>\\_netrc:
    machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from pathlib import Path
import h3
from shapely.geometry import Polygon, box
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_NDVI_DIR = PROJECT_ROOT / 'data' / 'raw' / 'ndvi'
RAW_WC_DIR   = PROJECT_ROOT / 'data' / 'raw' / 'worldcover'
RAW_JRC_DIR  = PROJECT_ROOT / 'data' / 'raw' / 'jrc_water'
OUTPUT_DIR   = PROJECT_ROOT / 'data' / 'processed'

# Checkpoint files — track completed tiles so restarts skip them
CHECKPOINT_DIR  = PROJECT_ROOT / 'data' / 'raw' / 'checkpoints'

# India bounding box
INDIA_BBOX = (67.0, 6.0, 98.0, 38.0)

# ESA WorldCover: 3-degree tile SW corners covering India
def _worldcover_tiles():
    tiles = []
    for lat in range(6, 37, 3):
        for lon in range(66, 99, 3):
            tiles.append((lon, lat))
    return tiles

# JRC Global Surface Water Occurrence: official 10-degree tile index scheme.
# Index formula: row = (90 - top_lat) * 2000, col = (left_lon + 180) * 4000
# Only 8 tiles overlap India (verified from the JEODPP server listing):
# lat bands 0-10N and 20-30N, lon bands 60-100E
def _jrc_tiles():
    tiles = []
    for top_lat in [30, 10]:  # covers 20-30N and 0-10N; 10-20N and 30-40N tiles absent from server
        for left_lon in [60, 70, 80, 90]:
            row = (90 - top_lat) * 2000
            col = (left_lon + 180) * 4000
            filename = f'occurrence-{row:010d}-{col:010d}.tif'
            tiles.append((left_lon, top_lat - 10, filename))
    return tiles


# ── Checkpoint helpers ──────────────────────────────────────────────────────

def load_checkpoint(name):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f'{name}.json'
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()

def save_checkpoint(name, done_set):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f'{name}.json'
    with open(path, 'w') as f:
        json.dump(list(done_set), f)


# ── Hex grid helpers ────────────────────────────────────────────────────────

def hex_to_polygon(h3_index):
    boundary = h3.cell_to_boundary(h3_index)
    return Polygon([(lng, lat) for lat, lng in boundary])


def load_hex_grid():
    """Load H3 grid as GeoDataFrame, using a cached GeoPackage for fast restarts."""
    cache_path = PROJECT_ROOT / 'data' / 'raw' / 'hex_grid_cache.gpkg'

    if cache_path.exists():
        print(f"Loading cached hex GeoDataFrame from {cache_path}...")
        hex_gdf = gpd.read_file(cache_path)
        print(f"  {len(hex_gdf):,} hexagons loaded from cache")
        return hex_gdf

    grid_path = PROJECT_ROOT / 'data' / 'processed' / 'india_h3_grid_res7.csv'
    print(f"Loading H3 grid from {grid_path}...")
    grid = pd.read_csv(grid_path, encoding='utf-8-sig')
    grid.columns = grid.columns.str.strip()
    print(f"  {len(grid):,} hexagons loaded. Converting to polygons (2-3 min)...")

    grid['geometry'] = grid['h3_index'].apply(hex_to_polygon)
    hex_gdf = gpd.GeoDataFrame(grid, geometry='geometry', crs='EPSG:4326')
    hex_gdf = hex_gdf.reset_index(drop=True)
    hex_gdf['hex_id'] = range(len(hex_gdf))

    print(f"  Saving cache to {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hex_gdf.to_file(cache_path, driver='GPKG')
    print("  Cache saved.")
    return hex_gdf


# ── Raster extraction helpers ───────────────────────────────────────────────

def extract_mean_values(hex_gdf, raster_path, n_total, nodata_val=None, scale=1.0,
                        valid_min=None, valid_max=None):
    """Extract per-hexagon mean raster values via bulk rasterize + bincount.

    Args:
        hex_gdf: GeoDataFrame (may be a subset of the full grid)
        raster_path: Path to GeoTIFF
        n_total: Total number of hexagons in the full grid (sets bincount minlength)
    Returns:
        Array of length n_total with mean values; NaN where no pixels covered.
    """
    with rasterio.open(raster_path) as src:
        if str(hex_gdf.crs) != str(src.crs):
            hex_gdf = hex_gdf.to_crs(src.crs)

        shapes = [(geom, hid) for geom, hid in zip(hex_gdf.geometry, hex_gdf['hex_id'])]
        hex_raster = features.rasterize(
            shapes=shapes, out_shape=src.shape, transform=src.transform,
            fill=-1, all_touched=True, dtype=np.int32
        )
        data = src.read(1).astype(np.float32)

    valid = hex_raster >= 0
    if nodata_val is not None:
        valid &= data != nodata_val
    if valid_min is not None:
        valid &= data >= valid_min
    if valid_max is not None:
        valid &= data <= valid_max

    data = data * scale
    ids  = hex_raster[valid].astype(np.int32)
    vals = data[valid]

    sum_v   = np.bincount(ids, weights=vals,               minlength=n_total)
    count_v = np.bincount(ids, weights=np.ones_like(vals), minlength=n_total)

    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count_v > 0, sum_v / count_v, np.nan)


def extract_class_fractions(hex_subset, raster_path, class_values, resolution=0.01,
                            nodata_val=None):
    """
    Extract categorical land-cover fractions per hexagon from a downsampled tile.

    WorldCover tiles are 36,000x36,000 pixels (1.3 billion pixels, >10GB per
    array). At H3 resolution 7 (~2.3km), 10m detail cannot be distinguished.
    We read each tile through a WarpedVRT at 0.01 degrees (~1km), taking the
    mode of source classes in each output cell. This reduces a tile to 300x300
    pixels and keeps calculations fast and memory-safe.

    Returns a DataFrame indexed by the original hex_id values, with one fraction
    column per requested class group.
    """
    with rasterio.open(raster_path) as src:
        if str(hex_subset.crs) != str(src.crs):
            hex_subset = hex_subset.to_crs(src.crs)

        bounds = src.bounds
        width = int(np.ceil((bounds.right - bounds.left) / resolution))
        height = int(np.ceil((bounds.top - bounds.bottom) / resolution))
        transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                                width, height)

        with WarpedVRT(
            src,
            crs='EPSG:4326', transform=transform, width=width, height=height,
            resampling=Resampling.mode,
            src_nodata=nodata_val, nodata=nodata_val,
        ) as vrt:
            shapes = [(geom, hid) for geom, hid
                      in zip(hex_subset.geometry, hex_subset['hex_id'])]
            hex_raster = features.rasterize(
                shapes=shapes, out_shape=(height, width), transform=transform,
                fill=-1, all_touched=True, dtype=np.int32
            )
            data = vrt.read(1).astype(np.int16)

    valid = hex_raster >= 0
    if nodata_val is not None:
        valid &= data != nodata_val

    ids = hex_raster[valid].astype(np.int32)
    vals = data[valid]
    local_ids = hex_subset['hex_id'].values
    id_to_pos = {hid: pos for pos, hid in enumerate(local_ids)}
    positions = np.fromiter((id_to_pos[hid] for hid in ids), dtype=np.int32,
                            count=len(ids))

    total = np.bincount(positions, minlength=len(hex_subset))
    result = {}
    for col_name, classes in class_values.items():
        mask = np.isin(vals, classes)
        count = np.bincount(positions[mask], minlength=len(hex_subset))
        with np.errstate(invalid='ignore', divide='ignore'):
            result[col_name] = np.where(total > 0, count / total, np.nan)

    return pd.DataFrame(result, index=local_ids)


# ── MODIS NDVI via AppEEARS annual GeoTIFF ──────────────────────────────────
# AppEEARS lets you request a spatial subset as a GeoTIFF (one file per year,
# ~50-100MB instead of 23 HDF files per tile per year). We submit a task,
# poll until ready, then download.

APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

def appeears_token(username, password):
    """Obtain a short-lived AppEEARS bearer token."""
    r = requests.post(
        f"{APPEEARS_BASE}/login",
        auth=(username, password),
        timeout=30
    )
    r.raise_for_status()
    return r.json()['token']


def submit_ndvi_task(token, year):
    """Submit an AppEEARS point-sample task for annual mean NDVI over India."""
    headers = {'Authorization': f'Bearer {token}'}
    task = {
        "task_type": "area",
        "task_name": f"VectorHotspot_NDVI_{year}",
        "params": {
            "dates": [{"startDate": f"01-01-{year}", "endDate": f"12-31-{year}"}],
            "layers": [{"product": "MOD13A2.061", "layer": "_1_km_16_days_NDVI"}],
            "output": {"format": {"type": "geotiff"}, "projection": "geographic"},
            "geo": {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[67.0, 6.0], [98.0, 6.0],
                                          [98.0, 38.0], [67.0, 38.0],
                                          [67.0, 6.0]]]
                    },
                    "properties": {}
                }]
            }
        }
    }
    r = requests.post(f"{APPEEARS_BASE}/task", json=task, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()['task_id']


def poll_task(token, task_id, timeout_hours=3):
    """Poll AppEEARS until task completes. Returns True if done."""
    headers = {'Authorization': f'Bearer {token}'}
    deadline = time.time() + timeout_hours * 3600

    while time.time() < deadline:
        r = requests.get(f"{APPEEARS_BASE}/task/{task_id}", headers=headers, timeout=30)
        r.raise_for_status()
        status = r.json().get('status', '')
        print(f"  Task {task_id}: {status}")

        if status == 'done':
            return True
        elif status in ('error', 'deleted'):
            return False

        time.sleep(60)  # poll every minute

    print(f"  Task {task_id} timed out after {timeout_hours}h")
    return False


def download_task_files(token, task_id, dest_dir):
    """Download all GeoTIFF files from a completed AppEEARS task."""
    headers = {'Authorization': f'Bearer {token}'}
    r = requests.get(f"{APPEEARS_BASE}/bundle/{task_id}", headers=headers, timeout=30)
    r.raise_for_status()

    files = r.json().get('files', [])
    tif_files = [f for f in files if f['file_name'].endswith('.tif')]
    print(f"  {len(tif_files)} GeoTIFF files to download")

    downloaded = []
    for finfo in tqdm(tif_files, desc="  Downloading NDVI GeoTIFFs"):
        file_id = finfo['file_id']
        fname   = Path(finfo['file_name']).name
        dest    = dest_dir / fname

        if dest.exists():
            downloaded.append(dest)
            continue

        dl_r = requests.get(
            f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
            headers=headers, stream=True, timeout=120
        )
        dl_r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in dl_r.iter_content(65536):
                if chunk:
                    f.write(chunk)
        downloaded.append(dest)

    return downloaded


def _read_netrc_credentials():
    """Read NASA Earthdata credentials directly from the _netrc file."""
    import netrc as netrc_module
    netrc_path = Path.home() / '_netrc'
    if not netrc_path.exists():
        netrc_path = Path.home() / '.netrc'
    if not netrc_path.exists():
        raise RuntimeError("No _netrc or .netrc file found in home directory.")

    n = netrc_module.netrc(str(netrc_path))
    host = 'urs.earthdata.nasa.gov'
    creds = n.authenticators(host)
    if not creds:
        raise RuntimeError(f"No credentials found for {host} in {netrc_path}")
    username, _, password = creds
    return username, password


def download_modis_ndvi(hex_gdf):
    """
    Download annual mean NDVI for India via NASA AppEEARS (one GeoTIFF per year,
    ~50-100MB total instead of 15-25GB of HDF files). Checkpointed per year.
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError("pip install earthaccess")

    print("\n=== MODIS NDVI (via AppEEARS annual GeoTIFF) ===")
    RAW_NDVI_DIR.mkdir(parents=True, exist_ok=True)

    earthaccess.login(strategy="netrc")
    username, password = _read_netrc_credentials()
    print(f"  Authenticated as: {username}")

    ndvi_sum   = np.zeros(len(hex_gdf), dtype=np.float64)
    ndvi_count = np.zeros(len(hex_gdf), dtype=np.int32)

    # Load pre-existing partial NDVI accumulation if it exists
    partial_path = RAW_NDVI_DIR / 'ndvi_partial.npz'
    done_years = load_checkpoint('ndvi_years')

    if partial_path.exists() and done_years:
        saved = np.load(partial_path)
        ndvi_sum   = saved['ndvi_sum']
        ndvi_count = saved['ndvi_count']
        print(f"  Resuming from checkpoint: {done_years} already done")

    token = appeears_token(username, password)
    years_to_process = [y for y in [2018, 2019, 2020] if str(y) not in done_years]

    for year in years_to_process:
        print(f"\nSubmitting AppEEARS task for year {year}...")
        year_dir = RAW_NDVI_DIR / str(year)
        year_dir.mkdir(exist_ok=True)

        # Check if already downloaded
        existing_tifs = list(year_dir.glob('*.tif'))
        if existing_tifs:
            print(f"  Found {len(existing_tifs)} existing GeoTIFFs for {year}")
            tif_paths = existing_tifs
        else:
            task_id = submit_ndvi_task(token, year)
            print(f"  Task submitted: {task_id}")
            print(f"  Polling for completion (checks every 60s)...")
            ok = poll_task(token, task_id)
            if not ok:
                warnings.warn(f"AppEEARS task failed for {year}, skipping")
                continue
            tif_paths = download_task_files(token, task_id, year_dir)

        # Extract per-hexagon NDVI mean from each granule GeoTIFF
        print(f"  Extracting hex NDVI from {len(tif_paths)} files...")
        for tif_path in tqdm(tif_paths, desc=f"  Year {year}"):
            try:
                # Values are raw int16 (-2000..10000); apply scale AFTER
                # validating raw values. The prior version compared scaled
                # values to raw thresholds and discarded nearly every pixel.
                hex_vals = extract_mean_values(
                    hex_gdf, tif_path, n_total=len(hex_gdf),
                    nodata_val=-3000, scale=1.0 / 10000.0,
                    valid_min=-2000, valid_max=10000
                )
                valid = ~np.isnan(hex_vals)
                ndvi_sum[valid]   += hex_vals[valid]
                ndvi_count[valid] += 1
            except Exception as e:
                warnings.warn(f"Failed to process {tif_path}: {e}")

        # Checkpoint after each year
        done_years.add(str(year))
        save_checkpoint('ndvi_years', done_years)
        np.savez(partial_path, ndvi_sum=ndvi_sum, ndvi_count=ndvi_count)
        print(f"  Year {year} done and checkpointed.")

    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi_mean = np.where(ndvi_count > 0, ndvi_sum / ndvi_count, np.nan)

    nan_mask = np.isnan(ndvi_mean)
    if nan_mask.any():
        median_val = float(np.nanmedian(ndvi_mean))
        ndvi_mean[nan_mask] = median_val
        print(f"  Filled {nan_mask.sum():,} hexagons with spatial median ({median_val:.4f})")

    print(f"  NDVI done. Mean={np.nanmean(ndvi_mean):.4f}, "
          f"Range=[{np.nanmin(ndvi_mean):.4f}, {np.nanmax(ndvi_mean):.4f}]")
    return ndvi_mean


# ── ESA WorldCover 2021 ─────────────────────────────────────────────────────

WORLDCOVER_CLASSES = {
    'frac_water': [80, 90, 95],
    'frac_trees': [10],
    'frac_built': [50],
    'frac_shrub': [20],
}

def _worldcover_tile_url(lon, lat):
    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    filename = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
    return f"https://esa-worldcover.s3.amazonaws.com/v200/2021/map/{filename}", filename


def _download_single_tile(url, dest_path):
    """Download one tile; return (dest_path, ok, error_msg)."""
    if dest_path.exists():
        return dest_path, True, None
    tmp = dest_path.with_suffix('.tmp')
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 404:
            return dest_path, False, "404"
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest_path)
        return dest_path, True, None
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return dest_path, False, str(e)


def process_worldcover(hex_gdf):
    """Download ESA WorldCover tiles in parallel, extract land cover fractions."""
    print("\n=== ESA WorldCover 2021 ===")
    RAW_WC_DIR.mkdir(parents=True, exist_ok=True)

    tiles = _worldcover_tiles()
    done_tiles = load_checkpoint('worldcover_tiles')

    result = {col: np.zeros(len(hex_gdf), dtype=np.float64) for col in WORLDCOVER_CLASSES}
    total_covered = np.zeros(len(hex_gdf), dtype=np.int32)

    # Load partial accumulation if exists
    partial_path = RAW_WC_DIR / 'wc_partial.npz'
    if partial_path.exists() and done_tiles:
        saved = np.load(partial_path)
        for col in WORLDCOVER_CLASSES:
            result[col] = saved[col]
        total_covered = saved['total_covered']
        print(f"  Resuming: {len(done_tiles)} tiles already processed")

    # Identify tiles not yet done
    pending = [(lon, lat) for lon, lat in tiles
               if f"{lon}_{lat}" not in done_tiles]
    print(f"  {len(pending)} tiles to process ({len(done_tiles)} already done)")

    # Parallel download (4 threads — throttled to be polite to S3)
    print("  Downloading tiles in parallel (4 threads)...")
    download_jobs = []
    for lon, lat in pending:
        url, fname = _worldcover_tile_url(lon, lat)
        download_jobs.append((url, RAW_WC_DIR / fname, lon, lat))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_download_single_tile, url, dest): (lon, lat)
                   for url, dest, lon, lat in download_jobs}

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Downloading WorldCover"):
            lon, lat = futures[future]
            dest_path, ok, err = future.result()
            if not ok and err != "404":
                warnings.warn(f"Download failed ({lon},{lat}): {err}")

    # Process tiles sequentially (rasterize is CPU-bound, not I/O-bound)
    print("  Extracting per-hexagon land cover fractions...")
    for lon, lat in tqdm(pending, desc="  Processing WorldCover tiles"):
        tile_key = f"{lon}_{lat}"
        _, fname = _worldcover_tile_url(lon, lat)
        tile_path = RAW_WC_DIR / fname

        if not tile_path.exists():
            continue  # leave uncheckpointed so a failed download retries next run

        try:
            with rasterio.open(tile_path) as src:
                tile_bounds = src.bounds
            tile_bbox = box(tile_bounds.left, tile_bounds.bottom,
                            tile_bounds.right, tile_bounds.top)
            intersecting = hex_gdf[hex_gdf.intersects(tile_bbox)]
            if len(intersecting) == 0:
                done_tiles.add(tile_key)
                continue

            fracs = extract_class_fractions(
                intersecting, tile_path,
                class_values=WORLDCOVER_CLASSES, nodata_val=0
            )
            hex_ids = intersecting['hex_id'].values
            for col in WORLDCOVER_CLASSES:
                vals = fracs.loc[hex_ids, col].to_numpy()
                result[col][hex_ids] += np.where(np.isnan(vals), 0.0, vals)
            total_covered[hex_ids] += 1

        except Exception as e:
            warnings.warn(f"Failed to process {tile_path}: {e}")
        finally:
            done_tiles.add(tile_key)
            gc.collect()

        # Checkpoint every 10 tiles
        if len(done_tiles) % 10 == 0:
            save_checkpoint('worldcover_tiles', done_tiles)
            save_data = {col: result[col] for col in WORLDCOVER_CLASSES}
            save_data['total_covered'] = total_covered
            np.savez(partial_path, **save_data)

    # Final checkpoint
    save_checkpoint('worldcover_tiles', done_tiles)

    # Normalize
    for col in WORLDCOVER_CLASSES:
        covered = total_covered > 0
        result[col][covered] /= total_covered[covered]
        result[col][~covered] = 0.0

    for col in WORLDCOVER_CLASSES:
        print(f"  {col}: mean={result[col].mean():.4f}, nonzero={(result[col] > 0).sum():,}")

    return result


# ── JRC Global Surface Water ─────────────────────────────────────────────────

JRC_TILE_BASE = (
    'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/'
    'GSWE/Aggregated/LATEST/occurrence/tiles/'
)


def _jrc_tile_url(filename):
    """Build the official JRC Global Surface Water Occurrence tile URL."""
    return f'{JRC_TILE_BASE}{filename}'


def extract_jrc_mean_values(hex_subset, tile_path, n_total):
    """
    Extract JRC occurrence means without loading a 40,000x40,000 (1.6GB) tile.

    The JRC files are 30m/40000x40000. A WarpedVRT downscales each tile to
    0.01 degrees (~1km), which is sufficient for res-7 (~2.3km) H3 cells and
    keeps RAM bounded. JRC occurrence is a long-term, structural covariate;
    sub-30m detail is not identifiable at the H3 resolution anyway.
    """
    with rasterio.open(tile_path) as src:
        bounds = src.bounds
        resolution = 0.01  # degrees, approximately 1km
        width = int(np.ceil((bounds.right - bounds.left) / resolution))
        height = int(np.ceil((bounds.top - bounds.bottom) / resolution))
        transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                                width, height)

        with WarpedVRT(
            src,
            crs='EPSG:4326', transform=transform,
            width=width, height=height,
            resampling=Resampling.average,
            src_nodata=255, nodata=255,
        ) as vrt:
            shapes = [(geom, hid) for geom, hid in zip(hex_subset.geometry,
                                                        hex_subset['hex_id'])]
            hex_raster = features.rasterize(
                shapes=shapes, out_shape=(height, width), transform=transform,
                fill=-1, all_touched=True, dtype=np.int32
            )
            data = vrt.read(1).astype(np.float32)

    valid = (hex_raster >= 0) & (data != 255) & (data >= 0) & (data <= 100)
    ids = hex_raster[valid].astype(np.int32)
    vals = data[valid] / 100.0
    sums = np.bincount(ids, weights=vals, minlength=n_total)
    counts = np.bincount(ids, weights=np.ones_like(vals), minlength=n_total)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(counts > 0, sums / counts, np.nan)


def process_jrc_water(hex_gdf):
    """Download official JRC occurrence tiles and extract res-7 hexagon means."""
    print("\n=== JRC Global Surface Water Occurrence ===")
    RAW_JRC_DIR.mkdir(parents=True, exist_ok=True)

    tiles = _jrc_tiles()
    done_tiles = load_checkpoint('jrc_tiles')
    jrc_sum = np.zeros(len(hex_gdf), dtype=np.float64)
    jrc_count = np.zeros(len(hex_gdf), dtype=np.int32)

    partial_path = RAW_JRC_DIR / 'jrc_partial.npz'
    if partial_path.exists() and done_tiles:
        saved = np.load(partial_path)
        jrc_sum, jrc_count = saved['jrc_sum'], saved['jrc_count']
        print(f"  Resuming: {len(done_tiles)} tiles already processed")

    pending = [(lon, lat, filename) for lon, lat, filename in tiles
               if filename not in done_tiles]
    print(f"  {len(pending)} official JRC tiles to process")

    # Download required tiles in parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_download_single_tile, _jrc_tile_url(filename),
                        RAW_JRC_DIR / filename): (lon, lat, filename)
            for lon, lat, filename in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='  Downloading JRC tiles'):
            lon, lat, filename = futures[future]
            _, ok, err = future.result()
            if not ok:
                warnings.warn(f'JRC tile {filename} failed: {err}')

    # Process sequentially — each tile is downsampled before read
    for lon, lat, filename in tqdm(pending, desc='  Processing JRC tiles'):
        tile_path = RAW_JRC_DIR / filename
        if not tile_path.exists():
            continue  # leave uncheckpointed to retry on next run

        try:
            with rasterio.open(tile_path) as src:
                b = src.bounds
            tile_bbox = box(b.left, b.bottom, b.right, b.top)
            intersecting = hex_gdf[hex_gdf.intersects(tile_bbox)]
            if len(intersecting) == 0:
                done_tiles.add(filename)
                continue

            hex_vals = extract_jrc_mean_values(intersecting, tile_path, len(hex_gdf))
            hex_ids = intersecting['hex_id'].values
            valid = ~np.isnan(hex_vals[hex_ids])
            selected_ids = hex_ids[valid]
            jrc_sum[selected_ids] += hex_vals[selected_ids]
            jrc_count[selected_ids] += 1
            done_tiles.add(filename)

        except Exception as e:
            warnings.warn(f'Failed to process {tile_path}: {e}')
        finally:
            gc.collect()

        if len(done_tiles) % 3 == 0:
            save_checkpoint('jrc_tiles', done_tiles)
            np.savez(partial_path, jrc_sum=jrc_sum, jrc_count=jrc_count)

    save_checkpoint('jrc_tiles', done_tiles)
    np.savez(partial_path, jrc_sum=jrc_sum, jrc_count=jrc_count)

    with np.errstate(invalid='ignore', divide='ignore'):
        jrc_occ = np.where(jrc_count > 0, jrc_sum / jrc_count, 0.0)

    print(f'  JRC done. Mean={jrc_occ.mean():.4f}, nonzero={(jrc_occ > 0).sum():,}')
    return jrc_occ


# ── Validation ───────────────────────────────────────────────────────────────

def validate_output(df):
    print("\n=== VALIDATION ===")
    checks = {
        'ndvi_mean':     (-0.2, 1.0),
        'frac_water':    ( 0.0, 1.0),
        'frac_trees':    ( 0.0, 1.0),
        'frac_built':    ( 0.0, 1.0),
        'frac_shrub':    ( 0.0, 1.0),
        'jrc_occurrence':( 0.0, 1.0),
    }
    for col, (lo, hi) in checks.items():
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        nulls = df[col].isna().sum()
        tag = "OK" if bad == 0 and nulls == 0 else "WARN"
        print(f"  [{tag}] {col}: [{df[col].min():.4f}, {df[col].max():.4f}], "
              f"nulls={nulls}, out_of_range={bad}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VectorHotspot Phase 6: Fetch Tier 2 Covariates (Optimized)")
    print("=" * 70)

    output_path = OUTPUT_DIR / 'hex_tier2_covariates.csv'
    if output_path.exists():
        print(f"Output already exists: {output_path}")
        print("Delete it to rerun.")
        return

    hex_gdf = load_hex_grid()

    ndvi_mean = download_modis_ndvi(hex_gdf)
    wc_fracs  = process_worldcover(hex_gdf)
    jrc_occ   = process_jrc_water(hex_gdf)

    print("\nAssembling output CSV...")
    result = pd.DataFrame({
        'h3_index':       hex_gdf['h3_index'].values,
        'ndvi_mean':      ndvi_mean.astype(np.float32),
        'frac_water':     np.nan_to_num(wc_fracs['frac_water'], nan=0.0).astype(np.float32),
        'frac_trees':     np.nan_to_num(wc_fracs['frac_trees'], nan=0.0).astype(np.float32),
        'frac_built':     np.nan_to_num(wc_fracs['frac_built'], nan=0.0).astype(np.float32),
        'frac_shrub':     np.nan_to_num(wc_fracs['frac_shrub'], nan=0.0).astype(np.float32),
        'jrc_occurrence': jrc_occ.astype(np.float32),
    })

    validate_output(result)
    result.to_csv(output_path, index=False, encoding='utf-8')
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB), {len(result):,} rows")
    print("\n" + "=" * 70)
    print("Phase 6 Tier 2 covariate extraction complete!")
    print("Next: py src/disaggregation/wire_disaggregation_model.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
