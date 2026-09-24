# VectorHotspot Phase 5 Completion Guide

**Date:** 2026-09-24  
**Current Status:** Repository structure complete, awaiting WorldPop data download

---

## Overview

This guide walks you through completing Phase 5 (Spatial Disaggregation) of the VectorHotspot project. After following these steps, you'll have generated fine-resolution (H3 hexagon-level) dengue and malaria case estimates for all of India.

---

## Prerequisites

### 1. Python Environment

You need Python 3.8+ with the following packages:

```bash
pip install pandas numpy scipy geopandas shapely h3 rasterio requests tqdm
```

**To check if you have Python:**
```bash
python --version
# or
python3 --version
```

If Python is not installed, download from: https://www.python.org/downloads/

---

## Step-by-Step Instructions

### Step 1: Download WorldPop Population Data

**Option A: Automated Download (Recommended)**

```bash
cd W:\MiniProject5\VectorHotspot
python src/data_prep/download_worldpop.py
```

This will download 5 GeoTIFF files (~90 MB total) to `data/raw/population/`.

**Option B: Manual Download**

If the script fails, download manually:

1. Visit: https://hub.worldpop.org/geodata/listing?id=29
2. Download these 5 files (right-click → Save As):
   - [2000](https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2000/IND/ind_ppp_2000_1km_Aggregated.tif)
   - [2005](https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2005/IND/ind_ppp_2005_1km_Aggregated.tif)
   - [2010](https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2010/IND/ind_ppp_2010_1km_Aggregated.tif)
   - [2015](https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2015/IND/ind_ppp_2015_1km_Aggregated.tif)
   - [2020](https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif)

3. Save all files to: `W:\MiniProject5\VectorHotspot\data\raw\population\`

4. Verify filenames are exactly:
   - `ind_ppp_2000_1km_Aggregated.tif` (18.3 MB)
   - `ind_ppp_2005_1km_Aggregated.tif` (18.3 MB)
   - `ind_ppp_2010_1km_Aggregated.tif` (18.3 MB)
   - `ind_ppp_2015_1km_Aggregated.tif` (18.3 MB)
   - `ind_ppp_2020_1km_Aggregated.tif` (18.3 MB)

---

### Step 2: Generate Hexagon-Level Population

```bash
cd W:\MiniProject5\VectorHotspot
python src/data_prep/generate_hex_population.py
```

**What this does:**
- Reads the 620,742 hexagons from `india_h3_grid_res7.csv`
- Extracts population for each hexagon from the WorldPop rasters
- Uses fast bulk rasterization (takes ~5-10 minutes for all 5 years)
- Validates results against district-level totals

**Expected output:**
```
Processing year 2000...
  Rasterizing 620,742 hexagon polygons...
  Aggregating population via bincount...
  Year 2000 total population: 1,056,575,549
...
✓ SUCCESS: hex_population_2000_2020.csv saved
```

**Output file:** `data/processed/hex_population_2000_2020.csv` (~40-50 MB)

**If errors occur:**
- Check that all 5 `.tif` files are in `data/raw/population/`
- Check that filenames match exactly (case-sensitive on some systems)
- Ensure you have `rasterio` installed: `pip install rasterio`

---

### Step 3: Run Spatial Disaggregation Model

```bash
cd W:\MiniProject5\VectorHotspot
python src/disaggregation/wire_disaggregation_model.py
```

**What this does:**
- Fits Poisson regression models for dengue (state-level) and malaria (district-level)
- Generates hexagon-level predictions for both diseases
- Rescales predictions to exactly preserve known state/district totals
- Validates mass-preservation (should show 0.000000 error)

**Expected output:**
```
=== DENGUE DISAGGREGATION ===
Building modeling dataset...
Fitting Poisson regression over 451 aggregation groups...
Fitted coefficients: β₀ = -10.5241, β₁ = +0.8652
Mass preservation check:
  Max absolute error: 0.000000
✓ Dengue results saved

=== MALARIA DISAGGREGATION ===
...
✓ Malaria results saved

✓ Disaggregation pipeline completed successfully!
```

**Output files:**
- `data/processed/dengue_hex_annual.csv` (~409 MB, gitignored)
- `data/processed/malaria_hex_annual.csv` (~748 MB, gitignored)
- `data/processed/disaggregation_fit_report.txt` (text summary with exact coefficients)

**Expected run time:** 5-15 minutes depending on your machine

**Key validation checks:**
1. ✓ Mass preservation max error should be 0.000000 (exact)
2. ✓ Dengue coefficient (β₁) should be POSITIVE (~+0.8 to +0.9)
3. ✓ Malaria coefficient (β₁) should be NEGATIVE (~-0.3 to -0.5)
4. ✓ Total national case counts should match input data

---

### Step 4: Verify Results

Check that these files now exist:

```bash
dir data\processed\hex_population_2000_2020.csv
dir data\processed\dengue_hex_annual.csv
dir data\processed\malaria_hex_annual.csv
dir data\processed\disaggregation_fit_report.txt
```

Open `disaggregation_fit_report.txt` to review:
- Fitted model coefficients
- Mass preservation validation results
- Total case counts per disease

---

## Troubleshooting

### "Module not found" errors

Install the missing package:
```bash
pip install <package_name>
```

Common missing packages:
- `geopandas`: `pip install geopandas`
- `h3`: `pip install h3`
- `rasterio`: `pip install rasterio`

### "File not found" errors

Check paths are correct:
- Scripts expect to be run from `W:\MiniProject5\VectorHotspot\`
- All scripts use relative paths from the project root
- If running from a different directory, provide absolute paths

### Memory errors

The disaggregation model processes millions of hexagon-year rows:
- Close other applications to free RAM
- If still failing, the script includes memory cleanup (`gc.collect()`)
- Minimum recommended: 8 GB RAM

### BOM/encoding errors (Windows)

If you see `KeyError: 'h3_index'` or similar:
- The script already includes `encoding='utf-8-sig'` to handle this
- If still occurring, open CSVs in a text editor and check for strange characters at the start
- Re-save as UTF-8 without BOM

---

## What's Next (Phase 6)

After completing Phase 5, you need to decide the next direction:

### Option A: Tier 2 Covariates
Pull additional environmental covariates to strengthen the disaggregation model:
- NDVI (vegetation index) from MODIS
- Land cover from ESA WorldCover
- Water bodies from JRC Global Surface Water

### Option B: Temporal Disaggregation
Design a method to redistribute annual case totals across weeks using weekly weather data, enabling the weekly forecasting the original blueprint envisions.

**Recommendation from handover:** Start with temporal disaggregation/feature engineering (Option B), as it's on the critical path to forecasting. Tier 2 covariates can be added later to improve the spatial model.

---

## Committing to GitHub

Once all scripts run successfully:

```bash
cd W:\MiniProject5\VectorHotspot

# Stage the new files (but not the large hex_annual.csv files, which are gitignored)
git add src/data_prep/generate_hex_population.py
git add src/data_prep/download_worldpop.py
git add src/disaggregation/wire_disaggregation_model.py
git add .gitignore
git add README.md
git add verify_setup.py

# Commit Phase 5 completion
git commit -m "Complete Phase 5: Spatial disaggregation model

- Add wire_disaggregation_model.py (real disaggregation, population covariate)
- Add generate_hex_population.py (fast bulk rasterize method)
- Add download_worldpop.py (automated WorldPop download)
- Update .gitignore to exclude large hex_annual.csv files
- Add comprehensive README.md
- Add verify_setup.py helper script

Phase 5 validated:
- Mass preservation: exact (0.000000 max error)
- Dengue: positive population coefficient (urban concentration)
- Malaria: negative population coefficient (rural concentration)

Co-Authored-By: Claude Code <noreply@anthropic.com>"

# Push to GitHub
git push origin main
```

**Note:** The large output files (`*_hex_annual.csv`, ~400-750 MB each) are gitignored and won't be committed. They can be regenerated by running the script.

---

## Summary Checklist

- [ ] Python 3.8+ installed with required packages
- [ ] Downloaded 5 WorldPop raster files to `data/raw/population/`
- [ ] Ran `generate_hex_population.py` successfully
- [ ] Generated `hex_population_2000_2020.csv` exists (~40-50 MB)
- [ ] Ran `wire_disaggregation_model.py` successfully
- [ ] Generated `dengue_hex_annual.csv` and `malaria_hex_annual.csv` exist
- [ ] Reviewed `disaggregation_fit_report.txt` for validation results
- [ ] Committed Phase 5 scripts to GitHub
- [ ] Decided on Phase 6 direction (Tier 2 covariates vs temporal disaggregation)

---

**For detailed project history and methodology decisions, see [`PROJECT_HANDOVER.md`](PROJECT_HANDOVER.md).**

**Questions or issues?** Check the handover document's debugging history (Part 13) for solutions to common problems.

---

*Last Updated: 2026-09-24*
