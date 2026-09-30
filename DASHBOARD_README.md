# VectorHotspot Dashboard - Quick Start

## Phase 13: Operational Early Warning Engine & Interactive Dashboard

### Components

**Task 1: Alert Generation Engine** ✅ COMPLETE
- Generates 3-tier alerts (Watch/Warning/High Alert) per hexagon
- Based on Gi* z-scores and predicted case growth
- Outputs: `outputs/alerts/alerts_{disease}_latest.json`

**Task 2: Interactive Dashboard** 
- Streamlit web application with Plotly visualizations
- Alert distribution, state-level summaries, data export
- No map API keys required (uses Plotly scatter)

### Installation

```bash
# Install dependencies (if not already installed)
pip install streamlit plotly h3

# For the advanced PyDeck version (optional):
pip install pydeck
```

### Running the Dashboard

**Simple Version (Recommended - no APIs required):**
```bash
streamlit run src/dashboard/app_simple.py
```

**Advanced Version (with H3 hexagon map):**
```bash
streamlit run src/dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

### Features

#### Tabs in Simple Dashboard:

1. **📊 Overview**
   - Alert level distribution (pie chart)
   - Case growth histogram
   - Summary metrics

2. **🗺️ By State**
   - State-level alert breakdown
   - Priority states (with alerts)
   - Full state listing

3. **📈 Top Alerts**
   - Top 20 highest-risk hexagons
   - Sorted by alert level and Gi* z-score
   - Includes taxonomy (Persistent/Emerging/etc)

4. **📋 Data Table**
   - Full alert data with filters
   - State and alert level selection
   - CSV export button

### Alert Levels Explained

- **🔴 Level 3 (Red/High Alert)**: Persistent or Intensifying hotspots with high significance (Gi* ≥ 2.58, p < 0.01)
- **🟠 Level 2 (Orange/Warning)**: Emerging hotspots with statistical significance (Gi* ≥ 1.96, p < 0.05)
- **🟡 Level 1 (Yellow/Watch)**: Moderate growth (>20%) or weak significance (Gi* > 1.65)
- **⚪ Level 0 (Gray/Normal)**: No alert

### Data Regeneration

To regenerate alerts after model updates:

```bash
python src/dashboard/generate_alerts.py
```

This will:
- Load latest predictions (Week 52, 2024 in test set)
- Compute case growth and assign alert levels
- Export JSON and CSV files to `outputs/alerts/`

### Architecture

```
outputs/alerts/
├── alerts_dengue_latest.json        (35,818 hexagons × fields)
├── alerts_malaria_latest.json       (35,818 hexagons × fields)
├── alerts_dengue_by_state.csv       (state-level summary)
├── alerts_malaria_by_state.csv
└── alert_summary.json               (top-level statistics)

src/dashboard/
├── generate_alerts.py               (alert engine)
├── app_simple.py                    (Streamlit dashboard - Plotly)
└── app.py                          (Advanced dashboard - PyDeck)
```

### Example Alert Data Structure

```json
{
  "h3_index": "873c8ca44ffffff",
  "state": "West Bengal",
  "district": "Kolkata",
  "year": 2024,
  "week": 52,
  "cases_lag_1": 45.23,
  "pred_lead_1": 67.89,
  "case_growth_pct": 50.1,
  "gi_zscore_pred_lead_1": 2.34,
  "is_hotspot_pred_lead_1": 1,
  "hotspot_taxonomy": "Intensifying",
  "alert_level": 2
}
```

### Performance Notes

- Dashboard loads ~36k hexagons per disease
- JSON files are ~13-14 MB each (pre-calculated, not real-time)
- Streamlit caches data with `@st.cache_data`
- Filtering/sorting is instant

### Next Steps

For production deployment:
1. Set up backend API to compute alerts weekly
2. Add authentication (if needed)
3. Deploy to Streamlit Cloud or Docker
4. Add email/SMS alert notifications
5. Integrate with WhatsApp/Telegram for public alerts

### Troubleshooting

**"Alert data not found"**
- Run: `python src/dashboard/generate_alerts.py`

**Dashboard runs slow**
- Try restarting: `streamlit cache clear`
- Reduce data range in filters

**Map not rendering (PyDeck version)**
- Ensure h3 library is installed: `pip install h3`
- Check that alert JSONs have valid geometry

---

**Phase 13 Complete.** The VectorHotspot system now provides:
- ✅ Dual-disease forecasting (Phase 9)
- ✅ Hotspot detection (Phase 10)
- ✅ Rigorous evaluation (Phase 11)
- ✅ Explainability analysis (Phase 12)
- ✅ **Operational alerts & dashboard (Phase 13)**
