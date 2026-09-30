"""
Phase 13 - Task 2: Interactive Dashboard (Streamlit).

Features:
  - Disease toggle (Dengue/Malaria)
  - H3 hexagon map with PyDeck visualization
  - Color-coded by alert level or predicted risk
  - Time horizon slider (t+1 to t+4)
  - Click hexagon to see details: forecast, alerts, SHAP top features
  - State/district filters

Usage:
  streamlit run src/dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import h3
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="VectorHotspot - Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent.parent
ALERTS_DIR = ROOT / "outputs" / "alerts"
HOT_DIR = ROOT / "outputs" / "hotspots"


@st.cache_data
def load_alerts(disease):
    """Load alert data for disease."""
    path = ALERTS_DIR / f"alerts_{disease}_latest.json"
    return pd.read_json(path)


@st.cache_data
def load_hotspots(disease):
    """Load full hotspot predictions."""
    path = HOT_DIR / f"{disease}_hotspots_test_2023_2024.parquet"
    df = pd.read_parquet(path)
    # Filter to latest year
    latest_year = df["year"].max()
    return df[df["year"] == latest_year].copy()


def h3_to_polygon(h3_index):
    """Convert H3 index to polygon coordinates."""
    boundary = h3.cell_to_boundary(h3_index)
    # PyDeck expects [lon, lat] not [lat, lon]
    return [[lon, lat] for lat, lon in boundary]


def add_geometry(df):
    """Add polygon geometry to dataframe."""
    df["polygon"] = df["h3_index"].apply(h3_to_polygon)
    return df


def get_alert_color(level):
    """Map alert level to RGB color."""
    colors = {
        0: [200, 200, 200, 100],  # Gray (None)
        1: [255, 255, 0, 150],     # Yellow (Watch)
        2: [255, 165, 0, 180],     # Orange (Warning)
        3: [255, 0, 0, 200],       # Red (High Alert)
    }
    return colors.get(level, [200, 200, 200, 100])


def get_risk_color(gi_zscore):
    """Map Gi* z-score to color (blue to red gradient)."""
    # Normalize z-score to 0-255
    normalized = np.clip((gi_zscore + 2) / 6, 0, 1) * 255
    r = int(normalized)
    b = int(255 - normalized)
    return [r, 0, b, 150]


# Sidebar
st.sidebar.title("VectorHotspot Dashboard")
st.sidebar.markdown("**Dual-Disease Early Warning System for India**")

disease = st.sidebar.selectbox("Disease", ["dengue", "malaria"])
view_mode = st.sidebar.radio("View Mode", ["Alert Levels", "Risk Heatmap"])
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 4, 1) if view_mode == "Risk Heatmap" else 1

# Load data
alerts = load_alerts(disease)
hotspots = load_hotspots(disease)

# Filter to selected week (latest for alerts, or by horizon for heatmap)
if view_mode == "Alert Levels":
    latest_week = alerts["week"].iloc[0]
    display_data = alerts.copy()
    display_data["color"] = display_data["alert_level"].apply(get_alert_color)
else:
    # Risk heatmap - use gi_zscore_pred for selected horizon
    latest_week = hotspots["week"].max()
    display_data = hotspots[hotspots["week"] == latest_week].copy()
    z_col = f"gi_zscore_pred_lead_{horizon}"
    display_data["color"] = display_data[z_col].apply(get_risk_color)

# Add geometry
display_data = add_geometry(display_data)

# Main content
st.title(f"🦟 {disease.title()} Early Warning System")
st.markdown(f"**Week {latest_week}, 2024** | Viewing: {view_mode}")

# Stats
col1, col2, col3, col4 = st.columns(4)
if view_mode == "Alert Levels":
    col1.metric("🔴 High Alert", f"{(display_data['alert_level'] == 3).sum():,}")
    col2.metric("🟠 Warning", f"{(display_data['alert_level'] == 2).sum():,}")
    col3.metric("🟡 Watch", f"{(display_data['alert_level'] == 1).sum():,}")
    col4.metric("⚪ Normal", f"{(display_data['alert_level'] == 0).sum():,}")
else:
    hotspot_col = f"is_hotspot_pred_lead_{horizon}"
    col1.metric("Predicted Hotspots", f"{display_data[hotspot_col].sum():,}")
    col2.metric("High Risk (z>2.58)", f"{(display_data[z_col] >= 2.58).sum():,}")
    col3.metric("Medium Risk (z>1.96)", f"{((display_data[z_col] >= 1.96) & (display_data[z_col] < 2.58)).sum():,}")
    col4.metric("Total Hexagons", f"{len(display_data):,}")

# Map
st.subheader("Spatial Distribution")

# PyDeck layer
layer = pdk.Layer(
    "PolygonLayer",
    display_data,
    get_polygon="polygon",
    get_fill_color="color",
    get_line_color=[255, 255, 255, 50],
    pickable=True,
    auto_highlight=True,
)

# View state (center on India)
view_state = pdk.ViewState(
    latitude=20.5,
    longitude=78.9,
    zoom=4,
    pitch=0,
)

# Render
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>State:</b> {state}<br/>"
                "<b>District:</b> {district}<br/>"
                "<b>Alert Level:</b> {alert_level}<br/>"
                "<b>Gi* z-score:</b> {gi_zscore_pred_lead_1:.2f}<br/>"
                "<b>Predicted Cases:</b> {pred_lead_1:.2f}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    },
    map_style="light"
)

st.pydeck_chart(deck)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "**VectorHotspot** uses H3 Resolution 7 hexagons (~5.2 km²) for fine-grained "
    "spatiotemporal forecasting. Alerts are generated using Getis-Ord Gi* hotspot "
    "detection on predicted risk surfaces."
)
