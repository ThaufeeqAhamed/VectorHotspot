"""
VectorHotspot Dashboard - Simplified Version with Plotly.

A lightweight interactive dashboard using Plotly for visualization.
Easier to run than PyDeck version, no API keys required.

Usage:
  streamlit run src/dashboard/app_simple.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="VectorHotspot Dashboard",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent.parent
ALERTS_DIR = ROOT / "outputs" / "alerts"


@st.cache_data
def load_alerts(disease):
    """Load alert data."""
    path = ALERTS_DIR / f"alerts_{disease}_latest.json"
    return pd.read_json(path)


# Sidebar
st.sidebar.title("🦟 VectorHotspot")
st.sidebar.markdown("**Dual-Disease Early Warning System**")

disease = st.sidebar.selectbox("Select Disease", ["dengue", "malaria"], format_func=str.title)

# Load data
try:
    alerts = load_alerts(disease)
except FileNotFoundError:
    st.error(f"Alert data not found. Please run: `python src/dashboard/generate_alerts.py`")
    st.stop()

# Main content
st.title(f"{disease.title()} Early Warning Dashboard")
st.markdown(f"**Week {alerts['week'].iloc[0]}, {alerts['year'].iloc[0]}** | {len(alerts):,} hexagons analyzed")

# Alert level distribution
col1, col2, col3, col4 = st.columns(4)

level_counts = alerts["alert_level"].value_counts()
col1.metric("🔴 High Alert (Level 3)", f"{level_counts.get(3, 0):,}")
col2.metric("🟠 Warning (Level 2)", f"{level_counts.get(2, 0):,}")
col3.metric("🟡 Watch (Level 1)", f"{level_counts.get(1, 0):,}")
col4.metric("⚪ Normal (Level 0)", f"{level_counts.get(0, 0):,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ By State", "📈 Top Alerts", "📋 Data Table"])

with tab1:
    st.subheader("Alert Distribution")

    # Alert level breakdown
    alert_labels = {0: "Normal", 1: "Watch", 2: "Warning", 3: "High Alert"}
    alert_colors = {0: "#cccccc", 1: "#ffff00", 2: "#ffa500", 3: "#ff0000"}

    alert_summary = alerts["alert_level"].value_counts().sort_index()
    alert_df = pd.DataFrame({
        "Level": [alert_labels[i] for i in alert_summary.index],
        "Count": alert_summary.values
    })

    fig = px.bar(
        alert_df,
        x="Level",
        y="Count",
        color="Level",
        color_discrete_map={
            "Normal": alert_colors[0],
            "Watch": alert_colors[1],
            "Warning": alert_colors[2],
            "High Alert": alert_colors[3]
        },
        title=f"{disease.title()} Alert Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Case growth distribution
    st.subheader("Predicted Case Growth Distribution")
    fig2 = px.histogram(
        alerts[alerts["case_growth_pct"] < 200],  # Cap at 200% for visibility
        x="case_growth_pct",
        nbins=50,
        title="Case Growth % (t to t+1)"
    )
    fig2.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="20% threshold")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Alerts by State")

    # State-level summary
    state_alerts = alerts.groupby(["state", "alert_level"]).size().unstack(fill_value=0)
    state_alerts.columns = [alert_labels.get(int(c), str(c)) for c in state_alerts.columns]
    state_alerts = state_alerts.reset_index()

    # Show only states with alerts
    if "High Alert" in state_alerts.columns or "Warning" in state_alerts.columns:
        priority_states = state_alerts[
            (state_alerts.get("High Alert", 0) > 0) |
            (state_alerts.get("Warning", 0) > 0)
        ].sort_values(by=["High Alert", "Warning"], ascending=False)

        if len(priority_states) > 0:
            st.markdown("**🔴 States with High Alert or Warning:**")
            st.dataframe(priority_states, use_container_width=True)
        else:
            st.info("No states currently have High Alert or Warning level.")

    # All states summary
    st.markdown("**All States:**")
    st.dataframe(state_alerts.sort_values("state"), use_container_width=True)

with tab3:
    st.subheader("Top 20 Alert Hexagons")

    # Filter to alerts only (level > 0)
    alert_hexes = alerts[alerts["alert_level"] > 0].copy()
    alert_hexes = alert_hexes.sort_values(
        ["alert_level", "gi_zscore_pred_lead_1"],
        ascending=[False, False]
    ).head(20)

    if len(alert_hexes) > 0:
        display_cols = ["state", "district", "alert_level", "gi_zscore_pred_lead_1",
                       "pred_lead_1", "case_growth_pct", "hotspot_taxonomy"]
        alert_hexes_display = alert_hexes[display_cols].copy()
        alert_hexes_display.columns = ["State", "District", "Alert Level", "Gi* Z-Score",
                                       "Predicted Cases", "Growth %", "Taxonomy"]
        st.dataframe(alert_hexes_display, use_container_width=True)
    else:
        st.info("No alerts generated for this week.")

with tab4:
    st.subheader("Full Alert Data")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_states = st.multiselect(
            "Filter by State",
            options=sorted(alerts["state"].unique()),
            default=None
        )
    with col2:
        min_alert_level = st.selectbox("Minimum Alert Level", [0, 1, 2, 3], index=0)

    # Apply filters
    filtered = alerts.copy()
    if selected_states:
        filtered = filtered[filtered["state"].isin(selected_states)]
    filtered = filtered[filtered["alert_level"] >= min_alert_level]

    st.markdown(f"**Showing {len(filtered):,} of {len(alerts):,} hexagons**")
    st.dataframe(
        filtered[["h3_index", "state", "district", "alert_level", "gi_zscore_pred_lead_1",
                 "pred_lead_1", "case_growth_pct", "hotspot_taxonomy"]],
        use_container_width=True
    )

    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"vectorhotspot_alerts_{disease}_filtered.csv",
        mime="text/csv"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "**About VectorHotspot**\n\n"
    f"- Resolution: H3 Level 7 (~5.2 km²)\n"
    f"- Hexagons: {len(alerts):,}\n"
    f"- Alert Levels: 4 (0-3)\n"
    f"- Forecast Horizon: 1-4 weeks\n\n"
    "Alerts are generated using Getis-Ord Gi* hotspot detection on predicted risk surfaces."
)
