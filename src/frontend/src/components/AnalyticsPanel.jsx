import React, { useState, useEffect, useRef, useMemo } from 'react';
import { X, Activity, BarChart2, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { cachedFetch } from '../apiCache';

// Human-friendly, one-word mapping for all 45 technical model features
const FEATURE_DICTIONARY = {
  // Population & Demographics
  'log_population':       { label: 'Population',   desc: 'Total human population count' },
  'pop_density':          { label: 'Density',       desc: 'Human population density per km²' },

  // Temperature & Heat
  'tmin_lag_1':           { label: 'Temperature',   desc: 'Minimum night temperature (last week)' },
  'tmin_lag_2':           { label: 'Temperature',   desc: 'Minimum night temperature (2 wks ago)' },
  'tmin_lag_4':           { label: 'Temperature',   desc: 'Minimum night temperature (4 wks ago)' },
  'tmax_lag_1':           { label: 'Heat',          desc: 'Maximum day temperature' },
  'tmax_lag_2':           { label: 'Heat',          desc: 'Maximum day temperature (2 wks ago)' },
  'tmax_lag_4':           { label: 'Heat',          desc: 'Past high temperatures' },
  'tmean_lag_1':          { label: 'Temperature',   desc: 'Average daily temperature' },
  'tmean_lag_2':          { label: 'Temperature',   desc: 'Average daily temperature (2 wks ago)' },
  'tmean_lag_4':          { label: 'Temperature',   desc: 'Average daily temperature (4 wks ago)' },
  'dtr_lag_1':            { label: 'Weather',       desc: 'Day-to-night temperature swing' },
  'dtr_lag_2':            { label: 'Weather',       desc: 'Temperature fluctuation' },

  // Rainfall & Surface Moisture
  'rain_lag_1':           { label: 'Rainfall',      desc: 'Recent rainfall volume' },
  'rain_lag_2':           { label: 'Rainfall',      desc: 'Rainfall 2 weeks ago' },
  'rain_lag_4':           { label: 'Rainfall',      desc: 'Rainfall 4 weeks ago' },
  'rain_lag_6':           { label: 'Rainfall',      desc: 'Rainfall 6 weeks ago' },
  'rain_roll_sum_2w':     { label: 'Rainfall',      desc: '2-week rainfall total' },
  'rain_roll_sum_4w':     { label: 'Rainfall',      desc: '4-week accumulated rainfall' },
  'jrc_occurrence':       { label: 'Water',         desc: 'Permanent and seasonal water bodies' },
  'frac_water':           { label: 'Water',         desc: 'Surface water cover percentage' },

  // Vegetation & Mosquito Suitability
  'suitability_lag_1':    { label: 'Breeding',      desc: 'Mosquito climate breeding suitability' },
  'suitability_lag_2':    { label: 'Breeding',      desc: 'Mosquito breeding index' },
  'suitability_lag_4':    { label: 'Breeding',      desc: 'Past mosquito breeding conditions' },
  'ndvi_mean':            { label: 'Greenery',      desc: 'Vegetation greenness index' },
  'frac_trees':           { label: 'Forest',        desc: 'Tree canopy cover' },
  'frac_built':           { label: 'Urban',         desc: 'Built-up urban environment' },
  'frac_shrub':           { label: 'Vegetation',    desc: 'Shrubland cover' },

  // Seasonality & Geography
  'cos_week':             { label: 'Season',        desc: 'Annual seasonal cycle' },
  'sin_week':             { label: 'Season',        desc: 'Seasonal monsoon timing' },
  'center_lat':           { label: 'Latitude',      desc: 'Geographic latitude' },
  'center_lon':           { label: 'Longitude',     desc: 'Geographic longitude' },

  // Epidemiological History
  'cases_lag_1':          { label: 'History',       desc: 'Past case history (1 wk ago)' },
  'cases_lag_2':          { label: 'History',       desc: 'Past case history (2 wks ago)' },
  'cases_lag_3':          { label: 'History',       desc: 'Past case history (3 wks ago)' },
  'cases_lag_4':          { label: 'History',       desc: 'Past case history (4 wks ago)' },
  'cases_lag_8':          { label: 'History',       desc: 'Past case history (8 wks ago)' },
  'case_rate_lag_1':      { label: 'Transmission',  desc: 'Recent transmission rate' },
  'cases_momentum_4w':    { label: 'Growth',        desc: 'Outbreak speed and acceleration' },
  'cases_roll_mean_4w':   { label: 'Baseline',      desc: 'Monthly baseline case average' },
  'cases_roll_mean_12w':  { label: 'Baseline',      desc: 'Quarterly baseline average' },
  'cases_roll_std_4w':    { label: 'Fluctuation',   desc: 'Outbreak volatility' },
  'neighbor_cases_k1_lag1':{ label: 'Spread',       desc: 'Nearby neighboring zone cases' },
  'neighbor_cases_k1_lag2':{ label: 'Spread',       desc: 'Neighboring zone cases' },
  'neighbor_cases_k2_lag1':{ label: 'Spread',       desc: 'Regional transmission spread' }
};

const getFeatureMeta = (rawKey) => {
  if (FEATURE_DICTIONARY[rawKey]) return FEATURE_DICTIONARY[rawKey];
  const clean = rawKey.split('_')[0];
  return {
    label: clean.charAt(0).toUpperCase() + clean.slice(1),
    desc: rawKey.replace(/_/g, ' ')
  };
};

/** Format a risk percentage — 1 decimal, capped at 99.9% */
const formatRisk = (raw) => {
  if (raw === undefined || raw === null) return null;
  return Math.min(99.9, Math.max(0.1, parseFloat(raw))).toFixed(1);
};

// ─────────────────────────────────────────────────────────────────────────────

const AnalyticsPanel = ({ cell, disease, horizon, onClose }) => {
  const [history, setHistory]   = useState([]);
  const [shap,    setShap]      = useState([]);
  // 'loading' | 'stale' | 'fresh'
  const [status,  setStatus]    = useState('loading');
  const aliveRef = useRef(null);

  useEffect(() => {
    if (!cell?.h3_index) return;

    // Mark previous effect as dead
    if (aliveRef.current) aliveRef.current = false;
    const alive = { current: true };
    aliveRef.current = alive;

    setStatus('loading');

    const histUrl = `/api/cell/${cell.h3_index}?disease=${disease}&horizon=${horizon}`;
    const shapUrl = `/api/cell/${cell.h3_index}/shap?disease=${disease}&horizon=${horizon}`;

    const applyHistory = (d) => {
      if (!alive.current) return;
      setHistory((d.history || []).map(h => ({
        week:      `W${h.week}`,
        'Past':    h.actual,
        'Forecast': h.pred_lgbm
      })));
    };

    const applyShap = (d) => {
      if (!alive.current) return;
      setShap(d.top_features || []);
    };

    // Fire both requests in parallel using SWR pattern
    Promise.all([
      cachedFetch(histUrl, (fresh) => { applyHistory(fresh); setStatus('fresh'); }),
      cachedFetch(shapUrl, (fresh) => { applyShap(fresh);    setStatus('fresh'); })
    ])
      .then(([histData, shapData]) => {
        if (!alive.current) return;
        applyHistory(histData);
        applyShap(shapData);
        // If data was served from cache immediately, status is already fine;
        // if it was a network wait, mark fresh
        setStatus(s => s === 'loading' ? 'fresh' : 'stale');
      })
      .catch(err => {
        if (!alive.current) return;
        console.error('Analytics fetch failed:', err);
        setStatus('fresh');
      });

    return () => { alive.current = false; };
  }, [cell?.h3_index, disease, horizon]);

  // Compute percentage contribution of top 10 drivers
  const processedFeatures = useMemo(() => {
    const top10 = shap.slice(0, 10);
    const sumShap = top10.reduce((a, c) => a + (c.mean_abs_shap || 0), 0) || 1;
    const maxVal  = top10.length > 0 ? (top10[0].mean_abs_shap || 1) : 1;

    return top10.map(item => {
      const meta = getFeatureMeta(item.feature);
      const val  = item.mean_abs_shap || 0;
      // Impact as a clean decimal percent, e.g. "23.4%"
      const impactPct = Math.min(99.9, Math.max(0.1, (val / sumShap) * 100)).toFixed(1);
      const barPct    = Math.max(6, Math.min(100, Math.round((val / maxVal) * 100)));
      return { key: item.feature, label: meta.label, desc: meta.desc, impactPct, barPct };
    });
  }, [shap]);

  const riskDisplay = formatRisk(cell.risk_percent);
  const isFirstLoad = status === 'loading' && history.length === 0 && shap.length === 0;

  return (
    <div className="analytics-panel glass-panel">
      {/* ── Header ── */}
      <div className="sidebar-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} color="var(--accent-blue)" />
          <h2>{cell.district || 'Area'} Overview</h2>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {status === 'stale' && (
            <RefreshCw size={12} style={{ opacity: 0.5, animation: 'spin 1.2s linear infinite' }} />
          )}
          {riskDisplay && (
            <span className="risk-badge" style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem' }}>
              {riskDisplay}% Risk
            </span>
          )}
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer', display: 'flex' }}
            title="Close"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* ── Content ── */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {isFirstLoad ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            Loading overview...
          </div>
        ) : (
          <>
            {/* ── Trend Chart ── */}
            <div className="panel-section" style={{ borderBottom: '1px solid var(--panel-border)' }}>
              <h3><BarChart2 size={14} /> Trend ({horizon} Wk Forecast)</h3>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="week" stroke="rgba(255,255,255,0.4)" fontSize={10} />
                    <YAxis stroke="rgba(255,255,255,0.4)" fontSize={10} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--bg-dark)', borderColor: 'var(--panel-border)', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff', fontSize: '11px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />
                    <Line type="monotone" name="Past Cases" dataKey="Past"     stroke="var(--text-secondary)" strokeWidth={2}   dot={false} />
                    <Line type="monotone" name="Forecast"   dataKey="Forecast" stroke="var(--accent-red)"    strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* ── Drivers ── */}
            <div className="panel-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3>Drivers</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Impact</span>
              </div>

              <div className="shap-list">
                {processedFeatures.map((item, idx) => (
                  <div key={idx} className="shap-item" title={`${item.label}: ${item.desc}`}>
                    <div className="shap-label" style={{ fontWeight: 500 }}>{item.label}</div>
                    <div className="shap-bar-bg">
                      <div className="shap-bar-fill" style={{ width: `${item.barPct}%` }}></div>
                    </div>
                    <div className="shap-value" style={{ fontWeight: 600, color: '#f8fafc', width: '46px' }}>
                      {item.impactPct}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default AnalyticsPanel;
