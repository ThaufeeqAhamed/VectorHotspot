import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, MapPin, RefreshCw } from 'lucide-react';
import { cachedFetch } from '../apiCache';

const HotspotSidebar = ({ disease, horizon, onHotspotClick }) => {
  const [hotspots, setHotspots] = useState([]);
  // 'idle' | 'loading' | 'stale' | 'fresh'
  const [status, setStatus] = useState('idle');
  const abortRef = useRef(null);

  useEffect(() => {
    // Cancel any in-flight background revalidation
    if (abortRef.current) abortRef.current = false;
    const alive = { current: true };
    abortRef.current = alive;

    const url = `/api/hotspots?disease=${disease}&horizon=${horizon}&limit=20`;

    setStatus('loading');

    cachedFetch(url, (freshData) => {
      // Background revalidation callback — only update if still mounted for same query
      if (!alive.current) return;
      setHotspots(freshData.top_cells || []);
      setStatus('fresh');
    })
      .then(data => {
        if (!alive.current) return;
        setHotspots(data.top_cells || []);
        // If we got cache, mark as stale (background revalidation running)
        setStatus(prev => prev === 'loading' ? 'fresh' : 'stale');
      })
      .catch(err => {
        if (!alive.current) return;
        console.error('Hotspot fetch failed:', err);
        setStatus('fresh');
      });

    return () => { alive.current = false; };
  }, [disease, horizon]);

  // Format risk_percent with up to 1 decimal place, capped at 99.9
  const formatPct = (spot) => {
    const raw = spot.risk_percent;
    if (raw === undefined || raw === null) return '—';
    const val = Math.min(99.9, Math.max(0.1, parseFloat(raw)));
    // Show one decimal unless it's a whole number (e.g. 45.0 → 45.0)
    return val.toFixed(1);
  };

  // Colour the badge by severity
  const badgeStyle = (pct) => {
    const val = parseFloat(pct);
    if (val >= 75) return { background: 'rgba(239,68,68,0.18)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.4)' };
    if (val >= 40) return { background: 'rgba(249,115,22,0.18)', color: '#fdba74', border: '1px solid rgba(249,115,22,0.4)' };
    return { background: 'rgba(34,197,94,0.12)', color: '#86efac', border: '1px solid rgba(34,197,94,0.3)' };
  };

  const isFirstLoad = status === 'loading' && hotspots.length === 0;

  return (
    <div className="sidebar glass-panel">
      <div className="sidebar-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertTriangle size={18} color="var(--accent-red)" />
          <h2>Hotspots</h2>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          {status === 'stale' && (
            <RefreshCw size={12} style={{ opacity: 0.5, animation: 'spin 1.2s linear infinite' }} />
          )}
          <span className="horizon-tag">
            {horizon} {horizon === 1 ? 'Week' : 'Weeks'}
          </span>
        </div>
      </div>

      <div className="hotspot-list">
        {isFirstLoad ? (
          <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-secondary)' }}>Loading...</div>
        ) : hotspots.length === 0 ? (
          <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-secondary)' }}>No high-risk areas found.</div>
        ) : (
          hotspots.map((spot, idx) => {
            const pct = formatPct(spot);
            return (
              <div
                key={`${spot.h3_index}-${idx}`}
                className="hotspot-card"
                onClick={() => onHotspotClick({ ...spot, fromSidebar: true })}
              >
                <div className="hotspot-info">
                  <h3>{spot.district || 'Area'}</h3>
                  <p style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                    <MapPin size={12} /> {spot.state || 'India'}
                  </p>
                  <p style={{ fontFamily: 'monospace', marginTop: '0.25rem', opacity: 0.6 }} title={spot.h3_index}>
                    Zone: {spot.h3_index.slice(0, 10)}...
                  </p>
                </div>
                <div className="risk-badge" style={badgeStyle(pct)}>
                  {pct}%
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default HotspotSidebar;
