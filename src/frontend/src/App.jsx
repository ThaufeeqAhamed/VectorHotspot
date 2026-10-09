import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Sun, Moon } from 'lucide-react';
import MapComponent from './components/MapComponent';
import HotspotSidebar from './components/HotspotSidebar';
import AnalyticsPanel from './components/AnalyticsPanel';
import SearchBar from './components/SearchBar';
import { primeCache } from './apiCache';
import { registerLocation } from './locationRegistry';
import { gridDisk } from 'h3-js';


function App() {
  const [loading, setLoading] = useState(true);
  const [disease, setDisease] = useState('dengue');
  const [horizon, setHorizon] = useState(1);
  const [metadata, setMetadata] = useState(null);
  // Data version string — used as IndexedDB cache key for GeoJSON
  // Changes when backend updates to a new week, auto-invalidating the cache
  const [dataVersion, setDataVersion] = useState(null);
  const [theme, setTheme] = useState('dark');
  const horizonDebounce = useRef(null);
  
  // Selected cell for analytics drill-down
  const [selectedCell, setSelectedCell] = useState(null);
  // Map navigation target from SearchBar
  const [navigationTarget, setNavigationTarget] = useState(null);
  // Feedback toast for navigation confirmations
  const [searchFeedback, setSearchFeedback] = useState(null);
  const feedbackTimeout = useRef(null);


  // Debounced horizon setter — avoids painting on every tick of a drag
  const setHorizonDebounced = useCallback((val) => {
    clearTimeout(horizonDebounce.current);
    horizonDebounce.current = setTimeout(() => setHorizon(val), 80);
  }, []);

  const showFeedback = useCallback((msg) => {
    clearTimeout(feedbackTimeout.current);
    setSearchFeedback(msg);
    feedbackTimeout.current = setTimeout(() => {
      setSearchFeedback(null);
    }, 4500);
  }, []);

  const handleSelectLocation = useCallback((target) => {
    setNavigationTarget({
      ...target,
      _ts: Date.now()
    });

    if (target.type === 'cell') {
      setSelectedCell({
        h3_index: target.h3_index,
        district: target.district,
        state: target.state,
        risk_percent: target.risk_percent,
        risk_score: target.risk_score,
        fromSearch: true
      });
      showFeedback(`⬡ Cell: ${target.h3_index} (${target.district || 'India'})`);
    } else if (target.type === 'district') {
      if (target.top_cell?.h3_index) {
        setSelectedCell({
          h3_index: target.top_cell.h3_index,
          district: target.name,
          state: target.state,
          risk_percent: target.top_cell.risk_percent,
          risk_score: target.top_cell.risk_score,
          fromSearch: true
        });
        showFeedback(`📍 Navigated to ${target.name}, ${target.state} • Peak Risk Cell selected (${target.top_cell.risk_percent}%)`);
      } else {
        showFeedback(`📍 Navigated to ${target.name}, ${target.state}`);
      }
    } else if (target.type === 'place') {
      if (target.h3_index) {
        let cellsToRegister = [target.h3_index];
        try {
          cellsToRegister = gridDisk(target.h3_index, 1);
        } catch {
          // ignore
        }
        cellsToRegister.forEach(h3 => {
          registerLocation(h3, {
            locationName: target.name,
            district: target.district || target.name,
            state: target.state || 'India',
            fullName: target.fullName
          });
        });

        setSelectedCell({
          h3_index: target.h3_index,
          locationName: target.name,
          district: target.district || target.name,
          state: target.state || 'India',
          fullName: target.fullName,
          fromSearch: true
        });
      }
      const distInfo = target.district && target.district !== target.name ? ` (${target.district})` : '';
      showFeedback(`📍 Navigated to ${target.name}${distInfo}`);
    } else if (target.type === 'state') {
      showFeedback(`🗺️ Region: ${target.name}`);
    }
  }, [showFeedback]);

  useEffect(() => {
    const init = async () => {
      try {
        const res = await fetch('/api/metadata');
        const data = await res.json();
        setMetadata(data);
        setDataVersion(`${data.latest_year}_W${data.latest_week}`);
        setLoading(false);
        // Prime hotspot cache for all disease-horizon pairs in background
        // so the sidebar is warm before the user changes anything
        const diseases = data.diseases || ['dengue', 'malaria', 'syndemic'];
        diseases.forEach(d => {
          [1, 2, 3, 4].forEach(h => {
            primeCache(`/api/hotspots?disease=${d}&horizon=${h}&limit=20`);
          });
        });
      } catch (err) {
        console.error('Failed to load metadata:', err);
        setLoading(false);
      }
    };
    init();
  }, []);

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  if (loading) {
    return (
      <div className="loader-overlay">
        <div className="spinner"></div>
        <p>Loading VectorHotspot...</p>
      </div>
    );
  }

  const getPredictedDate = (baseYear, baseWeek, horizonSteps) => {
    let w = baseWeek + horizonSteps;
    let y = baseYear;
    while (w > 52) {
      w -= 52;
      y += 1;
    }
    return `Week ${w} / ${y}`;
  };


  return (
    <>
      <MapComponent 
        disease={disease} 
        horizon={horizon} 
        onCellClick={setSelectedCell}
        selectedCell={selectedCell}
        navigationTarget={navigationTarget}
        dataVersion={dataVersion}
        theme={theme}
      />
      
      <div className="ui-layer">
        <header className="header-bar glass-panel">
          <div className="brand">
            <div className="brand-dot"></div>
            <h1>VectorHotspot</h1>
            <span className="brand-target-info">
              Target: <strong style={{color: '#fca5a5', letterSpacing: '0.5px'}}>{metadata ? getPredictedDate(metadata.latest_year, metadata.latest_week, horizon) : ''}</strong>
              <span style={{opacity: 0.5, marginLeft: '0.5rem'}}>(Current: Week {metadata?.latest_week})</span>
            </span>
          </div>

          {/* Global Search Bar */}
          <div className="header-search-slot">
            <SearchBar 
              disease={disease} 
              horizon={horizon} 
              onSelectLocation={handleSelectLocation} 
            />
          </div>
          
          <div className="controls">
            <div className="control-group">
              <label>Disease</label>
              <select value={disease} onChange={(e) => {setDisease(e.target.value); setSelectedCell(null);}}>
                {metadata?.diseases.map(d => (
                  <option key={d} value={d}>
                    {d === 'syndemic' ? 'Combined' : d.charAt(0).toUpperCase() + d.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="control-group" style={{marginLeft: '1rem'}}>
              <label>Forecast: {horizon} {horizon === 1 ? 'Week' : 'Weeks'}</label>
              <input 
                type="range" 
                min="1" max="4" 
                value={horizon} 
                onChange={(e) => { setHorizonDebounced(parseInt(e.target.value)); setSelectedCell(null);}}
                className="slider"
              />
            </div>
            
            <button 
              onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              style={{
                background: 'var(--select-bg)', 
                border: '1px solid var(--panel-border)', 
                color: 'var(--text-primary)', 
                padding: '0.5rem',
                marginLeft: '1rem',
                borderRadius: '6px', 
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              title="Toggle Theme"
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>

        {/* Floating Navigation Feedback Toast */}
        {searchFeedback && (
          <div className="search-toast glass-panel">
            <span>{searchFeedback}</span>
            <button className="toast-dismiss-btn" onClick={() => setSearchFeedback(null)}>×</button>
          </div>
        )}

        <div className="content-area">
          <HotspotSidebar 
            disease={disease} 
            horizon={horizon} 
            onHotspotClick={setSelectedCell}
          />
          
          {selectedCell && (
            <AnalyticsPanel 
              cell={selectedCell} 
              disease={disease} 
              horizon={horizon}
              onClose={() => setSelectedCell(null)}
            />
          )}
        </div>
      </div>
    </>
  );
}

export default App;

