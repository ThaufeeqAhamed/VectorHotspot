import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, X, MapPin, Layers, Hexagon, Compass, Clock, ArrowRight, Sparkles, Loader2 } from 'lucide-react';
import { latLngToCell, gridDisk } from 'h3-js';
import { registerLocation } from '../locationRegistry';

const POPULAR_HUBS = [
  { name: 'Mumbai', state: 'Maharashtra', desc: 'Financial capital • High population density' },
  { name: 'Bengaluru Urban', state: 'Karnataka', desc: 'Southern hub • Monsoon vector corridor' },
  { name: 'Belagavi', state: 'Karnataka', desc: 'Active surveillance hotspot' },
  { name: 'Delhi', state: 'Delhi', desc: 'National capital region' },
  { name: 'Pune', state: 'Maharashtra', desc: 'Western plateau urban center' },
  { name: 'Kolkata', state: 'West Bengal', desc: 'Eastern delta high-humidity zone' },
  { name: 'Chennai', state: 'Tamil Nadu', desc: 'Coastal tropical transmission zone' },
  { name: 'Ernakulam', state: 'Kerala', desc: 'High rainfall coastal belt' },
];

const STORAGE_KEY = 'vectorhotspot_recent_searches_v1';

export default function SearchBar({ disease, horizon, onSelectLocation }) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ districts: [], states: [], cells: [], places: [] });
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [recentSearches, setRecentSearches] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved).slice(0, 6) : [];
    } catch {
      return [];
    }
  });

  const containerRef = useRef(null);
  const inputRef = useRef(null);
  const abortControllerRef = useRef(null);
  const osmAbortRef = useRef(null);


  const saveToRecent = useCallback((item) => {
    try {
      const current = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      const filtered = current.filter(r => r.key !== item.key);
      const updated = [item, ...filtered].slice(0, 8);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      setRecentSearches(updated);
    } catch {
      // ignore
    }
  }, []);

  // Global keyboard shortcut to focus search: '/' or 'Ctrl+K' / 'Cmd+K'
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.key === '/' && document.activeElement !== inputRef.current && !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) ||
          ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k')) {
        e.preventDefault();
        inputRef.current?.focus();
        setIsOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  // Fetch search results with debouncing
  useEffect(() => {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      setResults({ districts: [], states: [], cells: [], places: [] });
      setLoading(false);
      setSelectedIndex(-1);
      return;
    }

    if (abortControllerRef.current) abortControllerRef.current.abort();
    if (osmAbortRef.current) osmAbortRef.current.abort();

    const ac = new AbortController();
    abortControllerRef.current = ac;
    setLoading(true);

    const timer = setTimeout(async () => {
      try {
        const res = await fetch(
          `/api/search?q=${encodeURIComponent(cleanQuery)}&disease=${disease}&horizon=${horizon}&limit=8`,
          { signal: ac.signal }
        );
        const data = await res.json();

        let places = [];
        // If query is text (not an H3 hex index) and length >= 3, query OSM Nominatim for landmarks/towns
        const isHexCell = cleanQuery.startsWith('87') || (cleanQuery.length >= 8 && /^[0-9a-fA-F]+$/.test(cleanQuery));
        if (!isHexCell && cleanQuery.length >= 3 && (!data.districts || data.districts.length < 3)) {
          try {
            const osmAc = new AbortController();
            osmAbortRef.current = osmAc;
            const osmRes = await fetch(
              `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(cleanQuery)}&countrycodes=in&limit=4&addressdetails=1`,
              {
                signal: osmAc.signal,
                headers: {
                  'Accept-Language': 'en',
                  'User-Agent': 'VectorHotspot-Dashboard/1.0'
                }
              }
            );
            if (osmRes.ok) {
              const osmData = await osmRes.json();
              places = osmData.map(p => {
                const lat = parseFloat(p.lat);
                const lon = parseFloat(p.lon);
                let derivedH3 = null;
                try {
                  derivedH3 = latLngToCell(lat, lon, 7);
                } catch {
                  // ignore
                }
                const addr = p.address || {};
                const localityName = p.name || addr.suburb || addr.neighbourhood || addr.residential ||
                                     addr.town || addr.village || addr.city_district || p.display_name.split(',')[0].trim();
                const districtName = addr.state_district || addr.county || addr.city || '';
                const stateName = addr.state || 'India';
                return {
                  name: localityName,
                  fullName: p.display_name,
                  district: districtName,
                  state: stateName,
                  lat,
                  lon,
                  type: p.type || p.class || 'place',
                  h3_index: derivedH3
                };
              });
            }
          } catch {
            // OSM lookup failed or aborted; ignore
          }
        }

        setResults({
          districts: data.districts || [],
          states: data.states || [],
          cells: data.cells || [],
          places
        });
        setLoading(false);
        setSelectedIndex(-1);
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Search error:', err);
          setLoading(false);
        }
      }
    }, 180);

    return () => {
      clearTimeout(timer);
      ac.abort();
    };
  }, [query, disease, horizon]);

  // Flatten results for keyboard navigation
  const flatItems = [
    ...(results.districts.map(d => ({ ...d, _type: 'district', key: `dist-${d.district}` }))),
    ...(results.states.map(s => ({ ...s, _type: 'state', key: `state-${s.state}` }))),
    ...(results.cells.map(c => ({ ...c, _type: 'cell', key: `cell-${c.h3_index}` }))),
    ...(results.places.map(p => ({ ...p, _type: 'place', key: `place-${p.lat}-${p.lon}` })))
  ];

  const handleSelect = (item) => {
    setIsOpen(false);
    if (!item) return;

    if (item._type === 'district') {
      saveToRecent({
        key: `dist-${item.district}`,
        _type: 'district',
        district: item.district,
        state: item.state,
        bounds: item.bounds,
        center: [item.center_lon, item.center_lat],
        cell_count: item.cell_count,
        top_cell: item.top_cell
      });

      onSelectLocation({
        type: 'district',
        name: item.district,
        state: item.state,
        bounds: item.bounds,
        center: [item.center_lon, item.center_lat],
        top_cell: item.top_cell
      });
      setQuery(item.district);
    } else if (item._type === 'state') {
      saveToRecent({
        key: `state-${item.state}`,
        _type: 'state',
        state: item.state,
        bounds: item.bounds,
        center: [item.center_lon, item.center_lat],
        district_count: item.district_count
      });

      onSelectLocation({
        type: 'state',
        name: item.state,
        bounds: item.bounds,
        center: [item.center_lon, item.center_lat]
      });
      setQuery(item.state);
    } else if (item._type === 'cell') {
      saveToRecent({
        key: `cell-${item.h3_index}`,
        _type: 'cell',
        h3_index: item.h3_index,
        district: item.district,
        state: item.state,
        center: [item.center_lon, item.center_lat],
        risk_percent: item.risk_percent
      });

      onSelectLocation({
        type: 'cell',
        h3_index: item.h3_index,
        district: item.district,
        state: item.state,
        center: [item.center_lon, item.center_lat],
        risk_percent: item.risk_percent,
        risk_score: item.risk_score
      });
      setQuery(item.h3_index);
    } else if (item._type === 'place') {
      saveToRecent({
        key: `place-${item.lat}-${item.lon}`,
        _type: 'place',
        name: item.name,
        fullName: item.fullName,
        district: item.district,
        state: item.state,
        center: [item.lon, item.lat],
        h3_index: item.h3_index
      });

      // Synchronize with shared location registry so hover and analytics panel show identical real-world name
      if (item.h3_index) {
        let cellsToRegister = [item.h3_index];
        try {
          cellsToRegister = gridDisk(item.h3_index, 1);
        } catch {
          // ignore
        }
        cellsToRegister.forEach(h3 => {
          registerLocation(h3, {
            locationName: item.name,
            district: item.district,
            state: item.state,
            fullName: item.fullName,
            lat: item.lat,
            lon: item.lon,
            isResolved: true
          });
        });
      }

      onSelectLocation({
        type: 'place',
        name: item.name,
        fullName: item.fullName,
        district: item.district,
        state: item.state,
        center: [item.lon, item.lat],
        h3_index: item.h3_index
      });
      setQuery(item.name);
    }
  };

  const handleKeyDown = (e) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown') {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (prev < flatItems.length - 1 ? prev + 1 : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev > 0 ? prev - 1 : flatItems.length - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && selectedIndex < flatItems.length) {
        handleSelect(flatItems[selectedIndex]);
      } else if (flatItems.length > 0) {
        handleSelect(flatItems[0]);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
      inputRef.current?.blur();
    }
  };

  const handleHubClick = (hub) => {
    setQuery(hub.name);
    setIsOpen(true);
    inputRef.current?.focus();
  };

  const clearSearch = () => {
    setQuery('');
    setResults({ districts: [], states: [], cells: [], places: [] });
    setSelectedIndex(-1);
    inputRef.current?.focus();
  };

  const getRiskBadge = (riskPct) => {
    if (riskPct === undefined || riskPct === null) return null;
    const val = parseFloat(riskPct);
    let colorClass = 'badge-low';
    if (val >= 70) colorClass = 'badge-critical';
    else if (val >= 40) colorClass = 'badge-moderate';

    return (
      <span className={`search-risk-badge ${colorClass}`}>
        {val.toFixed(1)}% Peak
      </span>
    );
  };

  const hasResults = flatItems.length > 0;
  const isQuerying = query.trim().length > 0;

  return (
    <div className="search-bar-container" ref={containerRef}>
      <div className={`search-input-wrapper ${isOpen ? 'focused' : ''}`}>
        <Search size={16} className="search-icon" />
        
        <input
          ref={inputRef}
          type="text"
          className="search-input"
          placeholder="Search place, district, state, or H3 cell... (/)"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
        />

        {loading ? (
          <Loader2 size={15} className="search-spinner" />
        ) : query ? (
          <button type="button" className="search-clear-btn" onClick={clearSearch} title="Clear">
            <X size={14} />
          </button>
        ) : (
          <kbd className="search-hotkey-badge">/</kbd>
        )}
      </div>

      {isOpen && (
        <div className="search-dropdown glass-panel">
          {/* Active query results */}
          {isQuerying && hasResults && (
            <div className="search-results-list">
              {/* Districts */}
              {results.districts.length > 0 && (
                <div className="search-section">
                  <div className="search-section-title">
                    <MapPin size={12} />
                    <span>Districts ({results.districts.length})</span>
                  </div>
                  {results.districts.map((d) => {
                    const isSelected = flatItems[selectedIndex]?.key === `dist-${d.district}`;
                    return (
                      <div
                        key={`dist-${d.district}`}
                        className={`search-item ${isSelected ? 'active' : ''}`}
                        onClick={() => handleSelect({ ...d, _type: 'district' })}
                      >
                        <div className="search-item-left">
                          <div className="search-item-title">
                            <strong>{d.district}</strong>
                            <span className="search-item-sub">{d.state}</span>
                          </div>
                          <div className="search-item-meta">
                            <span className="meta-tag">{d.cell_count?.toLocaleString()} risk cells</span>
                          </div>
                        </div>
                        <div className="search-item-right">
                          {d.top_cell?.risk_percent && getRiskBadge(d.top_cell.risk_percent)}
                          <ArrowRight size={13} className="item-arrow" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* H3 Cells */}
              {results.cells.length > 0 && (
                <div className="search-section">
                  <div className="search-section-title">
                    <Hexagon size={12} />
                    <span>H3 Resolution-7 Cells ({results.cells.length})</span>
                  </div>
                  {results.cells.map((c) => {
                    const isSelected = flatItems[selectedIndex]?.key === `cell-${c.h3_index}`;
                    return (
                      <div
                        key={`cell-${c.h3_index}`}
                        className={`search-item ${isSelected ? 'active' : ''}`}
                        onClick={() => handleSelect({ ...c, _type: 'cell' })}
                      >
                        <div className="search-item-left">
                          <div className="search-item-title">
                            <code className="cell-code">{c.h3_index}</code>
                          </div>
                          <div className="search-item-meta">
                            <span>{c.district}, {c.state}</span>
                          </div>
                        </div>
                        <div className="search-item-right">
                          {c.risk_percent !== undefined && getRiskBadge(c.risk_percent)}
                          <ArrowRight size={13} className="item-arrow" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* States */}
              {results.states.length > 0 && (
                <div className="search-section">
                  <div className="search-section-title">
                    <Layers size={12} />
                    <span>States & Regions</span>
                  </div>
                  {results.states.map((s) => {
                    const isSelected = flatItems[selectedIndex]?.key === `state-${s.state}`;
                    return (
                      <div
                        key={`state-${s.state}`}
                        className={`search-item ${isSelected ? 'active' : ''}`}
                        onClick={() => handleSelect({ ...s, _type: 'state' })}
                      >
                        <div className="search-item-left">
                          <div className="search-item-title">
                            <strong>{s.state}</strong>
                          </div>
                          <div className="search-item-meta">
                            <span className="meta-tag">{s.district_count} districts</span>
                            <span className="meta-tag">{s.cell_count?.toLocaleString()} cells</span>
                          </div>
                        </div>
                        <div className="search-item-right">
                          <ArrowRight size={13} className="item-arrow" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Places / Landmarks from Geocoder */}
              {results.places.length > 0 && (
                <div className="search-section">
                  <div className="search-section-title">
                    <Compass size={12} />
                    <span>Places & Landmarks</span>
                  </div>
                  {results.places.map((p) => {
                    const isSelected = flatItems[selectedIndex]?.key === `place-${p.lat}-${p.lon}`;
                    return (
                      <div
                        key={`place-${p.lat}-${p.lon}`}
                        className={`search-item ${isSelected ? 'active' : ''}`}
                        onClick={() => handleSelect({ ...p, _type: 'place' })}
                      >
                        <div className="search-item-left">
                          <div className="search-item-title">
                            <strong>{p.name}</strong>
                          </div>
                          <div className="search-item-meta" title={p.fullName}>
                            <span className="search-truncate">{p.fullName}</span>
                          </div>
                        </div>
                        <div className="search-item-right">
                          {p.h3_index && <span className="meta-h3-tag">⬡ {p.h3_index.slice(0, 7)}...</span>}
                          <ArrowRight size={13} className="item-arrow" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* No results */}
          {isQuerying && !hasResults && !loading && (
            <div className="search-empty-state">
              <Compass size={24} style={{ opacity: 0.4, marginBottom: '0.5rem' }} />
              <p>No places or H3 cells matching <strong>"{query}"</strong></p>
              <span className="empty-subtext">Try searching for a city, district, state name, or a 15-character H3 hex ID (e.g. 87608b...)</span>
            </div>
          )}

          {/* Default state: Popular Metros + Recent Searches */}
          {!isQuerying && (
            <div className="search-default-view">
              {recentSearches.length > 0 && (
                <div className="search-section">
                  <div className="search-section-title">
                    <Clock size={12} />
                    <span>Recent Searches</span>
                  </div>
                  <div className="recent-chips">
                    {recentSearches.map((rec) => (
                      <button
                        key={rec.key}
                        type="button"
                        className="recent-chip"
                        onClick={() => handleSelect(rec)}
                      >
                        <MapPin size={11} />
                        <span>{rec.district || rec.state || rec.name || rec.h3_index}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="search-section">
                <div className="search-section-title">
                  <Sparkles size={12} />
                  <span>Popular Regions & Hotspot Hubs</span>
                </div>
                <div className="hubs-grid">
                  {POPULAR_HUBS.map((hub) => (
                    <div
                      key={hub.name}
                      className="hub-card"
                      onClick={() => handleHubClick(hub)}
                    >
                      <div className="hub-header">
                        <strong>{hub.name}</strong>
                        <span className="hub-state">{hub.state}</span>
                      </div>
                      <p className="hub-desc">{hub.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="search-footer-hint">
                <span>Use <kbd>↑</kbd> <kbd>↓</kbd> to navigate • <kbd>Enter</kbd> to select • <kbd>Esc</kbd> to close</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
