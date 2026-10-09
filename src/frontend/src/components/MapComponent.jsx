import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import { cellToBoundary } from 'h3-js';
import { idbGetGeo, idbSetGeo, idbPrune } from '../geoJsonStore';

/**
 * MapComponent — Persistent-cache + Zero-computation architecture:
 *
 *  1. On FIRST-EVER launch     → fetches /api/geojson/all, paints, stores in IndexedDB.
 *  2. On EVERY subsequent load → reads from IndexedDB in ~10ms, paints instantly.
 *  3. Cache is keyed to data version (year_week). When backend data changes → miss
 *     → fresh fetch + re-store. Old entries pruned automatically.
 *  4. All 4 horizons stored together per disease → horizon switch = O(1) memory read.
 *  5. GZip compression still reduces wire transfer by ~70% on a cache miss.
 */

const COLOR_STOPS = [
  'interpolate', ['linear'], ['get', 'risk_percent'],
  0,   'rgba(59, 130, 246, 0.18)',   // Safe - Blue
  25,  'rgba(34, 197, 94, 0.45)',    // Low - Green
  50,  'rgba(249, 115, 22, 0.65)',   // Moderate - Orange
  75,  'rgba(239, 68, 68, 0.85)',    // High - Red
  100, 'rgba(153, 27, 27, 1.0)',     // Critical - Dark Red
];

const LEGEND_GRADIENT = 'linear-gradient(to right, rgba(59, 130, 246, 0.5) 0%, rgba(34, 197, 94, 0.7) 25%, rgba(249, 115, 22, 0.8) 50%, rgba(239, 68, 68, 0.95) 75%, rgba(153, 27, 27, 1.0) 100%)';

const DARK_STYLE = {
  version: 8,
  sources: {
    'esri-dark': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: '&copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    },
    'esri-dark-ref': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
    },
  },
  layers: [
    {
      id: 'esri-dark-tiles',
      type: 'raster',
      source: 'esri-dark',
      minzoom: 0,
      maxzoom: 16,
    },
    {
      id: 'esri-dark-ref-tiles',
      type: 'raster',
      source: 'esri-dark-ref',
      minzoom: 0,
      maxzoom: 16,
    },
  ],
};

const LIGHT_STYLE = {
  version: 8,
  sources: {
    'esri-light': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: '&copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    },
    'esri-light-ref': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
    },
  },
  layers: [
    {
      id: 'esri-light-tiles',
      type: 'raster',
      source: 'esri-light',
      minzoom: 0,
      maxzoom: 16,
    },
    {
      id: 'esri-light-ref-tiles',
      type: 'raster',
      source: 'esri-light-ref',
      minzoom: 0,
      maxzoom: 16,
    },
  ],
};

const MapComponent = ({ disease, horizon, onCellClick, selectedCell, dataVersion, theme = 'dark' }) => {
  const mapContainer = useRef(null);
  const map          = useRef(null);
  const mapLoaded    = useRef(false);

  // In-memory runtime cache (cleared on hard refresh)
  const geoCache    = useRef({});
  const fetchingRef = useRef(new Set());
  const retryTimers = useRef({});  // per-disease retry timers

  // Subtle status overlay for first-time loads
  const [mapStatus, setMapStatus] = useState('');

  // ── 1. Init MapLibre once ──────────────────────────────────────────────
  useEffect(() => {
    if (map.current) return;

    const initialStyle = theme === 'light' ? LIGHT_STYLE : DARK_STYLE;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: initialStyle,
      center: [78.9629, 20.5937],
      zoom: 4,
      pitch: 40,
      attributionControl: false,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    const setupLayers = () => {
      if (!map.current.getSource('hex-grid')) {
        map.current.addSource('hex-grid', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
          buffer: 0,
          tolerance: 0.375,
        });
      }

      const refLayer = map.current.getLayer('esri-dark-ref-tiles') ? 'esri-dark-ref-tiles' : (map.current.getLayer('esri-light-ref-tiles') ? 'esri-light-ref-tiles' : undefined);

      if (!map.current.getLayer('hex-fill')) {
        map.current.addLayer({
          id: 'hex-fill',
          type: 'fill',
          source: 'hex-grid',
          paint: {
            'fill-color': COLOR_STOPS,
            'fill-opacity': 0.8,
          },
        }, refLayer);
      }

      if (!map.current.getSource('hex-highlight')) {
        map.current.addSource('hex-highlight', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
      }
      if (!map.current.getLayer('hex-highlight-line')) {
        map.current.addLayer({
          id: 'hex-highlight-line',
          type: 'line',
          source: 'hex-highlight',
          paint: { 'line-color': document.body.getAttribute('data-theme') === 'light' ? '#0f172a' : '#ffffff', 'line-width': 2.5 },
        }, refLayer);
      }
    };

    map.current.on('load', () => {
      setupLayers();

      // ── Hover popup ──
      const hoverPopup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
        offset: 12,
      });

      map.current.on('click', 'hex-fill', (e) => {
        if (e.features[0]) onCellClick(e.features[0].properties);
      });

      map.current.on('mousemove', 'hex-fill', (e) => {
        if (e.features.length > 0) {
          map.current.getCanvas().style.cursor = 'pointer';
          const p = e.features[0].properties;
          const rawPct = p.risk_percent !== undefined
            ? parseFloat(p.risk_percent)
            : (p.risk_score || 0) * 99.9;
          const pct = Math.min(99.9, Math.max(0.1, rawPct)).toFixed(1);
          hoverPopup
            .setLngLat(e.lngLat)
            .setHTML(`
              <div style="font-size: 11px; line-height: 1.5;">
                <div style="font-weight: 600;">${p.district || 'Area'}</div>
                ${p.state ? `<div style="font-size: 9px; opacity: 0.7;">${p.state}</div>` : ''}
                <div style="margin-top: 4px; color: #fca5a5; font-weight: 700; font-size: 13px;">
                  ${pct}% Risk
                </div>
              </div>
            `)
            .addTo(map.current);
        }
      });

      map.current.on('mouseleave', 'hex-fill', () => {
        map.current.getCanvas().style.cursor = '';
        hoverPopup.remove();
      });

      mapLoaded.current = true;

      // 1. Prioritize active disease immediately for fast first paint
      prefetchDisease(disease);

      // 2. Preload other diseases in background with delay to prevent bandwidth congestion
      ['dengue', 'malaria', 'syndemic']
        .filter(d => d !== disease)
        .forEach((d, idx) => {
          setTimeout(() => prefetchDisease(d), (idx + 1) * 2500);
        });
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── 2. Paint from in-memory cache instantly (O(1)) ────────────────────
  const paintFromCache = useCallback((dis, hor) => {
    if (!mapLoaded.current || !map.current) return;
    const entry = geoCache.current[dis]?.[String(hor)];
    if (!entry) return;
    map.current.getSource('hex-grid').setData(entry.geojson);
    map.current.setPaintProperty('hex-fill', 'fill-color', COLOR_STOPS);
  }, []);

  // ── 3. Load a disease — IDB-first, then network with retry ───────────
  const prefetchDisease = useCallback(async (dis, retryDelay = 5000) => {
    if (fetchingRef.current.has(dis)) return;
    if (geoCache.current[dis]) {
      paintFromCache(dis, horizon);
      return;
    }

    fetchingRef.current.add(dis);

    // ── 3a. Try IndexedDB first (survives page reload) ──────────────────
    if (dataVersion) {
      const cached = await idbGetGeo(dis, dataVersion);
      if (cached) {
        geoCache.current[dis] = cached;
        fetchingRef.current.delete(dis);
        setMapStatus('');
        paintFromCache(dis, horizon);
        console.log(`[GeoCache] IndexedDB hit: ${dis} v${dataVersion}`);
        return;
      }
    }

    // ── 3b. Cache miss — fetch from API ────────────────────────────────
    console.log(`[GeoCache] Fetching from API: ${dis}`);
    if (dis === disease) setMapStatus('Building map data...');

    fetch(`/api/geojson/all?disease=${dis}`)
      .then(async r => {
        if (r.status === 503) {
          // Backend GeoJSON is still baking — schedule a retry
          const after = parseInt(r.headers.get('Retry-After') || '10', 10) * 1000;
          const delay = Math.max(retryDelay, after);
          console.log(`[GeoCache] Backend not ready (503). Retry in ${delay / 1000}s…`);
          if (dis === disease) setMapStatus(`Map loading… (ready in ~${Math.round(delay / 1000)}s)`);
          fetchingRef.current.delete(dis);
          retryTimers.current[dis] = setTimeout(() => {
            prefetchDisease(dis, Math.min(delay * 1.5, 30000));
          }, delay);
          return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(payload => {
        if (!payload) return; // was a 503 retry — handled above
        const horizons = payload.horizons;
        geoCache.current[dis] = horizons;
        fetchingRef.current.delete(dis);
        if (dis === disease) setMapStatus('');
        paintFromCache(dis, horizon);

        // Persist to IndexedDB so next load is instant
        if (dataVersion) {
          idbSetGeo(dis, dataVersion, horizons).then(() => {
            console.log(`[GeoCache] Saved to IndexedDB: ${dis} v${dataVersion}`);
          });
        }
      })
      .catch(err => {
        console.error('[GeoCache] Fetch error:', err);
        fetchingRef.current.delete(dis);
        // On network error, retry after a short delay
        if (dis === disease) setMapStatus('Connecting to server…');
        retryTimers.current[dis] = setTimeout(() => prefetchDisease(dis, 8000), 5000);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horizon, paintFromCache, dataVersion, disease]);

  // ── 4. Prune stale IndexedDB entries once version is known ────────────
  useEffect(() => {
    if (!dataVersion) return;
    idbPrune(dataVersion);
  }, [dataVersion]);

  // ── 5. Disease change → load from IDB or fetch ───────────────────────
  useEffect(() => {
    if (!mapLoaded.current) return;
    prefetchDisease(disease);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disease, dataVersion]);

  // ── 6. Horizon change → instant in-memory read ───────────────────────
  useEffect(() => {
    if (!mapLoaded.current) return;
    paintFromCache(disease, horizon);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horizon]);

  // ── 7. Cell highlight ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapLoaded.current || !map.current) return;

    if (selectedCell?.h3_index) {
      const boundary = cellToBoundary(selectedCell.h3_index, true);
      map.current.getSource('hex-highlight').setData({
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [boundary] },
          properties: {}
        }],
      });

      if (selectedCell.fromSidebar) {
        let lng = 0, lat = 0;
        boundary.forEach(([x, y]) => { lng += x; lat += y; });
        map.current.flyTo({
          center: [lng / boundary.length, lat / boundary.length],
          zoom: 8, essential: true
        });
      }
    } else {
      map.current.getSource('hex-highlight')?.setData({ type: 'FeatureCollection', features: [] });
    }
  }, [selectedCell]);

  // ── 8. Theme toggle ──────────────────────────────────────────────────
  useEffect(() => {
    if (!mapLoaded.current || !map.current) return;
    const nextStyle = theme === 'light' ? LIGHT_STYLE : DARK_STYLE;
    
    map.current.setStyle(nextStyle);
    map.current.once('style.load', () => {
      if (!map.current.getSource('hex-grid')) {
        map.current.addSource('hex-grid', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
          buffer: 0,
          tolerance: 0.375,
        });
      }
      if (!map.current.getLayer('hex-fill')) {
        map.current.addLayer({
          id: 'hex-fill',
          type: 'fill',
          source: 'hex-grid',
          paint: {
            'fill-color': COLOR_STOPS,
            'fill-opacity': 0.8,
          },
        });
      }
      if (!map.current.getSource('hex-highlight')) {
        map.current.addSource('hex-highlight', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] },
        });
      }
      if (!map.current.getLayer('hex-highlight-line')) {
        map.current.addLayer({
          id: 'hex-highlight-line',
          type: 'line',
          source: 'hex-highlight',
          paint: { 'line-color': theme === 'light' ? '#0f172a' : '#ffffff', 'line-width': 2.5 },
        });
      }
      
      paintFromCache(disease, horizon);
      
      if (selectedCell?.h3_index) {
        const boundary = cellToBoundary(selectedCell.h3_index, true);
        map.current.getSource('hex-highlight').setData({
          type: 'FeatureCollection',
          features: [{
            type: 'Feature',
            geometry: { type: 'Polygon', coordinates: [boundary] },
            properties: {}
          }],
        });
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);

  return (
    <>
      <div ref={mapContainer} className="map-container" />

      {/* Subtle status banner shown only when map is first loading */}
      {mapStatus && (
        <div style={{
          position: 'absolute',
          bottom: '80px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(15,23,42,0.85)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: '8px',
          padding: '0.5rem 1.2rem',
          color: 'rgba(255,255,255,0.7)',
          fontSize: '0.8rem',
          letterSpacing: '0.4px',
          pointerEvents: 'none',
          zIndex: 10,
        }}>
          ⏳ {mapStatus}
        </div>
      )}

      <div className="map-legend glass-panel">
        <div className="legend-header">
          <span className="legend-title">Risk</span>
          <span className="legend-scale">0% – 99.9%</span>
        </div>
        <div className="legend-gradient" style={{ background: LEGEND_GRADIENT }}></div>
        <div className="legend-labels">
          <span>Safe</span>
          <span>Moderate</span>
          <span>Critical</span>
        </div>
      </div>
    </>
  );
};

export default MapComponent;
