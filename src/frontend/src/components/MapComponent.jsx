import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import { cellToBoundary } from 'h3-js';
import { idbGetGeo, idbSetGeo, idbPrune } from '../geoJsonStore';
import { getLocation, registerLocation, resolveCellLocation } from '../locationRegistry';

/**
 * MapComponent — Multi-Resolution India-Wide Risk Map Engine
 *
 * Architecture:
 *  1. Level 1 (Zoom < 7.0): National Overview Layer
 *     - 2,030 H3 Res-4 parent hexagons covering 100% of India (North, South, East, West, Central, Northeast).
 *     - Precomputed hybrid risk (mean + peak hotspot awareness) loaded in ~50ms (<180 KB).
 *     - Eliminates the previous 50k top-N bias entirely.
 *
 *  2. Level 2 & 3 (Zoom >= 7.0): Viewport Detail Layer
 *     - Exact H3 Res-7 cells lazy-loaded dynamically for the visible viewport.
 *     - Powered by SSD-backed SQLite R*Tree spatial index on backend (<25ms).
 *     - In-memory spatial LRU cache prevents refetching during panning.
 *     - Smooth cross-fade interpolation eliminates blank flashes.
 */

const COLOR_STOPS = [
  'interpolate', ['linear'], ['get', 'risk_percent'],
  0,   'rgba(59, 130, 246, 0.22)',   // Safe - Blue
  25,  'rgba(34, 197, 94, 0.50)',    // Low - Green
  50,  'rgba(249, 115, 22, 0.70)',   // Moderate - Orange
  75,  'rgba(239, 68, 68, 0.88)',    // High - Red
  100, 'rgba(153, 27, 27, 1.0)',     // Critical - Dark Red
];

const LEGEND_GRADIENT = 'linear-gradient(to right, rgba(59, 130, 246, 0.5) 0%, rgba(34, 197, 94, 0.7) 25%, rgba(249, 115, 22, 0.8) 50%, rgba(239, 68, 68, 0.95) 75%, rgba(153, 27, 27, 1.0) 100%)';

const createPopupHTML = ({ displayName, displaySub, pct, isAgg, h3_index, cell_count }) => `
  <div class="vh-popup-card">
    <div class="vh-popup-header">
      <span class="vh-popup-title" title="${displayName}">${displayName}</span>
      ${isAgg ? '<span class="vh-popup-badge">Regional</span>' : ''}
    </div>
    ${displaySub ? `<div class="vh-popup-subtitle">${displaySub}</div>` : ''}
    <div class="vh-popup-risk">
      <span>${pct}%</span>
      <span class="vh-popup-risk-label">Predicted Risk</span>
    </div>
    ${isAgg ? `
      <div class="vh-popup-footer">
        &bull; Regional average of ${cell_count || 'multiple'} cells<br/>
        &bull; Click to drill down into district
      </div>
    ` : `
      <div class="vh-popup-footer">
        Cell: <span style="font-family: monospace; font-size: 9px; opacity: 0.8;">${h3_index?.slice(0, 10)}...</span>
      </div>
    `}
  </div>
`;

const UNIFIED_STYLE = {
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
      id: 'esri-dark-tiles',
      type: 'raster',
      source: 'esri-dark',
      minzoom: 0,
      maxzoom: 16,
    },
    {
      id: 'esri-light-tiles',
      type: 'raster',
      source: 'esri-light',
      minzoom: 0,
      maxzoom: 16,
      layout: { visibility: 'none' },
    },
    {
      id: 'esri-dark-ref-tiles',
      type: 'raster',
      source: 'esri-dark-ref',
      minzoom: 0,
      maxzoom: 16,
    },
    {
      id: 'esri-light-ref-tiles',
      type: 'raster',
      source: 'esri-light-ref',
      minzoom: 0,
      maxzoom: 16,
      layout: { visibility: 'none' },
    },
  ],
};

const MapComponent = ({ disease, horizon, onCellClick, selectedCell, dataVersion, theme = 'dark', navigationTarget }) => {
  const mapContainer = useRef(null);
  const map          = useRef(null);
  const mapLoaded    = useRef(false);
  const activeRippleMarker = useRef(null);
  const selectedCellRef = useRef(selectedCell);
  const hoverResolveTimer = useRef(null);
  const hoveredCellId = useRef(null);
  const hoverPopupRef = useRef(null);
  useEffect(() => {
    selectedCellRef.current = selectedCell;
    if (selectedCell?.h3_index && selectedCell?.locationName) {
      registerLocation(selectedCell.h3_index, {
        locationName: selectedCell.locationName,
        district: selectedCell.district,
        state: selectedCell.state,
        fullName: selectedCell.fullName
      });
    }
  }, [selectedCell]);

  // In-memory runtime cache for national overview (all 4 horizons)
  const geoCache    = useRef({});
  const fetchingRef = useRef(new Set());
  const retryTimers = useRef({});

  // Viewport request controller & client spatial cache
  const activeAbortController = useRef(null);
  const viewportCache = useRef(new Map());
  const viewportDebounce = useRef(null);

  // UI state
  const [mapStatus, setMapStatus] = useState('');
  const [currentZoom, setCurrentZoom] = useState(4);

  // ── 1. Setup Map Layers ──────────────────────────────────────────────
  const setupLayers = useCallback(() => {
    if (!map.current) return;
    const refLayer = map.current.getLayer('esri-dark-ref-tiles') ? 'esri-dark-ref-tiles' : (map.current.getLayer('esri-light-ref-tiles') ? 'esri-light-ref-tiles' : undefined);

    // 1a. National Overview Source & Layer (H3 Res-4)
    if (!map.current.getSource('overview-grid')) {
      map.current.addSource('overview-grid', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
        buffer: 0,
        tolerance: 0.375,
      });
    }

    if (!map.current.getLayer('overview-fill')) {
      map.current.addLayer({
        id: 'overview-fill',
        type: 'fill',
        source: 'overview-grid',
        paint: {
          'fill-color': COLOR_STOPS,
          // Smoothly fade out overview as detailed cells appear
          'fill-opacity': [
            'interpolate', ['linear'], ['zoom'],
            6.2, 0.8,
            7.8, 0.0
          ],
        },
      }, refLayer);
    }

    // 1b. Viewport Detailed Cells Source & Layer (H3 Res-7)
    if (!map.current.getSource('detail-grid')) {
      map.current.addSource('detail-grid', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
        buffer: 0,
        tolerance: 0.1,
      });
    }

    if (!map.current.getLayer('detail-fill')) {
      map.current.addLayer({
        id: 'detail-fill',
        type: 'fill',
        source: 'detail-grid',
        paint: {
          'fill-color': COLOR_STOPS,
          // Smoothly fade in detailed cells
          'fill-opacity': [
            'interpolate', ['linear'], ['zoom'],
            6.2, 0.0,
            7.4, 0.85
          ],
        },
      }, refLayer);
    }

    // 1c. Hex Highlight Source & Layer (for selected cells)
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
        paint: {
          'line-color': document.body.getAttribute('data-theme') === 'light' ? '#0f172a' : '#ffffff',
          'line-width': 2.5
        },
      }, refLayer);
    }
  }, []);

  // ── 2. Paint National Overview from In-Memory Cache ──────────────────
  const paintOverviewFromCache = useCallback((dis, hor) => {
    if (!mapLoaded.current || !map.current) return;
    const entry = geoCache.current[dis]?.[String(hor)];
    if (!entry) return;
    const src = map.current.getSource('overview-grid');
    if (src) {
      src.setData(typeof entry.geojson === 'string' ? JSON.parse(entry.geojson) : entry.geojson);
    }
  }, []);

  // ── 3. Viewport Detail Fetcher (Zoom >= 6.8) ─────────────────────────
  const fetchViewport = useCallback((dis, hor) => {
    if (!map.current || !mapLoaded.current) return;
    const zoom = map.current.getZoom();
    if (zoom < 6.5) return; // overview is active

    const bounds = map.current.getBounds();
    // Expand bounds with 0.06 deg buffer to avoid blank borders during panning
    const min_lon = Math.max(65.0, Number((bounds.getWest() - 0.06).toFixed(3)));
    const max_lon = Math.min(100.0, Number((bounds.getEast() + 0.06).toFixed(3)));
    const min_lat = Math.max(6.0, Number((bounds.getSouth() - 0.06).toFixed(3)));
    const max_lat = Math.min(38.0, Number((bounds.getNorth() + 0.06).toFixed(3)));

    const cacheKey = `${dis}_${hor}_${min_lon.toFixed(1)}_${min_lat.toFixed(1)}_${max_lon.toFixed(1)}_${max_lat.toFixed(1)}`;

    if (viewportCache.current.has(cacheKey)) {
      const cached = viewportCache.current.get(cacheKey);
      const detailSrc = map.current.getSource('detail-grid');
      if (detailSrc) detailSrc.setData(cached);
      return;
    }

    // Cancel in-flight request
    if (activeAbortController.current) {
      activeAbortController.current.abort();
    }
    const ac = new AbortController();
    activeAbortController.current = ac;

    const url = `/api/map/viewport?min_lon=${min_lon}&min_lat=${min_lat}&max_lon=${max_lon}&max_lat=${max_lat}&disease=${dis}&horizon=${hor}&limit=12000`;

    fetch(url, { signal: ac.signal })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(data => {
        if (!data) return;
        viewportCache.current.set(cacheKey, data);
        if (viewportCache.current.size > 60) {
          const firstKey = viewportCache.current.keys().next().value;
          viewportCache.current.delete(firstKey);
        }
        const detailSrc = map.current.getSource('detail-grid');
        if (detailSrc) detailSrc.setData(data);
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.warn('[MapViewport] fetch error:', err);
        }
      });
  }, []);

  // Debounced moveend handler
  const handleMapMovement = useCallback(() => {
    if (!map.current) return;
    const z = map.current.getZoom();
    setCurrentZoom(z);

    clearTimeout(viewportDebounce.current);
    viewportDebounce.current = setTimeout(() => {
      fetchViewport(disease, horizon);
    }, 160);
  }, [disease, horizon, fetchViewport]);

  // ── 4. Load National Overview (IDB-first, then network) ──────────────
  const prefetchDisease = useCallback(async (dis, retryDelay = 5000) => {
    if (fetchingRef.current.has(dis)) return;
    if (geoCache.current[dis]) {
      paintOverviewFromCache(dis, horizon);
      fetchViewport(dis, horizon);
      return;
    }

    fetchingRef.current.add(dis);

    // Try IndexedDB first
    if (dataVersion) {
      const cached = await idbGetGeo(dis, dataVersion);
      if (cached) {
        geoCache.current[dis] = cached;
        fetchingRef.current.delete(dis);
        setMapStatus('');
        paintOverviewFromCache(dis, horizon);
        fetchViewport(dis, horizon);
        console.log(`[GeoCache] IndexedDB hit for ${dis} v${dataVersion}`);
        return;
      }
    }

    if (dis === disease) setMapStatus('Loading India risk overview...');

    fetch(`/api/geojson/all?disease=${dis}`)
      .then(async r => {
        if (r.status === 503) {
          const after = parseInt(r.headers.get('Retry-After') || '5', 10) * 1000;
          const delay = Math.max(retryDelay, after);
          fetchingRef.current.delete(dis);
          retryTimers.current[dis] = setTimeout(() => {
            prefetchDisease(dis, Math.min(delay * 1.5, 20000));
          }, delay);
          return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(payload => {
        if (!payload) return;
        const horizons = payload.horizons;
        geoCache.current[dis] = horizons;
        fetchingRef.current.delete(dis);
        if (dis === disease) setMapStatus('');
        paintOverviewFromCache(dis, horizon);
        fetchViewport(dis, horizon);

        if (dataVersion) {
          idbSetGeo(dis, dataVersion, horizons);
        }
      })
      .catch(err => {
        console.error('[GeoCache] Fetch error:', err);
        fetchingRef.current.delete(dis);
        if (dis === disease) setMapStatus('Connecting to server…');
        retryTimers.current[dis] = setTimeout(() => prefetchDisease(dis, 6000), 4000);
      });
  }, [horizon, paintOverviewFromCache, fetchViewport, dataVersion, disease]);

  // ── 5. Initialize MapLibre Once ──────────────────────────────────────
  useEffect(() => {
    if (map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: UNIFIED_STYLE,
      center: [78.9629, 21.0],
      zoom: 4.2,
      pitch: 35,
      attributionControl: false,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    map.current.on('load', () => {
      // Set initial base tile visibility based on theme
      const isLight = theme === 'light';
      map.current.setLayoutProperty('esri-dark-tiles', 'visibility', isLight ? 'none' : 'visible');
      map.current.setLayoutProperty('esri-dark-ref-tiles', 'visibility', isLight ? 'none' : 'visible');
      map.current.setLayoutProperty('esri-light-tiles', 'visibility', isLight ? 'visible' : 'none');
      map.current.setLayoutProperty('esri-light-ref-tiles', 'visibility', isLight ? 'visible' : 'none');

      setupLayers();

      const hoverPopup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
        offset: 12,
      });
      hoverPopupRef.current = hoverPopup;

      // Hover on Overview or Detailed cells
      const handleHover = (e) => {
        if (e.features && e.features.length > 0) {
          map.current.getCanvas().style.cursor = 'pointer';
          const p = e.features[0].properties;
          const rawPct = p.risk_percent !== undefined
            ? parseFloat(p.risk_percent)
            : (p.risk_score || 0) * 99.9;
          const pct = Math.min(99.9, Math.max(0.1, rawPct)).toFixed(1);
          const isAgg = p.is_aggregate === true || p.is_aggregate === 'true';

          // Look up real-world locality name from shared registry or currently selected cell
          const loc = getLocation(p.h3_index);
          const isSelected = selectedCellRef.current?.h3_index === p.h3_index;

          let displayName = loc?.locationName;
          if (!displayName && isSelected && selectedCellRef.current?.locationName) {
            displayName = selectedCellRef.current.locationName;
          }
          if (!displayName) {
            displayName = p.location_name || p.district || 'Area';
          }

          // Format clean subtitle with district and state
          let displaySub = '';
          const dist = loc?.district || p.district;
          const st = loc?.state || p.state;
          if (dist && dist.toLowerCase() !== displayName.toLowerCase()) {
            displaySub = `${dist}${st ? ', ' + st : ''}`;
          } else if (st) {
            displaySub = st;
          }
          displaySub = displaySub.replace(/,\s*$/, '').trim();

          hoverPopup
            .setLngLat(e.lngLat)
            .setHTML(createPopupHTML({
              displayName,
              displaySub,
              pct,
              isAgg,
              h3_index: p.h3_index,
              cell_count: p.cell_count
            }))
            .addTo(map.current);

          // If detailed cell and locality is not yet resolved, dynamically resolve directly on hover!
          if (!isAgg && p.h3_index && (!loc || !loc.isResolved)) {
            clearTimeout(hoverResolveTimer.current);
            hoveredCellId.current = p.h3_index;
            hoverResolveTimer.current = setTimeout(async () => {
              if (hoveredCellId.current !== p.h3_index) return;
              const resolved = await resolveCellLocation(p.h3_index, p.center_lat, p.center_lon, p.district, p.state);
              if (resolved && hoveredCellId.current === p.h3_index && hoverPopup.isOpen()) {
                const updatedName = resolved.locationName || p.district || 'Area';
                let updatedSub = '';
                if (resolved.district && resolved.district.toLowerCase() !== updatedName.toLowerCase()) {
                  updatedSub = `${resolved.district}${resolved.state ? ', ' + resolved.state : ''}`;
                } else if (resolved.state) {
                  updatedSub = resolved.state;
                }
                updatedSub = updatedSub.replace(/,\s*$/, '').trim();

                hoverPopup.setHTML(createPopupHTML({
                  displayName: updatedName,
                  displaySub: updatedSub,
                  pct,
                  isAgg,
                  h3_index: p.h3_index,
                  cell_count: p.cell_count
                }));
              }
            }, 180);
          }
        }
      };

      const handleMouseLeave = () => {
        map.current.getCanvas().style.cursor = '';
        clearTimeout(hoverResolveTimer.current);
        hoveredCellId.current = null;
        hoverPopup.remove();
      };

      // Overview layer interactions
      map.current.on('mousemove', 'overview-fill', handleHover);
      map.current.on('mouseleave', 'overview-fill', handleMouseLeave);
      map.current.on('click', 'overview-fill', (e) => {
        // Zoom in to inspect detailed cells
        map.current.flyTo({
          center: e.lngLat,
          zoom: Math.max(8.5, map.current.getZoom() + 2.5),
          essential: true,
          duration: 1200
        });
      });

      // Detailed layer interactions
      map.current.on('mousemove', 'detail-fill', handleHover);
      map.current.on('mouseleave', 'detail-fill', handleMouseLeave);
      map.current.on('click', 'detail-fill', (e) => {
        if (e.features[0]) {
          const props = { ...e.features[0].properties };
          const loc = getLocation(props.h3_index);
          if (loc?.locationName) {
            props.locationName = loc.locationName;
          } else if (selectedCellRef.current?.h3_index === props.h3_index && selectedCellRef.current.locationName) {
            props.locationName = selectedCellRef.current.locationName;
          }
          onCellClick(props);
        }
      });

      // Viewport change listener
      map.current.on('moveend', handleMapMovement);

      mapLoaded.current = true;

      // Prioritize active disease immediately
      prefetchDisease(disease);

      // Preload other diseases in background
      ['dengue', 'malaria', 'syndemic']
        .filter(d => d !== disease)
        .forEach((d, idx) => {
          setTimeout(() => prefetchDisease(d), (idx + 1) * 2000);
        });
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── 6. Prune Stale IDB Once Version is Known ─────────────────────────
  useEffect(() => {
    if (!dataVersion) return;
    idbPrune(dataVersion);
  }, [dataVersion]);

  // ── 7. Disease Change ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapLoaded.current) return;
    prefetchDisease(disease);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disease, dataVersion]);

  // ── 8. Horizon Change ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapLoaded.current) return;
    paintOverviewFromCache(disease, horizon);
    fetchViewport(disease, horizon);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horizon]);

  // ── 9. Cell Highlight ────────────────────────────────────────────────
  useEffect(() => {
    if (!mapLoaded.current || !map.current) return;

    if (selectedCell?.h3_index) {
      const boundary = cellToBoundary(selectedCell.h3_index, true);
      map.current.getSource('hex-highlight')?.setData({
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [boundary] },
          properties: {}
        }],
      });

      if (selectedCell.fromSidebar || selectedCell.fromSearch) {
        let lng = 0, lat = 0;
        boundary.forEach(([x, y]) => { lng += x; lat += y; });
        const targetZoom = selectedCell.fromSearch ? 11 : 8.8;
        map.current.flyTo({
          center: [lng / boundary.length, lat / boundary.length],
          zoom: targetZoom,
          essential: true
        });
      }
    } else {
      map.current.getSource('hex-highlight')?.setData({ type: 'FeatureCollection', features: [] });
    }
  }, [selectedCell]);

  // ── 10. Navigation Target from SearchBar ──────────────────────────────
  useEffect(() => {
    if (!mapLoaded.current || !map.current || !navigationTarget) return;

    const { type, center, bounds, zoom } = navigationTarget;

    if (activeRippleMarker.current) {
      activeRippleMarker.current.remove();
      activeRippleMarker.current = null;
    }

    if (bounds && bounds.length === 4) {
      map.current.fitBounds(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        {
          padding: { top: 90, bottom: 80, left: 340, right: 380 },
          maxZoom: type === 'state' ? 7.5 : 10.5,
          duration: 1600,
          essential: true,
        }
      );
    } else if (center && center.length === 2) {
      const targetZoom = zoom || (type === 'cell' ? 12 : (type === 'place' ? 12 : 9.5));
      map.current.flyTo({
        center: center,
        zoom: targetZoom,
        essential: true,
        duration: 1600,
      });
    }

    if (center && center.length === 2) {
      const el = document.createElement('div');
      el.className = 'search-pulse-ring';
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat(center)
        .addTo(map.current);
      activeRippleMarker.current = marker;

      setTimeout(() => {
        if (activeRippleMarker.current === marker) {
          marker.remove();
          activeRippleMarker.current = null;
        }
      }, 3500);
    }
  }, [navigationTarget]);

  // ── 11. Theme Toggle (Instantaneous, Zero-Flash, Zero Data Loss) ──────
  useEffect(() => {
    if (!mapLoaded.current || !map.current) return;
    const isLight = theme === 'light';

    if (map.current.getLayer('esri-dark-tiles')) {
      map.current.setLayoutProperty('esri-dark-tiles', 'visibility', isLight ? 'none' : 'visible');
    }
    if (map.current.getLayer('esri-dark-ref-tiles')) {
      map.current.setLayoutProperty('esri-dark-ref-tiles', 'visibility', isLight ? 'none' : 'visible');
    }
    if (map.current.getLayer('esri-light-tiles')) {
      map.current.setLayoutProperty('esri-light-tiles', 'visibility', isLight ? 'visible' : 'none');
    }
    if (map.current.getLayer('esri-light-ref-tiles')) {
      map.current.setLayoutProperty('esri-light-ref-tiles', 'visibility', isLight ? 'visible' : 'none');
    }
    if (map.current.getLayer('hex-highlight-line')) {
      map.current.setPaintProperty('hex-highlight-line', 'line-color', isLight ? '#0f172a' : '#ffffff');
    }
  }, [theme]);

  return (
    <>
      <div ref={mapContainer} className="map-container" />

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
        <div style={{
          marginTop: '6px',
          fontSize: '9px',
          color: 'rgba(255,255,255,0.6)',
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          borderTop: '1px solid rgba(255,255,255,0.08)',
          paddingTop: '5px'
        }}>
          <span style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            backgroundColor: currentZoom >= 6.8 ? '#38bdf8' : '#a78bfa'
          }}></span>
          <span>{currentZoom >= 6.8 ? 'Detailed H3 Cells (Res-7)' : 'National Overview (100% India)'}</span>
        </div>
      </div>
    </>
  );
};

export default MapComponent;
