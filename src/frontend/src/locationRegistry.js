/**
 * locationRegistry.js
 * ====================
 * Shared location name resolver and registry for VectorHotspot.
 *
 * Ensures 100% synchronization between:
 *  1. SearchBar place results (e.g. Mumbra, Kalamboli, Bandra, etc.)
 *  2. MapComponent hover popups
 *  3. AnalyticsPanel (explainability tab)
 *
 * Guarantees that the real-world locality name is displayed consistently
 * across all components, avoiding discrepancies between district names
 * (e.g. Raigad, Thane) and real-world town/locality names.
 */

import { cellToLatLng, gridDisk } from 'h3-js';

const STORAGE_KEY = 'vh_location_registry_v1';

// In-memory cache backed by sessionStorage for persistence across page refreshes
const registry = new Map();

// Initialize from sessionStorage if available
try {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved) {
    const entries = JSON.parse(saved);
    Object.entries(entries).forEach(([k, v]) => registry.set(k, v));
  }
} catch {
  // ignore storage errors
}

function persistRegistry() {
  try {
    const obj = {};
    // Store up to 200 most recent locations
    const entries = Array.from(registry.entries()).slice(-200);
    entries.forEach(([k, v]) => { obj[k] = v; });
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
  } catch {
    // ignore
  }
}

/**
 * Register a known location for an H3 cell.
 *
 * @param {string} h3Index
 * @param {Object} info - { locationName, district, state, fullName, lat, lon }
 */
export function registerLocation(h3Index, info) {
  if (!h3Index || !info) return;
  const existing = registry.get(h3Index) || {};
  const updated = {
    ...existing,
    ...info,
    locationName: info.locationName || info.name || existing.locationName || info.district,
    district: info.district || existing.district,
    state: info.state || existing.state,
    fullName: info.fullName || existing.fullName,
    lat: info.lat ?? existing.lat,
    lon: info.lon ?? existing.lon,
    updatedAt: Date.now()
  };
  registry.set(h3Index, updated);
  persistRegistry();
  return updated;
}

/**
 * Get registered location info for an H3 cell (synchronous, instant).
 */
export function getLocation(h3Index) {
  if (!h3Index) return null;
  return registry.get(h3Index) || null;
}

/**
 * Format a unified display title and subtitle for any cell.
 *
 * @param {string} h3Index
 * @param {string} fallbackDistrict
 * @param {string} fallbackState
 * @returns {{ title: string, subtitle: string, isRealName: boolean }}
 */
export function getCellLocationDisplay(h3Index, fallbackDistrict = '', fallbackState = '') {
  const loc = getLocation(h3Index);
  if (loc && loc.locationName) {
    const title = loc.locationName;
    const subParts = [];
    if (loc.district && loc.district.toLowerCase() !== title.toLowerCase()) {
      subParts.push(loc.district);
    }
    if (loc.state) {
      subParts.push(loc.state);
    } else if (fallbackState) {
      subParts.push(fallbackState);
    }
    return {
      title,
      subtitle: subParts.join(', '),
      isRealName: true
    };
  }

  // Fallback to district and state
  const title = fallbackDistrict || 'Area';
  const subtitle = fallbackState || '';
  return {
    title,
    subtitle,
    isRealName: false
  };
}

/**
 * Asynchronously resolve real-world locality for an H3 cell using reverse geocoding.
 * Caches result so subsequent hovers and clicks are O(1) instant.
 */
export async function resolveCellLocation(h3Index, lat, lon, fallbackDistrict = '', fallbackState = '') {
  if (!h3Index) return null;

  // Check cache first
  const existing = registry.get(h3Index);
  if (existing && existing.isResolved) {
    return existing;
  }

  if (!lat || !lon) {
    try {
      const [cLat, cLon] = cellToLatLng(h3Index);
      lat = cLat;
      lon = cLon;
    } catch {
      // ignore
    }
  }

  if (!lat || !lon) {
    return registerLocation(h3Index, {
      locationName: fallbackDistrict,
      district: fallbackDistrict,
      state: fallbackState,
      isResolved: false
    });
  }

  try {
    const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&addressdetails=1`;
    const res = await fetch(url, {
      headers: {
        'Accept-Language': 'en',
        'User-Agent': 'VectorHotspot-Dashboard/1.0'
      }
    });
    if (res.ok) {
      const data = await res.json();
      const addr = data.address || {};
      
      // Extract specific locality/town/suburb
      const locality = addr.suburb || addr.neighbourhood || addr.residential ||
                       addr.village || addr.town || addr.city_district ||
                       addr.city || data.name || fallbackDistrict;

      const district = addr.state_district || addr.county || addr.city || fallbackDistrict;
      const state = addr.state || fallbackState;

      const registered = registerLocation(h3Index, {
        locationName: locality,
        district: district,
        state: state,
        fullName: data.display_name,
        lat: lat,
        lon: lon,
        isResolved: true
      });

      // Also propagate locality name to immediate neighbors so nearby hovers resolve instantly
      if (locality && locality.toLowerCase() !== (fallbackDistrict || '').toLowerCase()) {
        try {
          const neighbors = gridDisk(h3Index, 1);
          neighbors.forEach(nh => {
            if (nh !== h3Index && (!registry.has(nh) || !registry.get(nh).isResolved)) {
              registerLocation(nh, {
                locationName: locality,
                district: district,
                state: state,
                fullName: data.display_name,
                isResolved: true
              });
            }
          });
        } catch {
          // ignore
        }
      }

      return registered;
    }
  } catch {
    // Network or rate-limit fail; use fallback
  }

  return registerLocation(h3Index, {
    locationName: fallbackDistrict,
    district: fallbackDistrict,
    state: fallbackState,
    isResolved: false
  });
}
