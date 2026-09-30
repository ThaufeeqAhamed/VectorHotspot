/**
 * HexWorker — runs off-thread to avoid UI freezes.
 * Receives a raw [h3_index, risk] array for each horizon and converts
 * all of them to MapLibre-ready GeoJSON FeatureCollections in parallel.
 */
import { cellToBoundary } from 'h3-js';

function buildGeoJSON(data) {
  let maxRisk = 0.0001;
  const features = data.map(([h3Index, risk]) => {
    if (risk > maxRisk) maxRisk = risk;
    return {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [cellToBoundary(h3Index, true)]
      },
      properties: { h3_index: h3Index, risk_score: risk }
    };
  });
  return { features, maxRisk };
}

self.onmessage = function (e) {
  const { horizons } = e.data; // { "1": [...], "2": [...], "3": [...], "4": [...] }

  const result = {};
  for (const [h, data] of Object.entries(horizons)) {
    result[h] = buildGeoJSON(data);
  }

  // Post back all 4 ready-to-paint GeoJSON sets at once
  self.postMessage(result);
};
