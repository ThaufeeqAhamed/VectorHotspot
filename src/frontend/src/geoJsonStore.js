/**
 * geoJsonStore.js — IndexedDB-backed persistent cache for large GeoJSON payloads.
 *
 * Why IndexedDB?
 *   - sessionStorage/localStorage: 5-10 MB limit → too small for 50k-hexagon GeoJSON.
 *   - IndexedDB: No hard limit (typically 50% of free disk). Survives page reload,
 *     browser restart, F5 — cleared only by the user or our own invalidation logic.
 *
 * Cache key format:
 *   `geojson_v2_{disease}_{year}_W{week}`
 *
 * When the backend data version changes (new week), the key changes and the old
 * entry is skipped automatically — the browser will lazily clean it up via idbPrune().
 *
 * Usage:
 *   const horizons = await idbGetGeo(disease, version);   // null if not cached
 *   await idbSetGeo(disease, version, horiz);              // persist after fetch
 *   await idbPrune(version);                               // drop stale entries
 */

const DB_NAME = 'vh_geojson_v2';
const STORE  = 'geojson';
const DB_VER = 1;

let _db = null;

function openDB() {
  if (_db) return Promise.resolve(_db);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VER);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE)) {
        // keyPath = the `key` field in every stored record
        db.createObjectStore(STORE, { keyPath: 'key' });
      }
    };
    req.onsuccess  = (e) => { _db = e.target.result; resolve(_db); };
    req.onerror    = (e) => reject(e.target.error);
  });
}

/** Returns stored horizons object or null on miss/error. */
export async function idbGetGeo(disease, version) {
  try {
    const db = await openDB();
    return await new Promise((resolve) => {
      const key = `geojson_v3_${disease}_${version}`;
      const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
      req.onsuccess = () => resolve(req.result?.value ?? null);
      req.onerror   = () => resolve(null);
    });
  } catch {
    return null;
  }
}

/** Persists the horizons object to IndexedDB. Fails silently if storage is full. */
export async function idbSetGeo(disease, version, horizons) {
  try {
    const db = await openDB();
    await new Promise((resolve) => {
      const key = `geojson_v3_${disease}_${version}`;
      const tx  = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put({ key, value: horizons, ts: Date.now() });
      tx.oncomplete = () => resolve();
      tx.onerror    = () => resolve(); // storage-full → ignore
    });
  } catch {
    // IndexedDB unavailable (private-browsing in some browsers) → skip silently
  }
}

/**
 * Remove all entries whose key does NOT match the current version.
 * Call once on startup to keep the DB tidy.
 */
export async function idbPrune(version) {
  try {
    const db = await openDB();
    await new Promise((resolve) => {
      const tx    = db.transaction(STORE, 'readwrite');
      const store = tx.objectStore(STORE);
      store.openCursor().onsuccess = (e) => {
        const cursor = e.target.result;
        if (!cursor) return;
        if (!cursor.key.includes(version)) {
          cursor.delete();
        }
        cursor.continue();
      };
      tx.oncomplete = () => resolve();
      tx.onerror    = () => resolve();
    });
  } catch {
    // silent
  }
}
