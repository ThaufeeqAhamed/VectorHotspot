/**
 * apiCache.js — Stale-While-Revalidate cache for VectorHotspot.
 *
 * Strategy:
 *   1. On first call   → fetch from API, store in sessionStorage, return data.
 *   2. On repeat call  → return cached data INSTANTLY, then revalidate in
 *                        background; when fresh data arrives call onUpdate().
 *   3. Cache key       → `vh:<url>` so it never clashes with other sites.
 *   4. TTL             → 10 minutes.  Stale-but-acceptable for an epi dashboard.
 *
 * Usage:
 *   const data = await cachedFetch(url);                    // fire-and-forget
 *   const data = await cachedFetch(url, onUpdate);          // SWR pattern
 */

const CACHE_PREFIX = 'vh:';
const TTL_MS = 10 * 60 * 1000; // 10 minutes

function cacheKey(url) {
  return CACHE_PREFIX + url;
}

function readCache(url) {
  try {
    const raw = sessionStorage.getItem(cacheKey(url));
    if (!raw) return null;
    const { data, ts } = JSON.parse(raw);
    if (Date.now() - ts > TTL_MS) {
      sessionStorage.removeItem(cacheKey(url));
      return null;  // expired
    }
    return data;
  } catch {
    return null;
  }
}

function writeCache(url, data) {
  try {
    sessionStorage.setItem(cacheKey(url), JSON.stringify({ data, ts: Date.now() }));
  } catch {
    // sessionStorage may be full; silently ignore
  }
}

/**
 * Fetch with exponential-backoff retry.
 * Handles ECONNREFUSED (backend not up yet) and 5xx server errors.
 */
async function fetchWithRetry(url, maxAttempts = 5, baseDelay = 2000) {
  let lastErr;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const r = await fetch(url);
      if (r.ok) return r;
      // On 503 / 5xx, retry
      if (r.status >= 500) {
        const after = parseInt(r.headers.get('Retry-After') || '0', 10);
        const delay = after > 0 ? after * 1000 : baseDelay * Math.pow(1.5, attempt);
        lastErr = new Error(`HTTP ${r.status}`);
        await new Promise(res => setTimeout(res, delay));
        continue;
      }
      return r; // 4xx — don't retry
    } catch (err) {
      // Network error (ECONNREFUSED, DNS fail, etc.)
      lastErr = err;
      const delay = baseDelay * Math.pow(1.5, attempt);
      console.warn(`[apiCache] Retry ${attempt + 1}/${maxAttempts} for ${url} in ${Math.round(delay)}ms`);
      await new Promise(res => setTimeout(res, delay));
    }
  }
  throw lastErr;
}

/**
 * Stale-While-Revalidate fetch.
 *
 * @param {string}   url        - Relative API url.
 * @param {Function} [onUpdate] - Called with fresh data when background revalidation finishes.
 * @returns {Promise<any>}      - Resolves with cached data immediately if available,
 *                                otherwise waits for the first network response.
 */
export async function cachedFetch(url, onUpdate = null) {
  const cached = readCache(url);

  if (cached !== null) {
    if (onUpdate) {
      fetchWithRetry(url)
        .then(r => r.json())
        .then(fresh => {
          writeCache(url, fresh);
          onUpdate(fresh);
        })
        .catch(() => {});
    }
    return cached;
  }

  // Nothing in cache → must wait for first response
  const r = await fetchWithRetry(url);
  const data = await r.json();
  writeCache(url, data);
  return data;
}

/**
 * Prime the cache for a URL without blocking.
 * Call this on app start so subsequent calls are instant.
 */
export function primeCache(url) {
  if (readCache(url)) return;   // already warm
  fetch(url)
    .then(r => r.json())
    .then(data => writeCache(url, data))
    .catch(() => {});
}

/**
 * Invalidate all cached entries (e.g. after a live data refresh).
 */
export function clearCache() {
  Object.keys(sessionStorage)
    .filter(k => k.startsWith(CACHE_PREFIX))
    .forEach(k => sessionStorage.removeItem(k));
}
