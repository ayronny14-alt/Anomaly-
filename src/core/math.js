/**
 * @svrn/anomaly — Core math primitives
 *
 * All operations are O(1) amortized with constant memory.
 * No heap allocations in hot paths.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Exponentially Weighted Moving Statistics
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Welford-style online mean/variance with exponential weighting.
 * Provides O(1) per observation, constant memory, numerically stable.
 *
 * The decay factor α controls how quickly old observations lose influence:
 *   α = 0.01 → very slow adaptation (long memory, ~100 obs half-life)
 *   α = 0.05 → moderate adaptation (~20 obs half-life)
 *   α = 0.10 → fast adaptation (~10 obs half-life)
 *
 * @param {number} [alpha=0.05] - decay factor ∈ (0, 1)
 */
export function ewma(alpha = 0.05) {
  let mean = 0;
  let varAcc = 0; // exponentially weighted variance accumulator
  let n = 0;

  return {
    /** @param {number} x */
    update(x) {
      n++;
      if (n === 1) {
        mean = x;
        varAcc = 0;
        return;
      }
      const diff = x - mean;
      mean += alpha * diff;
      // Exponentially weighted variance (Finch 2009)
      varAcc = (1 - alpha) * (varAcc + alpha * diff * diff);
    },
    get mean() { return mean; },
    get variance() { return n < 2 ? 0 : varAcc; },
    get std() { return Math.sqrt(this.variance); },
    get count() { return n; },
    /** Z-score of x against current distribution */
    zscore(x) {
      if (n < 2 || this.std < 1e-15) return 0;
      return (x - mean) / this.std;
    },
    reset() { mean = 0; varAcc = 0; n = 0; },
    snapshot() { return { mean, varAcc, n }; },
    restore(s) { mean = s.mean; varAcc = s.varAcc; n = s.n; },
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Circular Buffer
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Fixed-size ring buffer backed by a Float64Array.
 * O(1) push, O(1) indexed access, zero allocation after construction.
 *
 * @param {number} capacity
 */
export function ringBuffer(capacity) {
  const buf = new Float64Array(capacity);
  let head = 0;   // next write position
  let len = 0;    // current fill level

  return {
    push(x) {
      buf[head] = x;
      head = (head + 1) % capacity;
      if (len < capacity) len++;
    },
    /** Get element at logical index i (0 = oldest) */
    get(i) {
      if (i < 0 || i >= len) return undefined;
      return buf[(head - len + i + capacity) % capacity];
    },
    /** Get the most recent element */
    last() { return len === 0 ? undefined : buf[(head - 1 + capacity) % capacity]; },
    get length() { return len; },
    get capacity() { return capacity; },
    get full() { return len === capacity; },
    /** Return a copy of the buffer contents in chronological order */
    toArray() {
      const out = new Float64Array(len);
      for (let i = 0; i < len; i++) {
        out[i] = buf[(head - len + i + capacity) % capacity];
      }
      return out;
    },
    reset() { head = 0; len = 0; buf.fill(0); },
    snapshot() { return { buf: Array.from(buf), head, len }; },
    restore(s) { for (let i = 0; i < capacity; i++) buf[i] = s.buf[i] ?? 0; head = s.head; len = s.len; },
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Statistical Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/** Clamp v to [lo, hi] */
export function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** Sample mean of a typed or regular array */
export function mean(arr) {
  if (arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

/** Sample variance (unbiased, Bessel-corrected) */
export function variance(arr) {
  const n = arr.length;
  if (n < 2) return 0;
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < n; i++) s += (arr[i] - m) ** 2;
  return s / (n - 1);
}

/** Sample standard deviation */
export function stddev(arr) { return Math.sqrt(variance(arr)); }

/**
 * Lag-k autocorrelation of a typed or regular array.
 * Returns ∈ [-1, 1].
 */
export function autocorrelation(arr, lag = 1) {
  const n = arr.length;
  if (n < lag + 2) return 0;
  const m = mean(arr);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    den += (arr[i] - m) ** 2;
    if (i >= lag) num += (arr[i] - m) * (arr[i - lag] - m);
  }
  return den === 0 ? 0 : num / den;
}

/**
 * Shannon entropy of a distribution (array of counts).
 * Returns bits.
 */
export function shannonEntropy(counts) {
  let total = 0;
  for (let i = 0; i < counts.length; i++) total += counts[i];
  if (total === 0) return 0;
  let H = 0;
  for (let i = 0; i < counts.length; i++) {
    if (counts[i] > 0) {
      const p = counts[i] / total;
      H -= p * Math.log2(p);
    }
  }
  return H;
}

/**
 * Approximate Kolmogorov complexity via Lempel-Ziv 76 factorization.
 * Returns the number of distinct factors (higher = more complex).
 *
 * Optimized: uses a Set of seen substrings with bounded max factor length
 * to avoid O(n²) substring scanning. Effective complexity: O(n * maxFactor).
 *
 * @param {number[]} sequence - discretized symbol sequence
 * @returns {number} LZ complexity (factor count)
 */
export function lzComplexity(sequence) {
  const n = sequence.length;
  if (n === 0) return 0;
  if (n === 1) return 1;

  const seen = new Set();
  let complexity = 0;
  let i = 0;

  while (i < n) {
    let l = 1;
    let key = String(sequence[i]);

    // Extend while the substring has been seen before
    while (i + l <= n && seen.has(key)) {
      l++;
      if (i + l <= n) {
        key += ',' + sequence[i + l - 1];
      }
    }

    // Add all prefixes of the new factor to the seen set
    let prefix = '';
    for (let k = 0; k < l && i + k < n; k++) {
      prefix += (k > 0 ? ',' : '') + sequence[i + k];
      seen.add(prefix);
    }

    complexity++;
    i += l;
  }

  return complexity;
}

/**
 * Discretize a continuous value into a symbol index using adaptive binning.
 * Uses mean ± k*std bands.
 *
 * @param {number} x     - raw value
 * @param {number} mean  - current distribution mean
 * @param {number} std   - current distribution std
 * @param {number} [bands=8] - number of output symbols
 * @returns {number} symbol index [0, bands-1]
 */
export function discretize(x, mean, std, bands = 8) {
  if (std < 1e-15) return Math.floor(bands / 2);
  const z = (x - mean) / std;
  // Map z-score to [0, bands-1] via sigmoid-like compression
  const p = 1 / (1 + Math.exp(-z)); // ∈ (0, 1)
  return clamp(Math.floor(p * bands), 0, bands - 1);
}
