/**
 * Engine 5 — Collective Anomaly Detection
 *
 * Detects when a subsequence of observations is anomalous as a group,
 * even though each individual observation may be normal. Examples:
 *   - A flat line in normally variable data (suspicious regularity)
 *   - A sudden increase in randomness in normally smooth data
 *   - A repeated pattern in normally aperiodic data
 *
 * Method: Lempel-Ziv complexity on a sliding window of discretized
 * symbols. LZ complexity measures the "information rate" of a sequence.
 *
 *   - High LZ complexity → high information rate → chaotic/random
 *   - Low LZ complexity  → low information rate → regular/repetitive
 *
 * The detector tracks baseline LZ complexity and flags windows where
 * complexity deviates significantly from the baseline — either direction.
 *
 * Why LZ complexity:
 *   - Approximates Kolmogorov complexity (the ultimate measure of randomness)
 *   - Computed in O(n log n) for a window of size n
 *   - Sensitive to subtle structural changes that statistical tests miss
 *   - Works on any data — no distributional assumptions
 *
 * Complexity: O(W log W) per window slide, where W = window size.
 * With default W=50, this is ~300 operations.
 */

import { ewma, ringBuffer, lzComplexity, discretize, clamp } from '../core/math.js';

/**
 * @param {object} [opts]
 * @param {number} [opts.windowSize=50]  - sliding window for LZ computation
 * @param {number} [opts.symbols=8]      - number of discrete symbols for quantization
 * @param {number} [opts.warmup=60]      - min observations before scoring
 * @param {number} [opts.stride=5]       - compute LZ every N observations (performance)
 */
export function createCollectiveEngine(opts = {}) {
  const windowSize = opts.windowSize ?? 50;
  const symbols    = opts.symbols ?? 8;
  const warmup     = opts.warmup ?? 55;
  const stride     = opts.stride ?? 3;

  const rawBuf = ringBuffer(windowSize);        // raw values
  const symBuf = ringBuffer(windowSize);         // discretized symbols
  const complexityStats = ewma(0.03);            // baseline LZ complexity
  const valueStats = ewma(0.05);                 // for discretization

  let idx = 0;
  let lastLZ = 0;
  let lastScore = 0;

  return {
    name: 'collective',

    /**
     * @param {number} x
     * @returns {{ score: number, lzComplexity: number, baselineLZ: number, complexityZ: number }}
     */
    observe(x) {
      idx++;
      rawBuf.push(x);
      valueStats.update(x);

      // Discretize into symbols using adaptive binning
      const sym = discretize(x, valueStats.mean, valueStats.std, symbols);
      symBuf.push(sym);

      // Only compute LZ on stride intervals (and when we have enough data)
      if (idx < warmup || idx % stride !== 0) {
        return {
          score: +lastScore.toFixed(4),
          lzComplexity: lastLZ,
          baselineLZ: +complexityStats.mean.toFixed(4),
          complexityZ: 0,
          warming: idx < warmup,
        };
      }

      // Compute LZ complexity of current window
      const window = [];
      for (let i = 0; i < symBuf.length; i++) window.push(symBuf.get(i));
      const lz = lzComplexity(window);
      lastLZ = lz;

      // Normalize by theoretical maximum: n / log2(n)
      const maxLZ = windowSize / Math.max(1, Math.log2(windowSize));
      const normalizedLZ = lz / maxLZ;

      const z = complexityStats.zscore(normalizedLZ);
      complexityStats.update(normalizedLZ);

      if (idx < warmup + stride * 5) {
        // Still building baseline
        return {
          score: 0,
          lzComplexity: lz,
          baselineLZ: +complexityStats.mean.toFixed(4),
          complexityZ: +z.toFixed(4),
          warming: true,
        };
      }

      // Score: deviation from baseline complexity in either direction
      const absZ = Math.abs(z);
      const raw = absZ <= 1.5 ? 0 : (absZ - 1.5) / 2.5;
      lastScore = clamp(raw, 0, 1);

      return {
        score: +lastScore.toFixed(4),
        lzComplexity: lz,
        baselineLZ: +complexityStats.mean.toFixed(4),
        complexityZ: +z.toFixed(4),
        direction: z > 0 ? 'more_chaotic' : z < 0 ? 'more_regular' : 'normal',
      };
    },

    explain(x, result) {
      if (result.score < 0.1) return null;
      const dir = result.complexityZ > 0
        ? 'more chaotic than usual (information rate increased)'
        : 'more regular than usual (suspicious repetition or flatline)';
      return `Collective anomaly: recent subsequence is ${dir}. ` +
             `LZ complexity: ${result.lzComplexity} (baseline: ${result.baselineLZ}). ` +
             `This suggests a structural change in the data pattern, not just a single outlier.`;
    },

    reset() {
      rawBuf.reset(); symBuf.reset(); complexityStats.reset();
      valueStats.reset(); idx = 0; lastLZ = 0; lastScore = 0;
    },
    snapshot() {
      return {
        rawBuf: rawBuf.snapshot(), symBuf: symBuf.snapshot(),
        complexityStats: complexityStats.snapshot(), valueStats: valueStats.snapshot(),
        idx, lastLZ, lastScore,
      };
    },
    restore(s) {
      rawBuf.restore(s.rawBuf); symBuf.restore(s.symBuf);
      complexityStats.restore(s.complexityStats); valueStats.restore(s.valueStats);
      idx = s.idx; lastLZ = s.lastLZ; lastScore = s.lastScore;
    },
  };
}
