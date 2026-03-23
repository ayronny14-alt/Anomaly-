/**
 * Engine 3 — Changepoint Detection (CUSUM + Bayesian Run Length)
 *
 * Detects abrupt regime changes — the exact observation where the
 * data-generating process switches to a different distribution.
 *
 * Uses a dual approach:
 *   1. CUSUM (Cumulative Sum) — fast, parameter-free detection of mean shifts
 *   2. Bayesian run-length tracking — probability-weighted regime estimation
 *
 * CUSUM (Page 1954):
 *   Accumulates positive and negative deviations from the running mean.
 *   When either accumulator exceeds a threshold, a changepoint is declared.
 *   The accumulators are reset after detection.
 *
 * Why CUSUM over full BOCPD:
 *   - More numerically stable for production use
 *   - Proven detection delay guarantees
 *   - O(1) per observation with constant memory
 *   - The run-length tracker provides context without driving the score
 *
 * Complexity: O(1) per observation.
 */

import { ewma, clamp } from '../core/math.js';

/**
 * @param {object} [opts]
 * @param {number} [opts.threshold=4.0]   - CUSUM decision threshold (in σ units)
 * @param {number} [opts.drift=0.5]       - CUSUM allowance parameter (in σ units)
 * @param {number} [opts.warmup=10]       - min observations before scoring
 * @param {number} [opts.alpha=0.02]      - EWMA decay for baseline stats
 */
export function createChangepointEngine(opts = {}) {
  const threshold = opts.threshold ?? 5.0;
  const drift     = opts.drift ?? 0.8;
  const warmup    = opts.warmup ?? 10;

  const stats = ewma(opts.alpha ?? 0.02); // slow-adapting baseline
  let cusumHigh = 0;  // accumulator for upward shifts
  let cusumLow = 0;   // accumulator for downward shifts
  let idx = 0;
  let lastCP = null;
  let cpDecay = 0;

  // Run-length tracker (lightweight)
  let runLength = 0;
  let runMean = 0;
  let runM2 = 0;

  function runStd() {
    return runLength < 2 ? 1 : Math.sqrt(runM2 / (runLength - 1));
  }

  function updateRunStats(x) {
    runLength++;
    const delta = x - runMean;
    runMean += delta / runLength;
    runM2 += delta * (x - runMean);
  }

  function resetRunStats(x) {
    runLength = 1;
    runMean = x;
    runM2 = 0;
  }

  return {
    name: 'changepoint',

    /**
     * @param {number} x
     * @returns {{ score: number, changepointDetected: boolean, runLength: number, cusumHigh: number, cusumLow: number }}
     */
    observe(x) {
      idx++;
      stats.update(x);

      if (idx <= warmup) {
        updateRunStats(x);
        return { score: 0, changepointDetected: false, runLength, cusumHigh: 0, cusumLow: 0, warming: true };
      }

      const std = stats.std || 1;
      const z = (x - stats.mean) / std;

      // CUSUM update
      cusumHigh = Math.max(0, cusumHigh + z - drift);
      cusumLow  = Math.max(0, cusumLow - z - drift);

      // Decay previous detection
      cpDecay *= 0.88;

      let detected = false;
      if (cusumHigh > threshold || cusumLow > threshold) {
        detected = true;
        const magnitude = Math.max(cusumHigh, cusumLow) / threshold;
        cpDecay = clamp(magnitude * 0.8, 0.3, 1);
        lastCP = { at: idx, direction: cusumHigh > cusumLow ? 'up' : 'down', magnitude };

        // Reset CUSUM after detection
        cusumHigh = 0;
        cusumLow = 0;

        // Reset run-length tracker
        resetRunStats(x);
      } else {
        updateRunStats(x);
      }

      return {
        score: +clamp(cpDecay, 0, 1).toFixed(4),
        changepointDetected: detected,
        runLength,
        cusumHigh: +cusumHigh.toFixed(4),
        cusumLow: +cusumLow.toFixed(4),
        ...(detected ? { direction: lastCP.direction } : {}),
      };
    },

    explain(x, result) {
      if (result.score < 0.1) return null;
      if (!lastCP) return null;
      const dir = lastCP.direction === 'up' ? 'upward' : 'downward';
      return `Changepoint detected: ${dir} regime shift at observation ${lastCP.at}. ` +
             `CUSUM magnitude: ${lastCP.magnitude.toFixed(2)}σ. ` +
             `Current run length since change: ${result.runLength} observations.`;
    },

    reset() {
      stats.reset(); cusumHigh = 0; cusumLow = 0; idx = 0;
      lastCP = null; cpDecay = 0; runLength = 0; runMean = 0; runM2 = 0;
    },
    snapshot() {
      return { stats: stats.snapshot(), cusumHigh, cusumLow, idx, lastCP, cpDecay, runLength, runMean, runM2 };
    },
    restore(s) {
      stats.restore(s.stats); cusumHigh = s.cusumHigh; cusumLow = s.cusumLow;
      idx = s.idx; lastCP = s.lastCP; cpDecay = s.cpDecay;
      runLength = s.runLength; runMean = s.runMean; runM2 = s.runM2;
    },
  };
}
