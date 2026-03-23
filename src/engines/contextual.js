/**
 * Engine 4 — Contextual Anomaly Detection
 *
 * Detects values that are normal in isolation but abnormal given their
 * temporal context. Example: CPU at 80% is normal during business hours,
 * anomalous at 3 AM on a Sunday.
 *
 * Method: Auto-discovers periodicity via autocorrelation, then learns
 * per-phase statistics. A value is contextually anomalous if it deviates
 * from its PHASE's distribution, even if it's within the GLOBAL distribution.
 *
 * Periodicity discovery:
 *   - Compute autocorrelation at lags 2..maxPeriod
 *   - Find the lag with highest AC (peak period)
 *   - If peak AC > threshold, use that period for phase binning
 *   - Recheck periodicity every reassessInterval observations
 *     (the dominant period can change over time)
 *
 * Phase statistics:
 *   - Divide each period into `phaseBins` equal segments
 *   - Maintain EWMA mean/variance per bin
 *   - Score = Z-score within the current bin
 *
 * Complexity: O(maxPeriod) at reassessment intervals, O(1) otherwise.
 */

import { ewma, ringBuffer, autocorrelation, clamp } from '../core/math.js';

/**
 * @param {object} [opts]
 * @param {number} [opts.maxPeriod=168]       - maximum period to search for (default: 168 = 1 week in hours)
 * @param {number} [opts.phaseBins=12]        - number of phase bins per period
 * @param {number} [opts.acThreshold=0.3]     - min autocorrelation to declare periodicity
 * @param {number} [opts.reassessInterval=100] - observations between periodicity rechecks
 * @param {number} [opts.warmup=30]           - min observations before scoring
 */
export function createContextualEngine(opts = {}) {
  const maxPeriod       = opts.maxPeriod ?? 168;
  const phaseBins       = opts.phaseBins ?? 12;
  const acThreshold     = opts.acThreshold ?? 0.3;
  const reassessEvery   = opts.reassessInterval ?? 100;
  const warmup          = opts.warmup ?? 30;
  const alpha           = opts.alpha ?? 0.1;

  // Ring buffer for periodicity analysis
  const history = ringBuffer(maxPeriod * 3); // 3 full periods for robust AC

  // Phase-specific statistics
  let phaseStats = [];
  for (let i = 0; i < phaseBins; i++) phaseStats.push(ewma(alpha));

  // Global fallback stats
  const globalStats = ewma(0.05);

  let period = null;       // discovered period (null if no periodicity)
  let periodAC = 0;        // autocorrelation at discovered period
  let idx = 0;
  let nextReassess = warmup;

  /** Discover the dominant period from the history buffer */
  function discoverPeriod() {
    if (history.length < maxPeriod + 2) return;

    const arr = history.toArray();
    let bestLag = 0, bestAC = 0;

    for (let lag = 2; lag <= Math.min(maxPeriod, Math.floor(arr.length / 2)); lag++) {
      const ac = autocorrelation(arr, lag);
      if (ac > bestAC) {
        bestAC = ac;
        bestLag = lag;
      }
    }

    if (bestAC >= acThreshold) {
      period = bestLag;
      periodAC = bestAC;
    } else {
      period = null;
      periodAC = 0;
    }
  }

  /** Get the current phase bin index */
  function currentPhase() {
    if (period === null) return 0;
    const posInPeriod = idx % period;
    return Math.floor((posInPeriod / period) * phaseBins);
  }

  return {
    name: 'contextual',

    /**
     * @param {number} x
     * @returns {{ score: number, phase: number, period: number|null, periodAC: number, phaseZ: number }}
     */
    observe(x) {
      idx++;
      history.push(x);
      globalStats.update(x);

      // Periodicity reassessment
      if (idx >= nextReassess) {
        discoverPeriod();
        nextReassess = idx + reassessEvery;
      }

      if (idx <= warmup || period === null) {
        // No periodicity or warming up: use global Z-score (weaker signal)
        const z = globalStats.zscore(x);
        globalStats.update(x); // double update is intentional — pre-update z
        return {
          score: 0,
          phase: 0,
          period: null,
          periodAC: 0,
          phaseZ: 0,
          warming: idx <= warmup,
        };
      }

      const phase = currentPhase();
      const pStats = phaseStats[phase];
      const phaseZ = pStats.zscore(x);
      pStats.update(x);

      // Contextual anomaly: unusual for this phase
      const absZ = Math.abs(phaseZ);
      const raw = absZ <= 1.5 ? 0 : (absZ - 1.5) / 3;
      const score = clamp(raw, 0, 1);

      // Bonus: if globally normal but phase-anomalous, boost score
      const globalZ = Math.abs(globalStats.zscore(x));
      const contextBonus = (globalZ < 2 && absZ > 3) ? 0.2 : 0;

      return {
        score: +clamp(score + contextBonus, 0, 1).toFixed(4),
        phase,
        period,
        periodAC: +periodAC.toFixed(4),
        phaseZ: +phaseZ.toFixed(4),
      };
    },

    explain(x, result) {
      if (result.score < 0.1) return null;
      if (result.period === null) return null;
      const dir = result.phaseZ > 0 ? 'above' : 'below';
      return `Value ${x} is ${Math.abs(result.phaseZ).toFixed(1)}σ ${dir} the expected range ` +
             `for phase ${result.phase}/${phaseBins} of period ${result.period}. ` +
             `Globally it may appear normal, but it's unusual for this point in the cycle ` +
             `(periodicity AC: ${result.periodAC}).`;
    },

    reset() {
      history.reset(); globalStats.reset(); idx = 0;
      period = null; periodAC = 0; nextReassess = warmup;
      phaseStats = [];
      for (let i = 0; i < phaseBins; i++) phaseStats.push(ewma(alpha));
    },
    snapshot() {
      return {
        history: history.snapshot(), globalStats: globalStats.snapshot(),
        phaseStats: phaseStats.map(s => s.snapshot()),
        period, periodAC, idx, nextReassess,
      };
    },
    restore(s) {
      history.restore(s.history); globalStats.restore(s.globalStats);
      for (let i = 0; i < phaseBins; i++) phaseStats[i].restore(s.phaseStats[i]);
      period = s.period; periodAC = s.periodAC; idx = s.idx; nextReassess = s.nextReassess;
    },
  };
}
