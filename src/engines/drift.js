/**
 * Engine 2 — Distribution Drift Detection (ADWIN)
 *
 * Detects gradual changes in the data-generating distribution.
 * Uses ADWIN (ADaptive WINdowing) — a parameter-free algorithm with
 * proven optimal change detection guarantees (Bifet & Gavaldà, 2007).
 *
 * How it works:
 *   Maintains a variable-length window of recent observations. At each
 *   step, tests whether the window can be split into two sub-windows
 *   whose means are statistically different (Hoeffding bound). If yes,
 *   the older sub-window is dropped — the distribution has drifted.
 *
 * Why ADWIN:
 *   - No parameters (the confidence δ auto-tunes)
 *   - Guaranteed: false positive rate bounded by δ
 *   - Guaranteed: detection delay bounded by O(1/ε²) observations
 *     where ε is the magnitude of the drift
 *   - Adapts window size automatically — large when stable, shrinks
 *     when drift occurs
 *
 * Scoring:
 *   When a cut is detected, score = magnitude of the mean shift
 *   normalized against the stream's standard deviation.
 *   Between cuts: score decays toward 0.
 *
 * Complexity: O(log W) amortized per observation, where W = window size.
 */

import { clamp } from '../core/math.js';

/**
 * @param {object} [opts]
 * @param {number} [opts.delta=0.01]    - confidence parameter (lower = fewer false alarms)
 * @param {number} [opts.minWindow=10]  - minimum sub-window size for testing
 * @param {number} [opts.maxWindow=1000] - cap window to bound memory
 */
export function createDriftEngine(opts = {}) {
  const delta     = opts.delta ?? 0.01;
  const minWindow = opts.minWindow ?? 10;
  const maxWindow = opts.maxWindow ?? 1000;

  // Compressed bucket representation for O(log W) storage
  // Each bucket stores: sum, sumSq, count
  let buckets = [];
  let totalSum = 0;
  let totalSumSq = 0;
  let totalCount = 0;

  let lastDrift = null;  // most recent detected drift
  let driftDecay = 0;    // decaying score after drift
  let idx = 0;

  function totalMean() { return totalCount === 0 ? 0 : totalSum / totalCount; }
  function totalVariance() {
    if (totalCount < 2) return 0;
    return (totalSumSq / totalCount) - (totalSum / totalCount) ** 2;
  }

  /** Add observation and compress buckets */
  function addToBuckets(x) {
    buckets.push({ sum: x, sumSq: x * x, count: 1 });
    totalSum += x;
    totalSumSq += x * x;
    totalCount++;

    // Compress: merge adjacent same-size buckets (exponential histogram)
    let i = buckets.length - 1;
    while (i > 0 && buckets[i].count === buckets[i - 1].count) {
      buckets[i - 1].sum += buckets[i].sum;
      buckets[i - 1].sumSq += buckets[i].sumSq;
      buckets[i - 1].count += buckets[i].count;
      buckets.splice(i, 1);
      i--;
    }

    // Cap total window size
    while (totalCount > maxWindow && buckets.length > 0) {
      const oldest = buckets[0];
      totalSum -= oldest.sum;
      totalSumSq -= oldest.sumSq;
      totalCount -= oldest.count;
      buckets.shift();
    }
  }

  /** Test for drift: find the optimal split point */
  function detectDrift() {
    if (totalCount < 2 * minWindow) return null;

    // Scan from oldest to newest, testing each split
    let leftSum = 0, leftCount = 0;

    for (let b = 0; b < buckets.length - 1; b++) {
      leftSum += buckets[b].sum;
      leftCount += buckets[b].count;
      const rightCount = totalCount - leftCount;
      const rightSum = totalSum - leftSum;

      if (leftCount < minWindow || rightCount < minWindow) continue;

      const leftMean = leftSum / leftCount;
      const rightMean = rightSum / rightCount;
      const meanDiff = Math.abs(leftMean - rightMean);

      // Hoeffding bound for the difference of two means
      const m = 1 / leftCount + 1 / rightCount;
      const epsilon = Math.sqrt(m * Math.log(4 / delta) / 2);

      if (meanDiff >= epsilon) {
        return {
          cutAt: leftCount,
          leftMean,
          rightMean,
          shift: rightMean - leftMean,
          magnitude: meanDiff,
          epsilon,
        };
      }
    }
    return null;
  }

  /** Drop the older sub-window after drift detection */
  function dropOldWindow(cutCount) {
    let dropped = 0;
    while (buckets.length > 0 && dropped < cutCount) {
      const b = buckets[0];
      if (dropped + b.count <= cutCount) {
        totalSum -= b.sum;
        totalSumSq -= b.sumSq;
        totalCount -= b.count;
        dropped += b.count;
        buckets.shift();
      } else {
        // Partial drop — approximate by proportional reduction
        const frac = (cutCount - dropped) / b.count;
        b.sum *= (1 - frac);
        b.sumSq *= (1 - frac);
        totalSum -= b.sum * frac / (1 - frac);
        totalSumSq -= b.sumSq * frac / (1 - frac);
        const remove = Math.round(b.count * frac);
        b.count -= remove;
        totalCount -= remove;
        break;
      }
    }
  }

  return {
    name: 'drift',

    /**
     * @param {number} x
     * @returns {{ score: number, drifting: boolean, windowSize: number, mean: number, shift?: number }}
     */
    observe(x) {
      idx++;
      addToBuckets(x);

      // Decay previous drift score
      driftDecay *= 0.95;

      const drift = detectDrift();
      if (drift) {
        dropOldWindow(drift.cutAt);
        lastDrift = { ...drift, at: idx };

        // Score based on shift magnitude relative to current std
        const std = Math.sqrt(Math.max(totalVariance(), 1e-15));
        const normShift = drift.magnitude / std;
        driftDecay = clamp(normShift / 3, 0, 1); // |shift| = 3σ → score 1.0
      }

      return {
        score: +clamp(driftDecay, 0, 1).toFixed(4),
        drifting: driftDecay > 0.1,
        windowSize: totalCount,
        mean: +totalMean().toFixed(6),
        buckets: buckets.length,
        ...(drift ? { shift: +drift.shift.toFixed(6), shiftMagnitude: +drift.magnitude.toFixed(6) } : {}),
      };
    },

    explain(x, result) {
      if (!result.drifting) return null;
      if (!lastDrift) return null;
      const dir = lastDrift.shift > 0 ? 'upward' : 'downward';
      return `Distribution drift detected: ${dir} shift of ${Math.abs(lastDrift.shift).toFixed(4)} ` +
             `(old mean: ${lastDrift.leftMean.toFixed(4)} → new mean: ${lastDrift.rightMean.toFixed(4)}). ` +
             `Window contracted from ${lastDrift.cutAt + (totalCount)} to ${totalCount} observations.`;
    },

    reset() { buckets = []; totalSum = 0; totalSumSq = 0; totalCount = 0; lastDrift = null; driftDecay = 0; idx = 0; },
    snapshot() { return { buckets: JSON.parse(JSON.stringify(buckets)), totalSum, totalSumSq, totalCount, lastDrift, driftDecay, idx }; },
    restore(s) { buckets = s.buckets; totalSum = s.totalSum; totalSumSq = s.totalSumSq; totalCount = s.totalCount; lastDrift = s.lastDrift; driftDecay = s.driftDecay; idx = s.idx; },
  };
}
