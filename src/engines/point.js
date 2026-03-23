/**
 * Engine 1 — Point Anomaly Detection
 *
 * Detects two classes of anomaly:
 *   1. Value anomalies: individual observations far from the mean
 *   2. Variance anomalies: sudden changes in the volatility itself
 *      (e.g., sensor stuck, signal going from chaotic to periodic)
 *
 * Uses dual adaptive Z-scores with EWMA:
 *   - Primary: tracks value distribution (mean, std)
 *   - Meta: tracks variance distribution (variance of squared deviations)
 *
 * Complexity: O(1) time, O(1) memory per observation.
 */

import { ewma, clamp } from '../core/math.js';

/**
 * @param {object} [opts]
 * @param {number} [opts.alpha=0.05]    - EWMA decay factor for value stats
 * @param {number} [opts.warmup=10]     - observations before scoring
 * @param {number} [opts.threshold=3.0] - z-score threshold for flagging
 */
export function createPointEngine(opts = {}) {
  const alpha   = opts.alpha ?? 0.05;
  const warmup  = opts.warmup ?? 10;
  const zThresh = opts.threshold ?? 3.0;

  const stats = ewma(alpha);

  // Meta-level: dual-rate variance tracking
  // Fast rate adapts to current regime, slow rate remembers the old regime
  // The ratio between them detects variance changes
  const metaFast = ewma(0.1);   // adapts in ~10 observations
  const metaSlow = ewma(0.005); // adapts in ~200 observations (long memory)
  let idx = 0;

  return {
    name: 'point',

    /**
     * @param {number} x - new observation
     * @returns {{ score: number, z: number, mean: number, std: number, varianceZ?: number }}
     */
    observe(x) {
      idx++;
      const z = stats.zscore(x);

      // Squared deviation from current mean = instantaneous variance proxy
      const sqDev = (x - stats.mean) ** 2;

      stats.update(x);
      metaFast.update(sqDev);
      metaSlow.update(sqDev);

      // Variance ratio: fast/slow. When variance collapses, fast drops
      // quickly while slow stays high → ratio << 1
      // When variance explodes, fast rises while slow stays low → ratio >> 1
      const slowVar = metaSlow.mean;
      const fastVar = metaFast.mean;
      const varRatio = slowVar > 1e-10 ? fastVar / slowVar : 1;
      // Convert ratio to a z-like score: log scale, centered at 0
      // Require significant history before variance tracking is meaningful
      const varZ = slowVar > 1e-10 && idx > Math.max(warmup * 3, 30)
        ? Math.log(varRatio + 1e-10) / Math.log(3) // log base 3: ratio=3 → varZ=1, ratio=1/3 → varZ=-1
        : 0;

      if (idx <= warmup) {
        return { score: 0, z: 0, mean: stats.mean, std: stats.std, warming: true };
      }

      const absZ = Math.abs(z);
      // Value anomaly: ramp from 0 at |z|=2 to 1 at |z|=5
      const valueScore = absZ <= 2 ? 0 : (absZ - 2) / 3;

      // Variance anomaly: ratio-based detection
      // varZ < -2 = variance collapsed (stuck sensor, regularity increase)
      // varZ > 2  = variance exploded (instability)
      const absVarZ = Math.abs(varZ);
      const varianceScore = absVarZ <= 2 ? 0 : (absVarZ - 2) / 4;

      const score = clamp(Math.max(valueScore, varianceScore * 0.7), 0, 1);

      return {
        score: +score.toFixed(4),
        z: +z.toFixed(4),
        varianceZ: +varZ.toFixed(4),
        mean: +stats.mean.toFixed(6),
        std: +stats.std.toFixed(6),
        direction: z > 0 ? 'high' : z < 0 ? 'low' : 'neutral',
        varianceDirection: varZ < -2 ? 'collapsed' : varZ > 2 ? 'exploded' : 'stable',
        varianceRatio: +(varRatio).toFixed(4),
      };
    },

    explain(x, result) {
      if (result.warming) return 'Warming up — insufficient data for scoring.';
      if (result.score === 0) return null;

      const parts = [];
      if (Math.abs(result.z) > 2) {
        const dir = result.z > 0 ? 'above' : 'below';
        parts.push(`Value ${x} is ${Math.abs(result.z).toFixed(1)}σ ${dir} the adaptive mean (${result.mean}).`);
      }
      if (result.varianceDirection === 'collapsed') {
        parts.push(`Variance collapsed — signal has become suspiciously regular (variance Z: ${result.varianceZ}).`);
      } else if (result.varianceDirection === 'exploded') {
        parts.push(`Variance exploded — signal instability detected (variance Z: ${result.varianceZ}).`);
      }
      return parts.join(' ') || null;
    },

    reset() { stats.reset(); metaFast.reset(); metaSlow.reset(); idx = 0; },
    snapshot() { return { stats: stats.snapshot(), metaFast: metaFast.snapshot(), metaSlow: metaSlow.snapshot(), idx }; },
    restore(s) { stats.restore(s.stats); metaFast.restore(s.metaFast); metaSlow.restore(s.metaSlow); idx = s.idx; },
  };
}
