/**
 * @svrn/anomaly — Zero-Shot Anomaly Detection
 *
 * Five parallel O(1) detection engines, ensemble-voted, zero training data.
 *
 *   import { detector } from '@svrn/anomaly'
 *
 *   const d = detector()
 *   for (const x of stream) {
 *     const result = d.observe(x)
 *     if (result.score > 0.5) console.log(result.explanation)
 *   }
 *
 * Engines:
 *   1. Point      — adaptive Z-score (EWMA), detects single outliers
 *   2. Drift      — ADWIN algorithm, detects gradual distribution shifts
 *   3. Changepoint — Bayesian online CPD, detects abrupt regime changes
 *   4. Contextual  — auto-periodicity + phase Z-score, detects temporal context violations
 *   5. Collective  — LZ complexity on sliding window, detects structural pattern changes
 *
 * Enterprise features:
 *   - snapshot() / restore() for persistence across restarts
 *   - Multi-stream: create one detector per metric, they're independent
 *   - Batch mode: scan(array) returns all anomalies with explanations
 *   - Zero config: works on any numeric stream with no parameters
 *   - Configurable: every engine parameter is tunable for specific workloads
 */

import { createPointEngine }       from './engines/point.js';
import { createDriftEngine }       from './engines/drift.js';
import { createChangepointEngine } from './engines/changepoint.js';
import { createContextualEngine }  from './engines/contextual.js';
import { createCollectiveEngine }  from './engines/collective.js';
import { clamp }                   from './core/math.js';

// Re-export individual engines for advanced usage
export { createPointEngine }       from './engines/point.js';
export { createDriftEngine }       from './engines/drift.js';
export { createChangepointEngine } from './engines/changepoint.js';
export { createContextualEngine }  from './engines/contextual.js';
export { createCollectiveEngine }  from './engines/collective.js';

// ═══════════════════════════════════════════════════════════════════════════════
// Ensemble Weights
// ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_WEIGHTS = {
  point:       0.30,  // high weight — most universal signal
  drift:       0.20,  // moderate — catches gradual shifts
  changepoint: 0.25,  // high — catches regime changes with probability
  contextual:  0.15,  // lower — requires periodicity to be useful
  collective:  0.10,  // lowest — LZ complexity is a supporting signal
};

// ═══════════════════════════════════════════════════════════════════════════════
// Detector Factory
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a zero-config anomaly detector.
 *
 * @param {object} [opts]
 * @param {object} [opts.point]       - point engine options
 * @param {object} [opts.drift]       - drift engine options
 * @param {object} [opts.changepoint] - changepoint engine options
 * @param {object} [opts.contextual]  - contextual engine options
 * @param {object} [opts.collective]  - collective engine options
 * @param {object} [opts.weights]     - custom engine weights (must sum to ~1.0)
 * @param {string[]} [opts.engines]   - subset of engines to use (default: all)
 * @returns {Detector}
 */
export function detector(opts = {}) {
  const weights = { ...DEFAULT_WEIGHTS, ...opts.weights };
  const enabledNames = opts.engines ?? Object.keys(DEFAULT_WEIGHTS);

  // Instantiate enabled engines
  const engines = {};
  if (enabledNames.includes('point'))       engines.point       = createPointEngine(opts.point);
  if (enabledNames.includes('drift'))       engines.drift       = createDriftEngine(opts.drift);
  if (enabledNames.includes('changepoint')) engines.changepoint = createChangepointEngine(opts.changepoint);
  if (enabledNames.includes('contextual'))  engines.contextual  = createContextualEngine(opts.contextual);
  if (enabledNames.includes('collective'))  engines.collective  = createCollectiveEngine(opts.collective);

  // Normalize weights to enabled engines
  const activeWeights = {};
  let totalWeight = 0;
  for (const name of Object.keys(engines)) {
    activeWeights[name] = weights[name] ?? 0;
    totalWeight += activeWeights[name];
  }
  if (totalWeight > 0) {
    for (const name of Object.keys(activeWeights)) {
      activeWeights[name] /= totalWeight;
    }
  }

  let observationCount = 0;

  // ── Core observe function ──────────────────────────────────────────────────

  /**
   * Feed a single observation to all engines.
   *
   * @param {number} x - numeric observation
   * @returns {AnomalyResult}
   */
  function observe(x) {
    if (typeof x !== 'number' || !isFinite(x)) {
      throw new TypeError(`observe() requires a finite number, got: ${x}`);
    }

    observationCount++;
    const engineResults = {};
    const explanations = [];
    let ensembleScore = 0;
    let firingEngines = 0;

    for (const [name, engine] of Object.entries(engines)) {
      const result = engine.observe(x);
      engineResults[name] = result;

      // Weighted contribution to ensemble
      ensembleScore += result.score * activeWeights[name];

      // Collect explanations from engines that fired
      if (result.score > 0.1) {
        firingEngines++;
        const explanation = engine.explain(x, result);
        if (explanation) explanations.push({ engine: name, score: result.score, detail: explanation });
      }
    }

    // Multi-engine agreement bonus: if 3+ engines fire, boost score
    if (firingEngines >= 3) {
      ensembleScore = clamp(ensembleScore * 1.3, 0, 1);
    }

    // Determine anomaly type from highest-scoring engine
    let primaryType = 'none';
    let maxEngineScore = 0;
    for (const [name, result] of Object.entries(engineResults)) {
      if (result.score > maxEngineScore) {
        maxEngineScore = result.score;
        primaryType = name;
      }
    }
    if (ensembleScore < 0.1) primaryType = 'none';

    // Severity classification
    let severity;
    if (ensembleScore >= 0.8) severity = 'critical';
    else if (ensembleScore >= 0.5) severity = 'warning';
    else if (ensembleScore >= 0.2) severity = 'info';
    else severity = 'normal';

    return {
      score: +clamp(ensembleScore, 0, 1).toFixed(4),
      severity,
      type: primaryType,
      engines: engineResults,
      explanations,
      firingEngines,
      observation: observationCount,
      value: x,
    };
  }

  // ── Batch API ──────────────────────────────────────────────────────────────

  /**
   * Scan an array of observations and return all anomalies.
   *
   * @param {number[]} data           - array of numeric observations
   * @param {object}   [scanOpts]
   * @param {number}   [scanOpts.threshold=0.3] - min score to include in results
   * @returns {ScanResult[]}
   */
  function scan(data, scanOpts = {}) {
    const threshold = scanOpts.threshold ?? 0.3;
    const anomalies = [];

    for (let i = 0; i < data.length; i++) {
      const result = observe(data[i]);
      if (result.score >= threshold) {
        anomalies.push({ index: i, ...result });
      }
    }

    return anomalies;
  }

  /**
   * Scan and return a full report with summary statistics.
   *
   * @param {number[]} data
   * @param {object}   [reportOpts]
   * @param {number}   [reportOpts.threshold=0.3]
   * @returns {Report}
   */
  function report(data, reportOpts = {}) {
    const threshold = reportOpts.threshold ?? 0.3;
    const all = [];
    const anomalies = [];
    let maxScore = 0;
    let totalScore = 0;

    for (let i = 0; i < data.length; i++) {
      const result = observe(data[i]);
      all.push(result);
      totalScore += result.score;
      if (result.score > maxScore) maxScore = result.score;
      if (result.score >= threshold) anomalies.push({ index: i, ...result });
    }

    // Aggregate type distribution
    const typeCounts = {};
    for (const a of anomalies) {
      typeCounts[a.type] = (typeCounts[a.type] ?? 0) + 1;
    }

    return {
      totalObservations: data.length,
      anomalyCount: anomalies.length,
      anomalyRate: +(anomalies.length / data.length).toFixed(4),
      maxScore: +maxScore.toFixed(4),
      meanScore: +(totalScore / data.length).toFixed(4),
      typeCounts,
      anomalies,
    };
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /** Reset all engines to initial state */
  function reset() {
    for (const engine of Object.values(engines)) engine.reset();
    observationCount = 0;
  }

  /** Serialize full detector state for persistence */
  function snapshot() {
    const engineSnapshots = {};
    for (const [name, engine] of Object.entries(engines)) {
      engineSnapshots[name] = engine.snapshot();
    }
    return {
      v: 1,
      observationCount,
      engines: engineSnapshots,
      weights: activeWeights,
    };
  }

  /** Restore detector state from a snapshot */
  function restore(s) {
    if (s.v !== 1) throw new Error(`Unknown snapshot version: ${s.v}`);
    observationCount = s.observationCount;
    for (const [name, engine] of Object.entries(engines)) {
      if (s.engines[name]) engine.restore(s.engines[name]);
    }
  }

  return {
    observe,
    scan,
    report,
    reset,
    snapshot,
    restore,
    get observationCount() { return observationCount; },
    get engineNames() { return Object.keys(engines); },
    get weights() { return { ...activeWeights }; },
  };
}

/**
 * @typedef {object} AnomalyResult
 * @property {number}  score          - ensemble score 0–1
 * @property {string}  severity       - 'normal' | 'info' | 'warning' | 'critical'
 * @property {string}  type           - primary anomaly type (engine name or 'none')
 * @property {object}  engines        - per-engine results
 * @property {object[]} explanations  - human-readable explanations from firing engines
 * @property {number}  firingEngines  - count of engines that detected something
 * @property {number}  observation    - sequential observation number
 * @property {number}  value          - the input value
 */

/**
 * @typedef {object} Report
 * @property {number}  totalObservations
 * @property {number}  anomalyCount
 * @property {number}  anomalyRate
 * @property {number}  maxScore
 * @property {number}  meanScore
 * @property {object}  typeCounts
 * @property {object[]} anomalies
 */
