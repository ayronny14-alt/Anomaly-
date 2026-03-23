# @svrn/anomaly

Zero-shot anomaly detection. Five parallel engines, ensemble-voted, zero training data, zero config.

```js
import { detector } from '@svrn/anomaly'

const d = detector()

for (const x of stream) {
  const { score, severity, type, explanations } = d.observe(x)
  if (severity === 'critical') alert(explanations[0].detail)
}
```

## Why this exists

Every anomaly detection system today requires training data. Datadog needs weeks of baseline. Custom ML models need labeled datasets. Threshold alerts miss gradual drift entirely.

This library detects anomalies on the first observation. No training. No baseline period. No infrastructure.

## Engines

Five detection engines run in parallel on every observation. Each targets a different anomaly class:

| Engine | Detects | Method | Complexity |
|--------|---------|--------|------------|
| **Point** | Single outliers + variance collapse | Adaptive Z-score (EWMA) + dual-rate variance ratio | O(1) |
| **Drift** | Gradual distribution shifts | ADWIN (Adaptive Windowing) | O(log W) |
| **Changepoint** | Abrupt regime changes | CUSUM with run-length tracking | O(1) |
| **Contextual** | Wrong value for the time | Auto-periodicity discovery + phase Z-score | O(1)* |
| **Collective** | Structural pattern changes | Lempel-Ziv complexity on sliding window | O(W log W) |

\* O(maxPeriod) at reassessment intervals

An ensemble voter combines all five with confidence-weighted scoring. An anomaly that triggers one engine is noise. Three engines is real.

## Anomaly types

```
POINT         — "this value is unusual"
DRIFT         — "the distribution is slowly shifting"
CHANGEPOINT   — "the system just switched to a different regime"
CONTEXTUAL    — "this value is normal globally but wrong for this time"
COLLECTIVE    — "these values are individually normal but the pattern changed"
```

## API

### Streaming

```js
const d = detector()

// Feed observations one at a time
const result = d.observe(42.5)
// {
//   score: 0.0000,        — ensemble score 0–1
//   severity: 'normal',   — 'normal' | 'info' | 'warning' | 'critical'
//   type: 'none',         — primary anomaly type
//   engines: { ... },     — per-engine results
//   explanations: [],     — human-readable explanations
//   firingEngines: 0,     — count of triggered engines
//   observation: 1,       — sequential counter
//   value: 42.5           — the input
// }
```

### Batch

```js
// Scan an array, return only anomalies
const anomalies = d.scan(data, { threshold: 0.3 })

// Full report with summary statistics
const report = d.report(data)
// {
//   totalObservations: 1000,
//   anomalyCount: 7,
//   anomalyRate: 0.007,
//   maxScore: 0.8432,
//   meanScore: 0.0341,
//   typeCounts: { point: 3, changepoint: 2, drift: 2 },
//   anomalies: [...]
// }
```

### Persistence

```js
// Save state (survives process restarts)
const snapshot = JSON.stringify(d.snapshot())
fs.writeFileSync('detector-state.json', snapshot)

// Restore in new process
const d2 = detector()
d2.restore(JSON.parse(fs.readFileSync('detector-state.json')))
```

### Configuration

```js
// Zero-config (recommended)
const d = detector()

// Custom engine weights
const d = detector({
  weights: { point: 0.4, drift: 0.2, changepoint: 0.2, contextual: 0.1, collective: 0.1 }
})

// Use only specific engines
const d = detector({ engines: ['point', 'drift'] })

// Tune individual engines
const d = detector({
  point: { alpha: 0.03, warmup: 20 },
  drift: { delta: 0.005, maxWindow: 2000 },
  changepoint: { threshold: 6.0 },
  contextual: { maxPeriod: 168 },  // 1 week in hours
  collective: { windowSize: 100 }
})
```

### Individual engines

```js
import { createPointEngine } from '@svrn/anomaly/engines/point'

const engine = createPointEngine({ alpha: 0.05 })
const result = engine.observe(42.5)
```

## Performance

Benchmarked on a single CPU core:

| Observations | Time | Throughput |
|-------------|------|------------|
| 10,000 | 143ms | 70K obs/sec |
| 100,000 | 1.5s | 62K obs/sec |

Memory usage is constant regardless of stream length — all engines use O(1) or O(W) bounded buffers.

## Validated scenarios

Tested against real-world failure modes at scale:

- **Sensor drift** — gradual radar calibration loss over 10,000 readings
- **Stuck sensor** — variance collapse detection (reading frozen at one value)
- **Bot engagement** — sawtooth manipulation patterns vs organic viral curves
- **Shadowban detection** — sudden engagement cliff
- **Bearing failure** — vibration signal going from chaotic to periodic
- **Fuel pressure leak** — slow drift during engine burn
- **Terminal degradation** — per-device packet loss increase across fleet
- **Multi-stream** — 100 independent detectors running simultaneously
- **p99 latency** — tail degradation hidden in healthy median
- **Connection pool leak** — gradual resource exhaustion
- **Revenue anomaly** — pricing bug detection from daily revenue stream
- **Process restart** — deterministic snapshot/restore across restarts

## Install

```
npm install @svrn/anomaly
```

## License

MIT
