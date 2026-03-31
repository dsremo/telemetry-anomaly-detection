# Dsremo — Detection Principles

Non-negotiable rules. Read before touching any detector, feature engine, or config.

---

## Pipeline

- **Order**: Raw → STL Decompose → Residuals → CUSUM/EWMA/PELT/BOCPD/IsoForest → Ensemble → Alert
- **Detectors run on STL residuals only. Never on raw telemetry values.**
- STL removes the predictable orbital sinusoid. Anomalies live in the residual.
- Until enough data for STL (< 2× orbital period): fallback to rolling z-score on raw values.

---

## Orbital Seasonality

- LEO orbital period = **5400 seconds** (90 min). This is the `period` parameter for STL.
- At 1 Hz: 5400 samples = 1 orbital cycle. STL needs ≥ 2 cycles = ≥ 10,800 samples minimum.
- Thermal, power, and comms all exhibit sinusoidal variation driven by eclipse entry/exit.
- **Without STL: eclipse transitions look like anomalies. This is the primary source of false positives.**
- For downsampled historical data (e.g., 1 sample/day): orbital seasonality is not visible → STL period must be scaled to the actual sampling interval.

---

## Detectors and Their Roles

| Detector | Catches | Misses |
|----------|---------|--------|
| CUSUM on residuals | Slow drift, gradual degradation, accumulating deviation | Spikes |
| EWMA-STR on residuals | Level shifts, step changes | Very slow drift |
| Z-score on residuals | Spikes, sudden outliers | Slow drift (memoryless) |
| PELT changepoint | Abrupt structural breaks in mean/variance | Gradual trends |
| Isolation Forest | Cross-parameter multivariate anomalies | Univariate drift |
| BOCPD (Bayesian Online Changepoint) | Structural regime changes with calibrated probability (Adams & MacKay 2007) | Gradual drift without regime shift |

- **Run all applicable detectors in parallel. No single detector is sufficient.**
- Z-score alone = memoryless. It will miss degradation because the rolling mean follows the drift.
- CUSUM alone = accumulates noise into false positives without STL residuals as input.
- Isolation Forest = only when ≥ 2 standard parameter names are present AND ≥ 200 training samples.

---

## CUSUM — Parameters

CUSUM is the primary drift detector. Per-channel calibration is mandatory.

```
Reference period: first 100 samples per channel → compute μ_ref, σ_ref

Allowance:   k = 0.5 × σ_ref       (sensitivity: smaller k = more sensitive)
Threshold:   H = 5.0 × σ_ref       (alarm threshold: smaller H = more false positives)

S_pos[t] = max(0, S_pos[t-1] + (residual[t] - k))
S_neg[t] = max(0, S_neg[t-1] + (-residual[t] - k))
Alarm when S_pos > H or S_neg > H
Reset S after alarm fires.
```

- k and H are **per-channel** values derived from σ_ref. Global values are wrong.
- CUSUM state is **per (satellite_id, parameter)**. Never shared across satellites.
- If μ_ref drifts by > 10 × σ_ref: recalibrate reference (new operational regime).

---

## EWMA-STR — Parameters

EWMA on STL residuals for level shift detection.

```
λ = 0.2   (smoothing factor; lower = more memory, better for slow shifts)

Z_ewma[t] = λ × residual[t] + (1 - λ) × Z_ewma[t-1]
UCL = +3 × σ_ref × sqrt(λ / (2 - λ))
LCL = -3 × σ_ref × sqrt(λ / (2 - λ))
Alarm when Z_ewma > UCL or Z_ewma < LCL
```

- UCL/LCL are **per-channel** using σ_ref from the calibration window.
- EWMA state is **per (satellite_id, parameter)**.

---

## Ensemble Voting

- **Severity is ALWAYS derived from final ensemble confidence. Never from an individual detector's raw score.**
- Confidence = normalized by triggered detectors' weights only (not all detector weights).
- Agreement factor scales confidence by how many detectors agree:
  - 1 of N triggered: × 0.60
  - 2 of N triggered: × 0.80
  - 3+ of N triggered: × 1.00
- Severity gates (from dsremo.yaml, never hardcoded):
  - watch ≥ 0.50, warning ≥ 0.65, critical ≥ 0.85
- Below watch threshold: discard, do not store anomaly.

---

## Per-Channel Calibration

- First 100 samples per channel per satellite = **reference window**.
- From reference window: compute μ_ref, σ_ref. These drive CUSUM k/H and EWMA UCL/LCL.
- **No global thresholds for CUSUM or EWMA.** Per-channel only.
- STL requires its own minimum data. Track calibration state: `WARMING_UP → CALIBRATED → RUNNING`.
- If channel goes silent for > 2 orbital periods: reset calibration (sensor may have restarted).

---

## Data Quality Guards

- Rolling std < 1e-4 → **NOMINAL**. Constant sensor = not anomalous.
- Fewer than 50 residual points available → do not run CUSUM/EWMA (cold-start).
- Rate-of-change: |dv/dt| > 3 × σ_roc is a standalone anomaly type (e.g., sudden spike).
- NaN or ±Inf in telemetry: reject at ingest boundary, never let reach detector.

---

## Isolation Forest Rules

- Only run when ≥ 2 parameters with **standard names** are present (`battery_voltage`, etc.).
- Requires ≥ 200 training samples across all parameters. Skip if insufficient.
- Refit every 1000 new samples using recent **normal** data only.
- Feature contributions computed via perturbation (mask to zero, measure delta score).
- Do not run for ESA channel_* naming pattern — fall through to statistical detectors.

---

## Evaluation

- ESA benchmark must be evaluated with **segment-wise F1**, not point-wise.
- Anomaly segment detected = TP if any flagged point falls within ±24h of a labeled anomaly.
- Point-wise precision/recall is misleading for satellite telemetry (sparse labels, long segments).
- Target: F1 ≥ 0.80 on ESA-ADB dataset. ESA-ADB paper baseline (EWMA-STR): 87-91% F1.

---

## Code Invariants

- **Config wired at startup only.** `init_detectors(settings)` is called once in `lifespan()`. Zero hardcoded thresholds in detector logic.
- All detector singletons initialized once, reused across cycles. No per-request instantiation.
- CUSUM and EWMA state stored in module-level dicts keyed by `(satellite_id, parameter)`.
- STL state per channel: recalculate only when ≥ 300 new points have arrived since last decomposition.
- No string interpolation in SQL. Parameterized queries only.
- All detector inputs are numpy arrays. No Python loops in the hot path.
- `structlog` for all logging. No `print()` in production code paths.

---

## What We Are Not

- Not a CCSDS stack. Not a ground station. Not mission control.
- Not a deep learning system (no torch, no neural nets in MVP).
- The product: clean telemetry time-series → early anomaly detection → human-readable explanation.

---

## Source Validation

These rules are derived from:
- ESA-ADB paper: EWMA on STL residuals → 87-91% F1 on ESA OPS-SAT (our benchmark dataset)
- NASA production: CUSUM with k=0.5σ, H=5σ is the workhorse for spacecraft drift detection
- Microsoft Azure Anomaly Detector: Spectral Residual for unsupervised detection (reference only)
- Alibaba Time-CAD: context-aware decomposition before detection (validates STL-first approach)
- Uber GAL-MAD: SHAP values for explainability (Phase 2 target)
