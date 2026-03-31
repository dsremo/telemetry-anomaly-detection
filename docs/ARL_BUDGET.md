# CUSUM ARL Budget — Average Run Length Analysis

## Parameters
- k = 0.5σ (allowance factor, from `CUSUM_K_FACTOR`)
- H = 8.0σ (alarm threshold, from `CUSUM_H_FACTOR`)

## ARL Tables (i.i.d. Normal, one-sided CUSUM)

Reference: Hawkins & Olwell (1998), "Cumulative Sum Charts", Table 8.1.

| Shift (δ/σ) | ARL₀ (no shift) | ARL₁ (detect) | Meaning |
|-------------|-----------------|---------------|---------|
| 0.0         | ~125,000        | ∞             | False positive rate: 1/125K samples |
| 0.5         | ~125,000        | ~465           | 465 samples to detect 0.5σ shift |
| 1.0         | ~125,000        | ~11            | 11 samples to detect 1σ shift |
| 2.0         | ~125,000        | ~3             | 3 samples to detect 2σ shift |
| 3.0         | ~125,000        | ~1.4           | Near-instant detection |

## Sampling Rate Impact

| Sampling | ARL₁ at 1σ (11 samples) | Real time |
|----------|-------------------------|-----------|
| 1 Hz     | 11 samples              | 11 seconds |
| 1/60 Hz  | 11 samples              | 11 minutes |
| 1/3600 Hz| 11 samples              | 11 hours |

## AR(1) Pre-whitening Effect

When telemetry has lag-1 autocorrelation ρ, the effective ARL without pre-whitening
is degraded. With pre-whitening (w_t = r_t - ρ·r_{t-1}), the innovation sequence
is approximately i.i.d. and the ARL tables above apply to the whitened residuals.

σ_innov = σ_raw × √(1 - ρ²)

The CUSUM thresholds k and H are computed from σ_innov (not σ_raw) when
pre-whitening is active, preserving the intended ARL budget.

## Configuration

All ARL-affecting parameters are configurable in `configs/dsremo.yaml`:
- `cusum_k_factor`: 0.5 (allowance, default)
- `cusum_h_factor`: 8.0 (threshold, default)
- `calibration_window`: 200 (samples for σ_ref estimation)

Operators can tune H for mission-specific tradeoffs:
- Lower H (e.g. 5.0): faster detection, more false positives
- Higher H (e.g. 10.0): fewer false positives, slower detection
