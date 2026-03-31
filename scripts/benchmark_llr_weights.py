#!/usr/bin/env python3
"""Benchmark per-detector TPR/FPR on CATS labeled dataset.

Derives data-driven LLR ensemble weights from per-channel, per-detector
TP/FP rates.  Run once after significant detector changes.

Usage:
    cd "Telemetry Anomaly Detection Systems"
    pip install -e .
    python scripts/benchmark_llr_weights.py

Outputs:
    configs/benchmark_weights.yaml  — weights + per-detector stats
    patches WEIGHTS dict in src/dsremo/detection/detector.py in-place

Algorithm:
    1. Load CATS data.parquet → resample to RESAMPLE_SECONDS
    2. 70/30 time-split (no shuffle — temporal integrity preserved)
    3. Per channel: stream STL + calibration on full series; collect
       detector predictions on test portion only
    4. Aggregate TP/FP/TN/FN across all 17 channels
    5. LLR weight: w_i = log(TPR_i / max(FPR_i, 1e-4)), clamped ≥ 0,
       normalised to sum=1 over benchmarked detectors
    6. Non-benchmarked detectors (ML, matrix_profile, correlation_graph)
       get a conservative weight = 0.5 × min(benchmarked weights)

Expected runtime: ~5-15 min depending on hardware (STL fitting is the
bottleneck; recomputed every RECOMPUTE_EVERY samples per channel).
"""

from __future__ import annotations

import math
import re
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
# Suppress structlog info/debug output during the benchmark run.
# structlog uses the stdlib logging bridge; raising the root logger to
# WARNING silences the channel_calibrated / sigma_refreshed messages
# that would otherwise flood the terminal for 17 × 83K samples.
logging.disable(logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Lazy imports so ImportError surfaces with a clear message ─────────────
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed.  Run: pip install pyyaml")
    sys.exit(1)

try:
    from dsremo.detection.calibration import CalibrationManager
    from dsremo.detection.stl_decomposer import STLDecomposer
    from dsremo.detection.cusum import CUSUMDetector
    from dsremo.detection.ewma import EWMADetector
    from dsremo.detection.statistical import StatisticalDetector
    from dsremo.detection.changepoint import ChangePointDetector
    from dsremo.detection.variance_detector import VarianceDetector
    from dsremo.detection.trend_velocity_detector import TrendVelocityDetector
    from dsremo.features.engine import FeatureEngine
except ImportError as e:
    print(f"ERROR: Cannot import dsremo: {e}")
    print("Run: pip install -e . from the project root")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────
_CATS_PARQUET = ROOT / "Resources" / "data.parquet"

_CATS_CHANNELS: list[str] = [
    "aimp", "amud", "arnd", "asin1", "asin2", "adbr", "adfl",
    "bed1", "bed2", "bfo1", "bfo2", "bso1", "bso2", "bso3",
    "ced1", "cfo1", "cso1",
]

# Detectors that run without ML training — benchmarked directly.
BENCHMARKED = ["cusum", "ewma", "statistical", "changepoint", "variance", "trend_velocity"]

# ML + shape + relationship detectors — assigned conservative weights.
NON_BENCHMARKED = ["lstm", "tcn", "matrix_profile", "correlation_graph"]

TRAIN_FRAC      = 0.70   # first 70% for calibration warm-up
RESAMPLE_SEC    = 60     # downsample from 1 Hz to 1-per-minute
WINDOW_SIZE     = 200    # STL sliding window (samples); ~3.3 h at 60 s

# ── Data loading ───────────────────────────────────────────────────────────

def load_and_resample() -> pd.DataFrame:
    """Load CATS parquet and resample to RESAMPLE_SEC seconds.

    Verified on the actual data.parquet before writing any logic:
    - 5,000,000 rows at 1 Hz
    - Columns: 17 CATS channel names + 'y' (0/1) + 'category'
    - Index: RangeIndex (not DatetimeIndex)
    """
    if not _CATS_PARQUET.exists():
        print(f"ERROR: CATS parquet not found at {_CATS_PARQUET}")
        sys.exit(1)

    print(f"Loading {_CATS_PARQUET} ...")
    df = pd.read_parquet(_CATS_PARQUET)
    print(f"  Raw shape: {df.shape}, columns: {list(df.columns)[:6]} ...")

    # Verify expected columns exist before proceeding
    missing = [c for c in _CATS_CHANNELS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing CATS channels in parquet: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)
    if "y" not in df.columns:
        print("ERROR: 'y' column (anomaly label) not found in parquet")
        sys.exit(1)

    # Build a DatetimeIndex so pd.resample() works.
    # CATS dataset is 1 Hz sequential — create artificial timestamps.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range("2023-01-01", periods=len(df), freq="1s")

    agg: dict[str, str] = {ch: "mean" for ch in _CATS_CHANNELS}
    agg["y"] = "max"   # anomalous if ANY sample in the bucket is anomalous

    df_r = df.resample(f"{RESAMPLE_SEC}s").agg(agg).dropna()
    print(f"  Resampled to {RESAMPLE_SEC}s: {df_r.shape}")
    print(f"  Anomaly rate: {df_r['y'].mean():.2%}")
    return df_r


# ── Per-channel streaming benchmark ───────────────────────────────────────

def _make_timestamps(n: int) -> np.ndarray:
    """Sequential unix-epoch timestamps at RESAMPLE_SEC spacing."""
    return np.arange(n, dtype=np.float64) * RESAMPLE_SEC


def run_channel(
    ch: str,
    values: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    train_n: int,
) -> dict[str, dict]:
    """Stream one channel through full pipeline; return per-detector stats.

    Runs the entire series in chronological order so calibration warm-up
    uses the training portion.  Detector predictions are collected only on
    the test portion (indices >= train_n).

    Args:
        ch:         Channel name (used as key prefix).
        values:     Float64 telemetry values, all rows (train + test).
        timestamps: Float64 epoch seconds, same length as values.
        labels:     Int32 anomaly labels (0/1), same length as values.
        train_n:    Index where test portion begins.

    Returns:
        {detector_name: {tp, fp, tn, fn, tpr, fpr}}
    """
    key = f"CATS:{ch}"

    # Fresh instances per channel so states don't bleed between channels.
    cal_mgr = CalibrationManager()
    # Use larger recompute_every (100 vs default 30) for benchmark speed.
    # Each STL fit takes ~60ms on 200 samples; 100-sample cadence gives 3×
    # speedup with negligible accuracy impact (residuals are smooth between fits).
    stl     = STLDecomposer(recompute_every=100)
    feat_eng = FeatureEngine(window_size=300)
    cusum   = CUSUMDetector()
    ewma    = EWMADetector()
    stat    = StatisticalDetector()
    cp      = ChangePointDetector(lookback=200)
    var     = VarianceDetector()
    tv      = TrendVelocityDetector()

    preds: dict[str, list[int]] = {d: [] for d in BENCHMARKED}
    test_labels: list[int] = []

    n_total = len(values)

    for i in range(n_total):
        v  = float(values[i])
        ts = float(timestamps[i])

        # Sliding window: keep last WINDOW_SIZE points.
        start      = max(0, i + 1 - WINDOW_SIZE)
        win_vals   = values[start : i + 1].astype(np.float64)
        win_ts     = timestamps[start : i + 1].astype(np.float64)

        if len(win_vals) < 2:
            if i >= train_n:
                for d in BENCHMARKED:
                    preds[d].append(0)
                test_labels.append(int(labels[i]))
            continue

        # STL decompose (cached every RECOMPUTE_EVERY=30 samples internally).
        decomp = stl.decompose(key, win_vals, win_ts)
        residuals = decomp.residual
        cur_res   = float(residuals[-1])

        # Calibration state machine (warming_up → calibrated).
        cal = cal_mgr.update(key, cur_res)

        # AR(1) pre-whitened residual for CUSUM / EWMA — called every sample
        # in order so _prev_residual stays consistent.
        w_res = cal.whiten(cur_res)

        # Feature vector for StatisticalDetector (operates on residual space).
        fv = feat_eng.compute(ch, cur_res, ts)

        # Run CUSUM/EWMA on every sample so accumulators warm up even during
        # the training period; predictions are stored only for the test set.
        cr  = cusum.detect(key, w_res, cal)
        er  = ewma.detect(key, w_res, cal)
        sr  = stat.detect(fv, residuals)
        vr  = var.detect(residuals, cal)
        tvr = tv.detect(decomp.trend, cal)

        # ChangePoint is window-based; run it periodically (every 30 samples
        # in test) rather than every sample to reduce ruptures overhead.
        # Assign the last computed cp result to each test sample in between.
        if i == 0:
            _cp_flag = False
        if i % 30 == 0 and len(residuals) >= cp.min_segment_size * 2:
            cpr = cp.detect(residuals, ch)
            # Anomaly: at least one real changepoint AND score ≥ 0.5
            # (changepoint within the last third of the window).
            _cp_flag = cpr.is_anomaly and cpr.score >= 0.5

        if i >= train_n:
            test_labels.append(int(labels[i]))
            preds["cusum"].append(int(cr.is_anomaly))
            preds["ewma"].append(int(er.is_anomaly))
            preds["statistical"].append(int(sr.is_anomaly))
            preds["changepoint"].append(int(_cp_flag))
            preds["variance"].append(int(vr.is_anomaly))
            preds["trend_velocity"].append(int(tvr.is_anomaly))

    # Compute TP / FP / TN / FN per detector.
    y_arr = np.array(test_labels, dtype=np.int32)
    results: dict[str, dict] = {}

    for det in BENCHMARKED:
        p = np.array(preds[det], dtype=np.int32)
        if len(p) != len(y_arr):
            print(f"  !! {ch}/{det}: length mismatch pred={len(p)} label={len(y_arr)}")
            continue
        tp = int(np.sum((p == 1) & (y_arr == 1)))
        fp = int(np.sum((p == 1) & (y_arr == 0)))
        tn = int(np.sum((p == 0) & (y_arr == 0)))
        fn = int(np.sum((p == 0) & (y_arr == 1)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        results[det] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                        "tpr": tpr, "fpr": fpr}

    return results


# ── Weight computation ─────────────────────────────────────────────────────

def compute_llr_weights(
    agg: dict[str, dict],
) -> dict[str, float]:
    """Compute normalised LLR weights from aggregated TP/FP/TN/FN counts.

    Formula: w_i ∝ max(0, log(TPR_i / max(FPR_i, 1e-4)))

    A detector with TPR ≫ FPR gets a high weight.  A detector that fires
    randomly (TPR ≈ FPR) gets w ≈ 0.  A detector with TPR < FPR would
    be harmful — clamped to 0.

    Returns weights normalised to sum=1.0.  If every detector's LLR is 0
    (e.g. all detectors are useless on this data), falls back to equal weights.
    """
    raw: dict[str, float] = {}
    for det, s in agg.items():
        tpr = s["tpr"]
        fpr = max(s["fpr"], 1e-4)
        llr = math.log(max(tpr, 1e-4) / fpr)
        raw[det] = max(llr, 0.0)

    total = sum(raw.values())
    if total < 1e-9:
        n = len(raw)
        return {d: 1.0 / n for d in raw}

    return {d: v / total for d, v in raw.items()}


# ── detector.py patching ───────────────────────────────────────────────────

_WEIGHT_COMMENTS: dict[str, str] = {
    "cusum":            "# primary drift detector (NASA CUSUM standard)",
    "ewma":             "# level-shift detector",
    "statistical":      "# single-point spike detector (z-score)",
    "changepoint":      "# structural break detector (PELT)",
    "isolation_forest": "# multivariate cross-parameter anomalies",
    "variance":         "# variance-spike detector (CATS-type oscillatory signals)",
    "lstm":             "# GRU autoencoder — temporal pattern anomalies (ML)",
    "tcn":              "# TCN — dilated causal convolutions, deeper ML patterns",
    "trend_velocity":   "# STL trend acceleration — onset detection",
    "matrix_profile":   "# Matrix Profile discord — shape anomaly detection",
    "correlation_graph":"# Correlation graph — relationship breakdown",
}

# Canonical weight ordering (matches existing WEIGHTS dict order).
_WEIGHT_ORDER = [
    "cusum", "ewma", "statistical", "changepoint",
    "isolation_forest", "variance", "lstm", "tcn",
    "trend_velocity", "matrix_profile", "correlation_graph",
]


def patch_detector_weights(weights: dict[str, float]) -> bool:
    """Replace WEIGHTS dict in detector.py.  Returns True on success."""
    det_path = ROOT / "src" / "dsremo" / "detection" / "detector.py"
    if not det_path.exists():
        print(f"  WARNING: {det_path} not found — skipping patch")
        return False

    content = det_path.read_text()

    lines = ["WEIGHTS: dict[str, float] = {\n"]
    for det in _WEIGHT_ORDER:
        if det not in weights:
            continue
        w       = weights[det]
        comment = _WEIGHT_COMMENTS.get(det, "")
        pad     = " " * max(1, 22 - len(det))
        lines.append(f'    "{det}":{pad}{w:.4f},   {comment}\n')
    lines.append("}\n")
    new_block = "".join(lines)

    # Regex matches the full WEIGHTS dict block (multi-line).
    pattern = r'WEIGHTS: dict\[str, float\] = \{[^}]+\}\n'
    new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)

    if new_content == content:
        print("  WARNING: WEIGHTS block regex did not match — manual patch needed")
        print("  New WEIGHTS block:")
        print(new_block)
        return False

    det_path.write_text(new_content)
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.monotonic()

    df = load_and_resample()

    channels  = [ch for ch in _CATS_CHANNELS if ch in df.columns]
    y_all     = df["y"].to_numpy(dtype=np.float64)
    ts_all    = _make_timestamps(len(df))
    train_n   = int(len(df) * TRAIN_FRAC)
    test_n    = len(df) - train_n

    print(f"\nDataset: {len(df)} rows at {RESAMPLE_SEC}s interval")
    print(f"Train: {train_n} ({TRAIN_FRAC:.0%}) | Test: {test_n} ({1-TRAIN_FRAC:.0%})")
    print(f"Test anomaly rate: {y_all[train_n:].mean():.2%}")
    print(f"Channels: {len(channels)}")
    print(f"Detectors benchmarked: {BENCHMARKED}")
    print()

    # Aggregate counts across all channels.
    agg: dict[str, dict] = {
        d: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for d in BENCHMARKED
    }

    for idx, ch in enumerate(channels):
        t_ch = time.monotonic()
        print(f"[{idx+1:2d}/{len(channels)}] {ch:<8}", end=" ", flush=True)
        ch_vals = df[ch].to_numpy(dtype=np.float64)

        ch_res = run_channel(ch, ch_vals, ts_all, y_all, train_n)

        for det, stats in ch_res.items():
            for k in ("tp", "fp", "tn", "fn"):
                agg[det][k] += stats[k]

        elapsed = time.monotonic() - t_ch
        # Print per-channel snapshot: best detector TPR/FPR
        best = max(ch_res, key=lambda d: ch_res[d]["tpr"] - ch_res[d]["fpr"])
        bs   = ch_res[best]
        print(f"done ({elapsed:.0f}s)  best={best} TPR={bs['tpr']:.2f} FPR={bs['fpr']:.2f}")

    # Global TPR / FPR from aggregated counts.
    for det in BENCHMARKED:
        s = agg[det]
        s["tpr"] = s["tp"] / max(s["tp"] + s["fn"], 1)
        s["fpr"] = s["fp"] / max(s["fp"] + s["tn"], 1)

    # LLR weights for benchmarked detectors.
    bench_w = compute_llr_weights(agg)

    # Conservative weight for non-benchmarked detectors.
    min_bench = min(bench_w.values())
    conserv   = min_bench * 0.5

    all_weights: dict[str, float] = {}
    for d in _WEIGHT_ORDER:
        if d in bench_w:
            all_weights[d] = bench_w[d]
        elif d == "isolation_forest":
            # IsoForest not benchmarked (multivariate; no CATS multivariate labels)
            all_weights[d] = conserv
        elif d in NON_BENCHMARKED:
            all_weights[d] = conserv

    total = sum(all_weights.values())
    all_weights = {d: w / total for d, w in all_weights.items()}

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-DETECTOR RESULTS  (aggregated across all channels)")
    print("=" * 70)
    header = f"{'Detector':<20} {'TP':>7} {'FP':>7} {'FN':>7} {'TN':>8} {'TPR':>6} {'FPR':>6} {'LLR-w':>8}"
    print(header)
    print("-" * 70)
    for det in BENCHMARKED:
        s  = agg[det]
        w  = bench_w[det]
        fpr_floor = max(s["fpr"], 1e-4)
        llr = math.log(max(s["tpr"], 1e-4) / fpr_floor)
        print(
            f"{det:<20} {s['tp']:>7} {s['fp']:>7} {s['fn']:>7} {s['tn']:>8}"
            f" {s['tpr']:>6.3f} {s['fpr']:>6.3f} {w:>8.4f}"
        )

    print()
    print("Final ensemble weights (all detectors, sum=1.0):")
    for det in _WEIGHT_ORDER:
        if det not in all_weights:
            continue
        w      = all_weights[det]
        marker = " [benchmarked]" if det in bench_w else " [estimated]"
        print(f"  {det:<25} {w:.4f}{marker}")

    total_runtime = time.monotonic() - t0
    print(f"\nTotal runtime: {total_runtime/60:.1f} min")

    # ── Save YAML ─────────────────────────────────────────────────────────
    out_path = ROOT / "configs" / "benchmark_weights.yaml"
    out = {
        "meta": {
            "generated_by": "scripts/benchmark_llr_weights.py",
            "dataset": str(_CATS_PARQUET.name),
            "resample_seconds": RESAMPLE_SEC,
            "train_fraction": TRAIN_FRAC,
            "n_channels": len(channels),
            "n_train": train_n,
            "n_test": test_n,
        },
        "benchmark_stats": {
            det: {
                "tp":  agg[det]["tp"],
                "fp":  agg[det]["fp"],
                "tn":  agg[det]["tn"],
                "fn":  agg[det]["fn"],
                "tpr": round(agg[det]["tpr"], 4),
                "fpr": round(agg[det]["fpr"], 4),
            }
            for det in BENCHMARKED
        },
        "weights": {d: round(w, 4) for d, w in all_weights.items()},
    }
    with open(out_path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved: {out_path}")

    # ── Patch detector.py ─────────────────────────────────────────────────
    if patch_detector_weights(all_weights):
        print("Patched: src/dsremo/detection/detector.py  (WEIGHTS dict updated)")
    else:
        print("Manual patch needed — see WEIGHTS block printed above")

    print("\nNext step: pytest tests/  (verify no regressions)")


if __name__ == "__main__":
    main()
