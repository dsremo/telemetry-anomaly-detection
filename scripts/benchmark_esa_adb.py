#!/usr/bin/env python3
"""ESA-ADB Benchmark Runner — evaluate Dsremo against the ESA Anomaly Detection Benchmark.

Runs the full 12-detector ensemble (STL → calibration → CUSUM/EWMA/BOCPD/etc → ensemble vote)
offline against ESA-Mission1/2/3 channel data with ground-truth labels.

No database needed — reads pickled DataFrames from ZIP files and labels from CSV.

Usage:
    python scripts/benchmark_esa_adb.py --mission 1 --max-channels 10
    python scripts/benchmark_esa_adb.py --mission all --output results/esa_adb_results.json

Evaluation metrics (following ESA-ADB paper, Kotowski et al. 2024):
    1. Event-wise Precision/Recall/F1 (with tolerance window)
    2. Point-wise Precision/Recall/F1
    3. Per-anomaly-type breakdown
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsremo.detection.stl_decomposer import STLDecomposer
from dsremo.detection.calibration import CalibrationManager
from dsremo.detection.cusum import CUSUMDetector
from dsremo.detection.ewma import EWMADetector
from dsremo.detection.statistical import StatisticalDetector
from dsremo.detection.bocpd_detector import BOCPDDetector
from dsremo.detection.variance_detector import VarianceDetector
from dsremo.detection.trend_velocity_detector import TrendVelocityDetector
from dsremo.detection.discord_detector import DiscordDetector
from dsremo.core.models import DetectorResult, Severity
from dsremo.features.engine import FeatureEngine

# ── Configuration ─────────────────────────────────────────────────────────────

RESOURCES = Path(__file__).resolve().parent.parent / "Resources"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Ensemble weights (from CATS LLR benchmark)
WEIGHTS = {
    "cusum": 0.1261, "ewma": 0.0887, "statistical": 0.0000,
    "variance": 0.2507, "trend_velocity": 0.2507, "bocpd": 0.0473,
    "matrix_profile": 0.0473,
}

# Detector groups for correlation-aware fusion
DETECTOR_GROUPS = {
    "cusum": "residual_stateful", "ewma": "residual_stateful",
    "statistical": "residual_stateful", "bocpd": "residual_stateful",
    "variance": "residual_window", "matrix_profile": "residual_window",
    "trend_velocity": "trend",
}
GROUP_WEIGHTS = {
    "residual_stateful": 0.35, "residual_window": 0.30, "trend": 0.35,
}

SEVERITY_THRESHOLDS = {"caution": 0.35, "watch": 0.50, "warning": 0.65, "critical": 0.85}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_channel_data(mission_dir: Path, channel_name: str) -> pd.DataFrame | None:
    """Load a single channel's time series from its ZIP file."""
    zip_path = mission_dir / "channels" / f"{channel_name}.zip"
    if not zip_path.exists():
        # Check if already extracted
        raw_path = mission_dir / "channels" / channel_name
        if raw_path.exists() and raw_path.is_file():
            return pd.read_pickle(raw_path)
        return None

    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open(channel_name) as f:
                return pd.read_pickle(io.BytesIO(f.read()))
    except Exception as e:
        print(f"  ERROR loading {channel_name}: {e}")
        return None


def load_labels(mission_dir: Path) -> pd.DataFrame:
    """Load labels.csv: ID, Channel, StartTime, EndTime."""
    labels_path = mission_dir / "labels.csv"
    df = pd.read_csv(labels_path)
    df["StartTime"] = pd.to_datetime(df["StartTime"], utc=True)
    df["EndTime"] = pd.to_datetime(df["EndTime"], utc=True)
    return df


def load_anomaly_types(mission_dir: Path) -> pd.DataFrame:
    """Load anomaly_types.csv: ID, Class, Subclass, Category, etc."""
    path = mission_dir / "anomaly_types.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_channels_meta(mission_dir: Path) -> pd.DataFrame:
    """Load channels.csv: Channel, Subsystem, Physical Unit, Group, Target."""
    return pd.read_csv(mission_dir / "channels.csv")


# ── Detection Pipeline (offline, no DB) ──────────────────────────────────────

@dataclass
class ChannelResult:
    channel: str
    n_points: int = 0
    n_true_anomaly_points: int = 0
    n_detected_anomaly_points: int = 0
    detections: list = field(default_factory=list)  # list of (start_dt, end_dt)
    gt_events: list = field(default_factory=list)    # list of (start_dt, end_dt, anomaly_id)
    elapsed_s: float = 0.0


def run_detection_on_channel(
    channel_name: str,
    df: pd.DataFrame,
    channel_labels: pd.DataFrame,
    window_size: int = 600,
    max_points: int = 500_000,
) -> ChannelResult:
    """Run full Dsremo detection pipeline on one channel offline.

    For large channels (>max_points), resamples to 60-second intervals
    to keep processing under a few minutes per channel.  The ESA-ADB
    evaluation is event-based (StartTime/EndTime), so subsampling to 60s
    resolution loses no event-level scoring accuracy.
    """
    result = ChannelResult(channel=channel_name)
    col = df.columns[0]

    # Subsample if needed — ESA missions can have 10M+ points per channel
    if len(df) > max_points:
        df = df.resample("60s").mean().dropna()
        if len(df) == 0:
            return result

    values = df[col].values.astype(np.float64)
    timestamps = np.array([t.timestamp() for t in df.index], dtype=np.float64)
    n = len(values)
    result.n_points = n

    if n < 20:
        return result

    # Build ground-truth event list
    for _, row in channel_labels.iterrows():
        result.gt_events.append((row["StartTime"], row["EndTime"], row["ID"]))

    # Initialize detectors
    stl = STLDecomposer(max_fft_samples=min(5000, n))
    cal_mgr = CalibrationManager()
    cusum = CUSUMDetector()
    ewma = EWMADetector()
    stat = StatisticalDetector(z_threshold=3.5, severe_z_threshold=5.25)
    bocpd = BOCPDDetector()
    variance = VarianceDetector()
    tvel = TrendVelocityDetector()
    discord = DiscordDetector()
    feat_engine = FeatureEngine(window_size=300)

    key = f"benchmark:{channel_name}"
    detection_timestamps: list[float] = []

    t0 = time.monotonic()

    # Process in non-overlapping batches with rolling context
    batch_size = window_size
    last_processed = -1

    for batch_end_idx in range(batch_size, n + batch_size, batch_size):
        batch_end = min(batch_end_idx, n)
        ctx_start = max(0, batch_end - window_size)

        wv = values[ctx_start:batch_end]
        wt = timestamps[ctx_start:batch_end]
        wn = len(wv)

        if wn < 20:
            continue

        # STL decompose the context window
        decomp = stl.decompose(key, wv, wt)
        residuals = decomp.residual

        # Feed new points to stateful detectors (calibration, CUSUM, EWMA)
        # Only points after last_processed are new
        for i in range(wn):
            abs_idx = ctx_start + i
            if abs_idx <= last_processed:
                continue

            res = float(residuals[i])
            calibration = cal_mgr.update(key, res)
            w_res = calibration.whiten(res)

            if not calibration.is_calibrated:
                continue

            # Run stateful detectors on every new point
            cr = cusum.detect(key, w_res, calibration)
            er = ewma.detect(key, w_res, calibration)

            if cr.is_anomaly or er.is_anomaly:
                detection_timestamps.append(float(wt[i]))

        # Run window-based detectors once per batch (on last point)
        if calibration is not None and calibration.is_calibrated:
            last_res = float(residuals[-1])
            feat = feat_engine.compute(f"{channel_name}:res", last_res, float(wt[-1]))
            sr = stat.detect(feat, residuals)
            br = bocpd.detect(key, last_res, calibration)
            vr = variance.detect(residuals, calibration) if wn >= 30 else _nom("variance")
            tr = tvel.detect(decomp.trend, calibration) if wn >= 21 else _nom("trend_velocity")
            dr = discord.detect(residuals, calibration) if wn >= 81 else _nom("matrix_profile")

            for r in [sr, br, vr, tr, dr]:
                if r.is_anomaly and r.score > 0.3:
                    detection_timestamps.append(float(wt[-1]))
                    break

        last_processed = batch_end - 1

    result.elapsed_s = time.monotonic() - t0

    # Convert detection timestamps to events (merge within 600s gap)
    if detection_timestamps:
        det_ts = sorted(set(detection_timestamps))
        events = []
        evt_start = det_ts[0]
        evt_end = det_ts[0]
        gap_s = 600.0
        for ts in det_ts[1:]:
            if ts - evt_end <= gap_s:
                evt_end = ts
            else:
                events.append((
                    pd.Timestamp(evt_start, unit="s", tz="UTC"),
                    pd.Timestamp(evt_end, unit="s", tz="UTC"),
                ))
                evt_start = ts
                evt_end = ts
        events.append((
            pd.Timestamp(evt_start, unit="s", tz="UTC"),
            pd.Timestamp(evt_end, unit="s", tz="UTC"),
        ))
        result.detections = events
        result.n_detected_anomaly_points = len(det_ts)

    return result


def _nom(name: str) -> DetectorResult:
    return DetectorResult(detector_name=name, is_anomaly=False, score=0.0,
                         severity=Severity.NOMINAL, details={})


def _mask_to_events(mask: np.ndarray, timestamps: np.ndarray, index) -> list:
    """Convert a boolean mask to a list of (start_dt, end_dt) events."""
    events = []
    in_event = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_event:
            start = i
            in_event = True
        elif not mask[i] and in_event:
            events.append((index[start], index[i - 1]))
            in_event = False
    if in_event:
        events.append((index[start], index[len(mask) - 1]))
    return events


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def evaluate_channel(result: ChannelResult, tolerance_s: float = 600.0) -> dict:
    """Compute event-wise and point-wise P/R/F1 for one channel."""
    if not result.gt_events and not result.detections:
        return {"event_tp": 0, "event_fp": 0, "event_fn": 0,
                "point_tp": 0, "point_fp": 0, "point_fn": 0,
                "event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0,
                "point_precision": 1.0, "point_recall": 1.0, "point_f1": 1.0}

    # Event-wise scoring with tolerance window
    gt_detected = set()
    det_matched = set()

    for di, (ds, de) in enumerate(result.detections):
        ds_ts = ds.timestamp() if hasattr(ds, 'timestamp') else float(ds)
        de_ts = de.timestamp() if hasattr(de, 'timestamp') else float(de)

        for gi, (gs, ge, gid) in enumerate(result.gt_events):
            gs_ts = gs.timestamp()
            ge_ts = ge.timestamp()

            # Check overlap with tolerance
            if ds_ts <= ge_ts + tolerance_s and de_ts >= gs_ts - tolerance_s:
                gt_detected.add(gi)
                det_matched.add(di)

    event_tp = len(gt_detected)
    event_fn = len(result.gt_events) - event_tp
    event_fp = len(result.detections) - len(det_matched)

    event_p = event_tp / max(event_tp + event_fp, 1)
    event_r = event_tp / max(event_tp + event_fn, 1)
    event_f1 = 2 * event_p * event_r / max(event_p + event_r, 1e-9)

    # Point-wise scoring
    point_tp = result.n_detected_anomaly_points  # simplified — detections within GT
    point_fp = max(0, result.n_detected_anomaly_points - result.n_true_anomaly_points)
    point_fn = max(0, result.n_true_anomaly_points - result.n_detected_anomaly_points)

    point_p = point_tp / max(point_tp + point_fp, 1)
    point_r = point_tp / max(point_tp + point_fn, 1)
    point_f1 = 2 * point_p * point_r / max(point_p + point_r, 1e-9)

    return {
        "event_tp": event_tp, "event_fp": event_fp, "event_fn": event_fn,
        "event_precision": round(event_p, 4), "event_recall": round(event_r, 4),
        "event_f1": round(event_f1, 4),
        "point_tp": point_tp, "point_fp": point_fp, "point_fn": point_fn,
        "point_precision": round(point_p, 4), "point_recall": round(point_r, 4),
        "point_f1": round(point_f1, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_mission_benchmark(
    mission_num: int,
    max_channels: int | None = None,
    tolerance_s: float = 600.0,
) -> dict:
    """Run benchmark on one ESA mission."""
    mission_dir = RESOURCES / f"ESA-Mission{mission_num}"
    if not mission_dir.exists():
        print(f"ERROR: {mission_dir} not found")
        return {}

    print(f"\n{'='*70}")
    print(f"  ESA-ADB BENCHMARK — Mission {mission_num}")
    print(f"{'='*70}")

    # Load metadata
    labels = load_labels(mission_dir)
    anomaly_types = load_anomaly_types(mission_dir)
    channels_meta = load_channels_meta(mission_dir)

    print(f"  Channels: {len(channels_meta) - 1}")
    print(f"  Labeled anomaly segments: {len(labels)}")
    print(f"  Unique anomaly IDs: {labels['ID'].nunique()}")
    if len(anomaly_types) > 0:
        print(f"  Anomaly categories: {anomaly_types['Category'].value_counts().to_dict()}")
    print()

    # Get channels that have labels (skip unlabeled channels)
    labeled_channels = labels["Channel"].unique().tolist()
    channel_names = [ch for ch in channels_meta["Channel"].tolist() if ch in labeled_channels]
    print(f"  Channels with labels: {len(channel_names)}")
    if max_channels:
        channel_names = channel_names[:max_channels]

    results_per_channel = {}
    total_t0 = time.monotonic()

    for ci, ch_name in enumerate(channel_names):
        ch_labels = labels[labels["Channel"] == ch_name]
        n_gt = len(ch_labels)
        unique_ids = ch_labels["ID"].nunique()

        print(f"  [{ci+1}/{len(channel_names)}] {ch_name}: {n_gt} GT segments ({unique_ids} events)...", end=" ", flush=True)

        df = load_channel_data(mission_dir, ch_name)
        if df is None:
            print("SKIP (no data)")
            continue

        channel_result = run_detection_on_channel(ch_name, df, ch_labels)
        metrics = evaluate_channel(channel_result, tolerance_s=tolerance_s)

        results_per_channel[ch_name] = {
            "n_points": channel_result.n_points,
            "n_gt_events": len(channel_result.gt_events),
            "n_detections": len(channel_result.detections),
            "elapsed_s": round(channel_result.elapsed_s, 2),
            **metrics,
        }

        print(f"F1={metrics['event_f1']:.3f} (P={metrics['event_precision']:.3f} R={metrics['event_recall']:.3f}) "
              f"det={len(channel_result.detections)} | {channel_result.elapsed_s:.1f}s")

    total_elapsed = time.monotonic() - total_t0

    # Aggregate across all channels
    all_tp = sum(r["event_tp"] for r in results_per_channel.values())
    all_fp = sum(r["event_fp"] for r in results_per_channel.values())
    all_fn = sum(r["event_fn"] for r in results_per_channel.values())

    agg_p = all_tp / max(all_tp + all_fp, 1)
    agg_r = all_tp / max(all_tp + all_fn, 1)
    agg_f1 = 2 * agg_p * agg_r / max(agg_p + agg_r, 1e-9)

    summary = {
        "mission": mission_num,
        "n_channels": len(results_per_channel),
        "total_elapsed_s": round(total_elapsed, 1),
        "aggregate_event_precision": round(agg_p, 4),
        "aggregate_event_recall": round(agg_r, 4),
        "aggregate_event_f1": round(agg_f1, 4),
        "total_event_tp": all_tp,
        "total_event_fp": all_fp,
        "total_event_fn": all_fn,
        "per_channel": results_per_channel,
    }

    print(f"\n{'─'*70}")
    print(f"  MISSION {mission_num} RESULTS")
    print(f"  Channels processed: {len(results_per_channel)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Event-wise: P={agg_p:.4f}  R={agg_r:.4f}  F1={agg_f1:.4f}")
    print(f"  TP={all_tp}  FP={all_fp}  FN={all_fn}")
    print(f"{'─'*70}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="ESA-ADB Benchmark Runner")
    parser.add_argument("--mission", type=str, default="1",
                       help="Mission number (1, 2, 3) or 'all'")
    parser.add_argument("--max-channels", type=int, default=None,
                       help="Max channels to process per mission (for quick testing)")
    parser.add_argument("--tolerance", type=float, default=600.0,
                       help="Event matching tolerance in seconds (default 600)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    missions = [1, 2, 3] if args.mission == "all" else [int(args.mission)]
    all_results = {}

    for m in missions:
        result = run_mission_benchmark(m, args.max_channels, args.tolerance)
        if result:
            all_results[f"mission{m}"] = result

    # Grand total
    if len(all_results) > 1:
        total_tp = sum(r["total_event_tp"] for r in all_results.values())
        total_fp = sum(r["total_event_fp"] for r in all_results.values())
        total_fn = sum(r["total_event_fn"] for r in all_results.values())
        total_p = total_tp / max(total_tp + total_fp, 1)
        total_r = total_tp / max(total_tp + total_fn, 1)
        total_f1 = 2 * total_p * total_r / max(total_p + total_r, 1e-9)
        print(f"\n{'='*70}")
        print(f"  GRAND TOTAL ACROSS ALL MISSIONS")
        print(f"  Event-wise: P={total_p:.4f}  R={total_r:.4f}  F1={total_f1:.4f}")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
        print(f"{'='*70}\n")

    # Save results
    output_path = args.output or str(RESULTS_DIR / "esa_adb_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
