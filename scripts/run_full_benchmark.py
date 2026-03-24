"""Full automated benchmark — detects on ALL datasets, scores, reports.

Zero manual input.  The script:
  1. Auto-discovers every tenant/satellite in the DB.
  2. For each satellite: clears old anomalies, resets detector state,
     runs the full 11-detector ensemble (ML disabled for >500 K rows/ch
     to stay practical, statistical ensemble always on).
  3. Auto-extracts ground-truth windows:
       - CATS  → y-column from Resources/data.parquet
       - SKAB  → downloads label CSV from GitHub
       - Others → no GT available, reports detection stats only
  4. Scores P / R / F1 with ±tolerance event-level matching.
  5. Prints one consolidated report.

Run:
    cd "Telemetry Anomaly Detection Systems"
    python3 scripts/run_full_benchmark.py

No flags needed.  Everything is auto-detected.
"""

from __future__ import annotations

import asyncio
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import asyncpg

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dsremo.core.tenant import set_tenant
from dsremo.eval.auto_scorer import AutoScorer
from dsremo.ingest.bulk_loader import print_detection_report, run_bulk_detection
from dsremo.ingest.pipeline import db_context, phase, print_run_header
from dsremo.ingest.utils import adaptive_cooldown_hours, detect_data_frequency

# ── DB connection (for raw queries bypassing pool) ──────────────────────────
_DB = dict(host="localhost", port=5432, database="sentinel",
           user="sentinel", password="sentinel_dev_only")

# ── Parquet GT source ────────────────────────────────────────────────────────
_CATS_PARQUET = _ROOT / "Resources" / "data.parquet"

# ── SKAB GitHub raw label CSVs ───────────────────────────────────────────────
_SKAB_VALVE2_URLS = [
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve2/1.csv",
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve2/2.csv",
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve2/3.csv",
]

# ── ML threshold — disable GRU+TCN when total detection effort > this ────────
# total_effort = max_rows_per_ch * num_channels
# At ~140 pts/s: 500_000 total pts ≈ 60 min.  Above that → stat-only.
_ML_EFFORT_LIMIT = 500_000

# ── Hard cap per channel — prevents OOM on continuous-fire datasets ───────────
# CATS-2 cfo1 has 5M rows with near-continuous anomaly firing (~1 anomaly/2pts).
# Processing all 5M rows generates ~2M Anomaly objects → OOM.  1M rows (≈11 days
# at 1Hz) is more than enough to characterise P/R for any benchmark dataset.
_MAX_ROWS_PER_CHANNEL = 1_000_000

# NOTE: Scoring tolerances are NO LONGER hardcoded here.
# AutoScorer auto-calibrates window_s from measured detection lead-times and
# derives gap_s from cooldown — zero manual tuning required per dataset.

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_cats_gt() -> list[tuple[datetime, datetime]]:
    """Auto-extract GT windows from CATS parquet y column (vectorized)."""
    import pandas as pd
    df = pd.read_parquet(_CATS_PARQUET, columns=["y"])
    df.index = pd.to_datetime(df.index, utc=True)
    y = df["y"].astype(int)
    # Find rising edges (0→1) and falling edges (1→0) vectorized
    diff = y.diff().fillna(0).astype(int)
    starts = df.index[diff == 1].tolist()
    ends   = df.index[diff == -1].tolist()
    # Handle case where series starts in anomaly
    if y.iloc[0] == 1:
        starts = [df.index[0]] + starts
    # Handle case where series ends in anomaly
    if y.iloc[-1] == 1:
        ends = ends + [df.index[-1]]
    windows = [
        (s.to_pydatetime(), e.to_pydatetime())
        for s, e in zip(starts, ends)
    ]
    return windows


def _download_skab_gt() -> list[tuple[datetime, datetime]]:
    """Download SKAB valve2 label CSVs from GitHub and extract anomaly windows."""
    import pandas as pd
    windows: list[tuple[datetime, datetime]] = []
    for url in _SKAB_VALVE2_URLS:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode()
            df = pd.read_csv(StringIO(raw), sep=";", parse_dates=["datetime"],
                             index_col="datetime")
            df.index = df.index.tz_localize("UTC")
            if "anomaly" not in df.columns:
                # Try alternative column name
                anom_col = [c for c in df.columns if "anom" in c.lower()]
                if not anom_col:
                    continue
                df = df.rename(columns={anom_col[0]: "anomaly"})
            in_anom = False
            start: datetime | None = None
            for ts, row in df.iterrows():
                if row["anomaly"] == 1 and not in_anom:
                    in_anom = True
                    start = ts.to_pydatetime()
                elif row["anomaly"] != 1 and in_anom:
                    windows.append((start, ts.to_pydatetime()))  # type: ignore[arg-type]
                    in_anom = False
            if in_anom and start:
                windows.append((start, df.index[-1].to_pydatetime()))
        except Exception as exc:
            print(f"  [warn] SKAB download failed for {url}: {exc}")
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _all_satellites(tenant_id: str) -> list[tuple[str, list[str], int]]:
    """Return [(satellite_id, [params], max_rows_per_ch)] for a tenant."""
    conn = await asyncpg.connect(**_DB)
    try:
        await conn.execute("SELECT set_config('app.tenant_id',$1,false)", tenant_id)
        rows = await conn.fetch(
            """
            SELECT satellite_id, parameter, COUNT(*) AS cnt
            FROM telemetry
            GROUP BY satellite_id, parameter
            ORDER BY satellite_id, cnt DESC
            """
        )
    finally:
        await conn.close()
    sats: dict[str, tuple[list[str], int]] = {}
    for r in rows:
        sid = r["satellite_id"]
        if sid not in sats:
            sats[sid] = ([], 0)
        params, mx = sats[sid]
        params.append(r["parameter"])
        sats[sid] = (params, max(mx, r["cnt"]))
    return [(sid, v[0], v[1]) for sid, v in sats.items()]


async def _fetch_anomaly_timestamps(tenant_id: str, satellite_id: str) -> list[datetime]:
    conn = await asyncpg.connect(**_DB)
    try:
        await conn.execute("SELECT set_config('app.tenant_id',$1,false)", tenant_id)
        rows = await conn.fetch(
            "SELECT timestamp FROM anomalies WHERE satellite_id=$1 ORDER BY timestamp",
            satellite_id,
        )
        return [r["timestamp"].astimezone(timezone.utc) for r in rows]
    finally:
        await conn.close()


async def _clear_anomalies(tenant_id: str, satellite_id: str) -> int:
    conn = await asyncpg.connect(**_DB)
    try:
        await conn.execute("SELECT set_config('app.tenant_id',$1,false)", tenant_id)
        result = await conn.execute(
            "DELETE FROM anomalies WHERE satellite_id=$1 AND tenant_id=$2",
            satellite_id, tenant_id,
        )
        return int(result.split()[-1])
    finally:
        await conn.close()


async def _clear_detector_state(tenant_id: str, satellite_id: str) -> None:
    conn = await asyncpg.connect(**_DB)
    try:
        await conn.execute("SELECT set_config('app.tenant_id',$1,false)", tenant_id)
        await conn.execute(
            "DELETE FROM detector_state WHERE satellite_id=$1 AND tenant_id=$2",
            satellite_id, tenant_id,
        )
        await conn.execute(
            "DELETE FROM channel_calibration WHERE satellite_id=$1 AND tenant_id=$2",
            satellite_id, tenant_id,
        )
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect cooldown from DB telemetry (no CSV file needed)
# ─────────────────────────────────────────────────────────────────────────────

async def _auto_cooldown_from_db(tenant_id: str, satellite_id: str,
                                  parameter: str) -> float:
    """Sample from start+end of DB timeseries to measure median interval → cooldown.

    Sampling only the FIRST 200 rows is biased toward the ingestion-start
    frequency, which may differ from the steady-state sampling rate.  We
    instead sample 100 rows from the start AND 100 rows from the end so that
    any change in frequency (e.g. re-sampling or data gaps) is represented.
    """
    conn = await asyncpg.connect(**_DB)
    try:
        await conn.execute("SELECT set_config('app.tenant_id',$1,false)", tenant_id)
        rows = await conn.fetch(
            """
            (SELECT timestamp FROM telemetry
             WHERE satellite_id=$1 AND parameter=$2
             ORDER BY timestamp ASC  LIMIT 100)
            UNION ALL
            (SELECT timestamp FROM telemetry
             WHERE satellite_id=$1 AND parameter=$2
             ORDER BY timestamp DESC LIMIT 100)
            ORDER BY timestamp
            """,
            satellite_id, parameter,
        )
        if len(rows) < 2:
            return 1.0
        ts = sorted(r["timestamp"] for r in rows)
        intervals = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)]
        # Filter out large gaps (> 10× the min interval) that come from
        # the start/end join boundary so they don't inflate the median.
        pos = [x for x in intervals if x > 0]
        if not pos:
            return 1.0
        pos.sort()
        min_iv = pos[0]
        pos = [x for x in pos if x <= min_iv * 100]
        median_s = pos[len(pos) // 2]
        return adaptive_cooldown_hours(median_s)
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset catalogue — which tenants/satellites to benchmark
# ─────────────────────────────────────────────────────────────────────────────

# Primary benchmark targets only (skip sprint-variant duplicates)
_TARGETS: list[tuple[str, str | None]] = [
    # (tenant_id, satellite_id or None=all)
    ("opssat",          "OPSSAT-3"),
    ("skab-valve2",     "SKAB-VALVE2"),
    ("cats-spacecraft", "CATS-2"),       # baseline z=3
    ("cats-spacecraft", "CATS-3"),       # tuned z=6
    ("satnogs",         "SATNOGS-25544"),
    ("satnogs",         "SATNOGS-40074"),
    ("satnogs",         "SATNOGS-43017"),
    ("esa-mission1",    "ESA-MISSION1"),
    ("gecco-water",     "GECCO-WATER-S10"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Per-satellite detection run
# ─────────────────────────────────────────────────────────────────────────────

async def run_satellite(
    tenant_id: str,
    satellite_id: str,
    params: list[str],
    max_rows_ch: int,
    gt_windows: list[tuple[datetime, datetime]] | None,
    results_acc: list[dict],
) -> None:
    # ── choose ML settings (auto: total effort = rows/ch × channels) ─────
    total_effort = max_rows_ch * len(params)
    use_ml = total_effort < _ML_EFFORT_LIMIT
    lstm_epochs = None if use_ml else 0
    tcn_epochs  = None if use_ml else 0

    # ── auto-cooldown from DB sample ─────────────────────────────────────
    cooldown_h = await _auto_cooldown_from_db(tenant_id, satellite_id, params[0])

    # ── z-threshold: higher for CATS oscillatory data ────────────────────
    sid_upper = satellite_id.upper()
    z_thresh = 6.0 if "CATS-3" in sid_upper else (3.5 if "CATS" in sid_upper else None)

    capped_rows = min(max_rows_ch, _MAX_ROWS_PER_CHANNEL)
    cap_note = f"  [capped at {_MAX_ROWS_PER_CHANNEL:,}]" if max_rows_ch > _MAX_ROWS_PER_CHANNEL else ""
    print(f"\n{'─'*65}")
    print(f"  Satellite : {satellite_id}  [{tenant_id}]")
    print(f"  Channels  : {len(params)}  |  Rows/ch : {max_rows_ch:,}{cap_note}  |  Effort : {total_effort:,}")
    ml_label = "ON (full 11-detector)" if use_ml else f"OFF stat-only (effort {total_effort:,} > {_ML_EFFORT_LIMIT:,})"
    print(f"  Cooldown  : {cooldown_h*60:.1f} min  |  ML : {ml_label}  |  z={z_thresh or 'default'}")
    if gt_windows is not None:
        print(f"  GT windows: {len(gt_windows)}")

    # ── clear previous results ────────────────────────────────────────────
    deleted = await _clear_anomalies(tenant_id, satellite_id)
    await _clear_detector_state(tenant_id, satellite_id)
    if deleted:
        print(f"  Cleared   : {deleted} old anomalies")

    set_tenant(tenant_id)
    t0 = time.monotonic()

    async with db_context():
        subsystem_map = {p: _infer_subsystem(satellite_id, p) for p in params}
        results = await run_bulk_detection(
            satellite_id=satellite_id,
            parameters=params,
            subsystem_map=subsystem_map,
            cooldown_hours=cooldown_h,
            lstm_epochs=lstm_epochs,
            tcn_epochs=tcn_epochs,
            z_threshold=z_thresh,
            max_rows_per_channel=_MAX_ROWS_PER_CHANNEL,
        )

    elapsed = time.monotonic() - t0
    all_anomalies = [a for v in results.values() for a in v]
    total_det = len(all_anomalies)

    print(f"  Detected  : {total_det} anomalies  ({elapsed:.1f}s)")

    # ── severity breakdown ────────────────────────────────────────────────
    from collections import Counter
    sev = Counter(a.severity.value if hasattr(a.severity, 'value') else str(a.severity)
                  for a in all_anomalies)
    print(f"  Severity  : {dict(sev)}")

    # ── detector breakdown ────────────────────────────────────────────────
    det_ctr: Counter = Counter()
    for a in all_anomalies:
        for d in a.detectors_triggered:
            det_ctr[d] += 1
    if det_ctr:
        top5 = det_ctr.most_common(5)
        print(f"  Detectors : {', '.join(f'{d}={n}' for d,n in top5)}")

    # ── scoring (fully auto-calibrated via AutoScorer) ────────────────────
    scored: dict | None = None
    if gt_windows is not None and len(gt_windows) > 0:
        detected_ts = await _fetch_anomaly_timestamps(tenant_id, satellite_id)
        scorer = AutoScorer(cooldown_hours=cooldown_h)
        result, meta = scorer.score(detected_ts, gt_windows)
        scored = {
            "satellite_id":  satellite_id,
            "tenant_id":     tenant_id,
            "gt_windows":    len(gt_windows),
            "detected":      result.detected_count,
            "tp":            result.tp,
            "fp":            result.fp,
            "fn":            result.fn,
            "precision":     result.precision,
            "recall":        result.recall,
            "f1":            result.f1,
            "raw_anomalies": total_det,
            "elapsed_s":     elapsed,
            "window_s":      meta["window_s"],
            "gap_s":         meta["gap_s"],
            "degenerate":    meta.get("degenerate", False),
            "ml":            use_ml,
        }
        print(f"  Score     : P={result.precision:.1%}  R={result.recall:.1%}  F1={result.f1:.1%}")
        print(f"              TP={result.tp}  FP={result.fp}  FN={result.fn}  (events={result.detected_count})")
        degen_note = "  [⚠ continuous-fire: point-level P/R]" if meta.get("degenerate") else ""
        print(f"  AutoScore : window={meta['window_s']:.0f}s  gap={meta['gap_s']:.0f}s  "
              f"lead_median={meta['lead_time_median_s']:.0f}s{degen_note}")
    else:
        scored = {
            "satellite_id": satellite_id,
            "tenant_id":    tenant_id,
            "gt_windows":   None,
            "detected":     total_det,
            "raw_anomalies": total_det,
            "elapsed_s":    elapsed,
            "ml":           use_ml,
        }
        print("  Score     : no GT available — detection stats only")

    results_acc.append(scored)


def _infer_subsystem(satellite_id: str, param: str) -> str:
    sid = satellite_id.upper()
    p   = param.lower()
    if "CADC" in param: return "eps"
    if any(x in p for x in ["temp", "thermo"]): return "thermal"
    if any(x in p for x in ["press", "volt", "curr"]): return "eps"
    if any(x in p for x in ["accel", "vibrat", "flow"]): return "mech"
    if any(x in p for x in ["frame", "byte", "gap"]): return "comms"
    if "channel" in p: return "eps"
    return "telemetry"


# ─────────────────────────────────────────────────────────────────────────────
# Final consolidated report
# ─────────────────────────────────────────────────────────────────────────────

def _print_final_report(results: list[dict]) -> None:
    W = 80
    print(f"\n{'═'*W}")
    print(f"{'DSREMO FULL BENCHMARK REPORT':^{W}}")
    print(f"{'═'*W}")
    print(f"{'Dataset':<26} {'GT':>5} {'Det':>6} {'TP':>4} {'FP':>4} {'FN':>4}  {'P':>7}  {'R':>7}  {'F1':>7}  {'Win(s)':>7}  {'Time':>6}  ML")
    print(f"{'─'*W}")

    scored_rows  = [r for r in results if r.get("f1") is not None]
    nogt_rows    = [r for r in results if r.get("gt_windows") is None]

    for r in results:
        sid = r["satellite_id"]
        if r.get("gt_windows") is not None:
            gt_str  = str(r["gt_windows"])
            det_str = str(r["detected"])
            tp_str  = str(r["tp"])
            fp_str  = str(r["fp"])
            fn_str  = str(r["fn"])
            p_str   = f"{r['precision']:.1%}"
            rec_str = f"{r['recall']:.1%}"
            f1_str  = f"{r['f1']:.1%}"
            win_str = ("pt-lvl" if r.get("degenerate") else f"{r.get('window_s', 0):.0f}")
        else:
            gt_str  = "—"
            det_str = str(r["detected"])
            tp_str = fp_str = fn_str = p_str = rec_str = f1_str = win_str = "—"
        t_str = f"{r['elapsed_s']:.0f}s"
        ml_str = "✓" if r.get("ml") else "stat"
        print(f"{sid:<26} {gt_str:>5} {det_str:>6} {tp_str:>4} {fp_str:>4} {fn_str:>4}  {p_str:>7}  {rec_str:>7}  {f1_str:>7}  {win_str:>7}  {t_str:>6}  {ml_str}")

    print(f"{'─'*W}")

    # Aggregate scored datasets only
    if scored_rows:
        total_gt  = sum(r["gt_windows"] for r in scored_rows)
        total_tp  = sum(r["tp"]  for r in scored_rows)
        total_fp  = sum(r["fp"]  for r in scored_rows)
        total_fn  = sum(r["fn"]  for r in scored_rows)
        agg_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        agg_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        agg_f1 = 2*agg_p*agg_r / (agg_p+agg_r) if (agg_p+agg_r) > 0 else 0.0
        print(f"{'AGGREGATE (scored)':<26} {total_gt:>5} {'':>6} {total_tp:>4} {total_fp:>4} {total_fn:>4}  {agg_p:>7.1%}  {agg_r:>7.1%}  {agg_f1:>7.1%}  {'auto':>7}  {'':>6}")

    print(f"{'═'*W}")
    print()
    print("Notes:")
    print("  stat    = statistical only (GRU+TCN disabled — >500K rows/ch)")
    print("  ✓       = full 11-detector ensemble including GRU + TCN")
    print("  GT      = ground-truth event windows used for scoring")
    print("  Det     = detected events after gap-clustering")
    print("  Win(s)  = auto-calibrated tolerance window (AutoScorer)")
    print("  P/R/F1  = event-level (not point-level)")
    print()

    if scored_rows:
        best = max(scored_rows, key=lambda r: r["f1"])
        worst = min(scored_rows, key=lambda r: r["f1"])
        print(f"  Best  F1: {best['satellite_id']} → {best['f1']:.1%}")
        print(f"  Worst F1: {worst['satellite_id']} → {worst['f1']:.1%}")
    print(f"{'═'*W}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Apply dsremo.yaml calibration constants before any detection runs.
    # Without this, calibration.py module defaults (100 samples, RECAL=10×)
    # diverge from config values (200 samples, RECAL=3×).
    from dsremo.core.config import load_config
    from dsremo.detection import calibration as _cal_mod
    try:
        _cfg = load_config()
        _cal_mod.init_from_config(_cfg)
    except Exception as _e:
        print(f"  [warn] Could not load dsremo.yaml: {_e} — using module defaults")

    print_run_header(
        "Dsremo — FULL AUTOMATED BENCHMARK",
        Datasets=f"{len(_TARGETS)} targets",
        GT_sources="CATS parquet y-col  |  SKAB GitHub CSV  |  ISS/ESA detection-only",
        ML_policy="Full ensemble (<500K rows/ch)  |  statistical-only above",
        Mode="Fresh run — anomalies cleared and re-detected",
    )

    # ── 1. Pre-load GT (before any DB work) ──────────────────────────────
    print("\n[GT] Loading ground-truth sources ...")

    cats_gt: list[tuple[datetime, datetime]] = []
    if _CATS_PARQUET.exists():
        with phase("CATS GT from parquet"):
            cats_gt = _extract_cats_gt()
            print(f"  {len(cats_gt)} anomaly windows extracted from y column")
    else:
        print("  [warn] CATS parquet not found — CATS will score without GT")

    skab_gt: list[tuple[datetime, datetime]] = []
    with phase("SKAB GT from GitHub"):
        skab_gt = _download_skab_gt()
        if skab_gt:
            print(f"  {len(skab_gt)} anomaly windows from valve2/1+2+3.csv")
        else:
            print("  [warn] SKAB download failed — using detection stats only")

    # ── 2. Run each target ───────────────────────────────────────────────
    results: list[dict] = []

    for tenant_id, satellite_id in _TARGETS:
        # Resolve parameters for this satellite
        all_sats = await _all_satellites(tenant_id)
        sat_map  = {s[0]: s for s in all_sats}

        if satellite_id not in sat_map:
            print(f"\n  [skip] {satellite_id} not found in tenant {tenant_id}")
            continue

        _, params, max_rows_ch = sat_map[satellite_id]

        # Assign GT
        sid_upper = satellite_id.upper()
        if "CATS"   in sid_upper and cats_gt:
            gt = cats_gt
        elif "SKAB"  in sid_upper and skab_gt:
            gt = skab_gt
        else:
            gt = None   # detection stats only

        await run_satellite(
            tenant_id=tenant_id,
            satellite_id=satellite_id,
            params=params,
            max_rows_ch=max_rows_ch,
            gt_windows=gt,
            results_acc=results,
        )

    # ── 3. Final report ──────────────────────────────────────────────────
    _print_final_report(results)


if __name__ == "__main__":
    asyncio.run(main())
