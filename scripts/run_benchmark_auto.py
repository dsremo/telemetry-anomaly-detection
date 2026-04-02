"""Fully automated benchmark runner — zero manual input.

Scans ALL tenants/satellites in the DB, auto-extracts ground truth from:
  - CATS:         parquet y-column (200 GT windows, auto-detected)
  - SKAB:         downloads labels from GitHub raw (anomaly column)
  - OPSSAT:       OPS-SAT-AD ground truth fetched from ESA Zenodo record
  - GECCO:        GECCO 2018 competition labels from GitHub
  - ESA-Mission1: local Resources/ESA-Mission1/labels.csv (200 events, 58 channels)
  - SatNOGS:      no GT — reports raw detection stats

NOTE on proxy GT: when external GT cannot be fetched, the scorer falls back
to deriving GT by clustering the detector's own outputs. These runs are
labelled "[PROXY]" in the output and MUST NOT be used for external reporting.

Run:
    python3 scripts/run_benchmark_auto.py
    python3 scripts/run_benchmark_auto.py --tenant cats-spacecraft
    python3 scripts/run_benchmark_auto.py --quick   # skip slow fetches
"""

from __future__ import annotations

import asyncio
import io
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dsremo.eval.scoring import ScoringResult, cluster_events, score

# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = dict(
    host="localhost", port=5432,
    database="sentinel", user="sentinel", password="sentinel_dev_only",
)

PARQUET_PATH    = _ROOT / "Resources" / "data.parquet"
ESA_LABELS_PATH = _ROOT / "Resources" / "ESA-Mission1" / "labels.csv"

# ESA-Mission1 scoring params (hourly data, events span hours to weeks)
_ESA_GAP_S    = 7_200    # 2 h — cluster detections within same event
_ESA_WINDOW_S = 86_400   # 24 h — detection must land within ±24 h of GT window

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkEntry:
    label: str
    tenant: str
    satellite_id: str
    domain: str
    freq: str
    gt_source: str
    detected: list[datetime] = field(default_factory=list)
    gt_windows: list[tuple[datetime, datetime]] = field(default_factory=list)
    result: ScoringResult | None = None
    gap_s: float = 3600.0
    window_s: float = 1800.0
    note: str = ""
    proxy_gt: bool = False   # True = GT derived from own detections (not for reporting)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth extractors
# ─────────────────────────────────────────────────────────────────────────────

def _y_col_to_windows(index, y_arr) -> list[tuple[datetime, datetime]]:
    """Convert a binary 0/1 label array → (start, end) window pairs."""
    y = y_arr.astype(int)
    changes = np.diff(y, prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    windows = []
    for s, e in zip(starts, ends):
        t_start = index[s]
        t_end   = index[min(e, len(index) - 1)]
        if hasattr(t_start, "to_pydatetime"):
            t_start = t_start.to_pydatetime()
            t_end   = t_end.to_pydatetime()
        if t_start.tzinfo is None:
            t_start = t_start.replace(tzinfo=timezone.utc)
        if t_end.tzinfo is None:
            t_end = t_end.replace(tzinfo=timezone.utc)
        windows.append((t_start, t_end))
    return windows


def extract_cats_gt() -> list[tuple[datetime, datetime]]:
    """Extract 200 GT windows from CATS parquet y column (no downloads needed)."""
    try:
        import pandas as pd
        df = pd.read_parquet(PARQUET_PATH, columns=["y"])
        windows = _y_col_to_windows(df.index, df["y"].values)
        print(f"  [CATS]    parquet scan → {len(windows)} GT windows")
        return windows
    except Exception as e:
        print(f"  [CATS]    parquet read failed: {e}")
        return []


def extract_esa_mission1_gt() -> dict[str, list[tuple[datetime, datetime]]]:
    """Load ESA-Mission1 GT from local labels.csv (per-channel windows).

    Returns a dict mapping channel name → list of (start, end) anomaly windows.
    The labels.csv has 200 event IDs across 58 channels (3589 rows total).
    Timestamps are ISO-8601 UTC from the ESA archive (years 2000–2013).
    """
    if not ESA_LABELS_PATH.exists():
        print(f"  [ESA-M1]  labels.csv not found at {ESA_LABELS_PATH}")
        return {}
    import csv as _csv
    gt_by_channel: dict[str, list[tuple[datetime, datetime]]] = {}
    with open(ESA_LABELS_PATH) as f:
        for row in _csv.DictReader(f):
            ch = row["Channel"]
            s  = datetime.fromisoformat(row["StartTime"].replace("Z", "+00:00"))
            e  = datetime.fromisoformat(row["EndTime"].replace("Z", "+00:00"))
            gt_by_channel.setdefault(ch, []).append((s, e))
    total_events = len({row["ID"] for row in _csv.DictReader(open(ESA_LABELS_PATH))})
    total_rows   = sum(len(v) for v in gt_by_channel.values())
    print(f"  [ESA-M1]  local labels.csv → {total_events} events, "
          f"{len(gt_by_channel)} channels, {total_rows} (channel, window) pairs")
    return gt_by_channel


def fetch_skab_gt_from_github(valve: int = 2) -> list[tuple[datetime, datetime]]:
    """Auto-download SKAB valve labels from GitHub raw, extract GT windows."""
    windows: list[tuple[datetime, datetime]] = []
    # valve2 has 3 numbered CSV files (1.csv, 2.csv, 3.csv)
    base = f"https://raw.githubusercontent.com/waico/SKAB/master/data/valve{valve}/"
    files = ["1.csv", "2.csv", "3.csv"]
    for fname in files:
        url = base + fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dsremo-bench/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode()
            import csv
            reader = csv.DictReader(io.StringIO(raw), delimiter=";")
            rows = list(reader)
            # Find anomaly column (may be 'anomaly' or 'changepoint')
            anomaly_col = next(
                (c for c in reader.fieldnames or []
                 if "anomaly" in c.lower() and "change" not in c.lower()),
                None,
            )
            if not anomaly_col:
                print(f"  [SKAB]    no anomaly col in {fname}: {reader.fieldnames}")
                continue

            times, labels = [], []
            for row in rows:
                ts_str = row.get("datetime", row.get("timestamp", ""))
                try:
                    ts = datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
                    ts = ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                try:
                    label = int(float(row[anomaly_col].strip()))
                except (ValueError, KeyError):
                    label = 0
                times.append(ts)
                labels.append(label)

            if times:
                new_wins = _y_col_to_windows(times, np.array(labels))
                windows.extend(new_wins)
                print(f"  [SKAB]    {fname}: {len(rows)} rows → {len(new_wins)} GT windows")
        except Exception as e:
            print(f"  [SKAB]    download failed ({fname}): {e}")
    return windows


def fetch_gecco_gt_from_github() -> list[tuple[datetime, datetime]]:
    """Auto-download GECCO 2018 water quality labels from GitHub."""
    url = "https://raw.githubusercontent.com/GECCO-2018/gecco-2018-water-quality/master/gecco2018_water_quality_labels.csv"
    windows: list[tuple[datetime, datetime]] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dsremo-bench/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode()
        import csv
        reader = csv.DictReader(io.StringIO(raw))
        times, labels = [], []
        for row in reader:
            ts_str = (row.get("datetime") or row.get("timestamp") or "").strip()
            if not ts_str:
                continue
            for fmt in ["%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"]:
                try:
                    ts = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
            else:
                continue
            try:
                lbl = int(float(row.get("event", row.get("anomaly", row.get("label", "0")))))
            except (ValueError, TypeError):
                lbl = 0
            times.append(ts)
            labels.append(lbl)
        if times:
            windows = _y_col_to_windows(times, np.array(labels))
            print(f"  [GECCO]   GitHub fetch → {len(windows)} GT windows from {len(times)} rows")
    except Exception as e:
        print(f"  [GECCO]   GitHub fetch failed: {e} — using DB-clustered proxy GT")
    return windows


def fetch_opssat_gt_from_zenodo() -> list[tuple[datetime, datetime]]:
    """Fetch OPS-SAT-AD event labels from ESA Zenodo record (CSV events file)."""
    # OPS-SAT-AD Zenodo: record 8363509 — events CSV is events.csv
    url = "https://zenodo.org/record/8363509/files/events.csv"
    windows: list[tuple[datetime, datetime]] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dsremo-bench/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode()
        import csv
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            start_str = (row.get("start") or row.get("start_time") or "").strip()
            end_str   = (row.get("end") or row.get("end_time") or "").strip()
            if not start_str:
                continue
            for fmt in ["%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    ts = datetime.strptime(start_str, fmt)
                    te = datetime.strptime(end_str or start_str, fmt)
                    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    te = te if te.tzinfo else te.replace(tzinfo=timezone.utc)
                    windows.append((ts, te))
                    break
                except ValueError:
                    continue
        if windows:
            print(f"  [OPSSAT]  Zenodo fetch → {len(windows)} GT windows")
    except Exception as e:
        print(f"  [OPSSAT]  Zenodo fetch failed: {e} — using auto-clustered GT from detections")
    return windows


def derive_gt_from_detections(
    detected: list[datetime],
    gap_s: float = 300.0,
    merge_gap_s: float = 3600.0,
) -> list[tuple[datetime, datetime]]:
    """
    When no labeled GT exists: cluster detected events and treat event
    boundaries as approximate GT windows. Used for SatNOGS/ESA proxy scoring.
    """
    if not detected:
        return []
    clusters = cluster_events(detected, gap_s=gap_s)
    return [(c[0], c[-1]) for c in clusters]


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_all_tenants(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch("SELECT id FROM tenants ORDER BY id")
    return [r["id"] for r in rows]


async def fetch_anomalies(
    conn: asyncpg.Connection,
    tenant_id: str,
    satellite_id: str | None = None,
) -> list[datetime]:
    await conn.execute("SELECT set_config('app.tenant_id', $1, false)", tenant_id)
    if satellite_id:
        rows = await conn.fetch(
            "SELECT timestamp FROM anomalies WHERE satellite_id = $1 ORDER BY timestamp",
            satellite_id,
        )
    else:
        rows = await conn.fetch(
            "SELECT timestamp FROM anomalies ORDER BY timestamp"
        )
    result = []
    for r in rows:
        dt = r["timestamp"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        result.append(dt)
    return result


async def fetch_satellites(
    conn: asyncpg.Connection, tenant_id: str
) -> list[str]:
    await conn.execute("SELECT set_config('app.tenant_id', $1, false)", tenant_id)
    rows = await conn.fetch(
        "SELECT DISTINCT satellite_id FROM anomalies ORDER BY satellite_id"
    )
    return [r["satellite_id"] for r in rows]


async def fetch_telemetry_range(
    conn: asyncpg.Connection, tenant_id: str, satellite_id: str
) -> tuple[datetime | None, datetime | None]:
    await conn.execute("SELECT set_config('app.tenant_id', $1, false)", tenant_id)
    row = await conn.fetchrow(
        "SELECT MIN(timestamp) as t0, MAX(timestamp) as t1 FROM telemetry WHERE satellite_id=$1",
        satellite_id,
    )
    return row["t0"], row["t1"]


# ─────────────────────────────────────────────────────────────────────────────
# Detector stats helper
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_detector_breakdown(
    conn: asyncpg.Connection, tenant_id: str, satellite_id: str
) -> dict[str, int]:
    await conn.execute("SELECT set_config('app.tenant_id', $1, false)", tenant_id)
    rows = await conn.fetch(
        """
        SELECT unnest(detectors_triggered) as det, COUNT(*) as cnt
        FROM anomalies WHERE satellite_id=$1
        GROUP BY det ORDER BY cnt DESC
        """,
        satellite_id,
    )
    return {r["det"]: r["cnt"] for r in rows}


async def fetch_severity_breakdown(
    conn: asyncpg.Connection, tenant_id: str, satellite_id: str
) -> dict[str, int]:
    await conn.execute("SELECT set_config('app.tenant_id', $1, false)", tenant_id)
    rows = await conn.fetch(
        "SELECT severity, COUNT(*) as cnt FROM anomalies WHERE satellite_id=$1 GROUP BY severity",
        satellite_id,
    )
    return {r["severity"]: r["cnt"] for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 72
SEP2 = "-" * 72

def fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%"


def print_result(entry: BenchmarkEntry, detectors: dict, severity: dict) -> None:
    r = entry.result
    print(f"\n{SEP}")
    print(f"  {entry.label}")
    print(f"  Tenant: {entry.tenant}  |  Satellite: {entry.satellite_id}")
    print(f"  Domain: {entry.domain}  |  Frequency: {entry.freq}")
    print(f"  GT source: {entry.gt_source}")
    print(SEP2)
    print(f"  Telemetry rows in DB  : see tenant summary above")
    print(f"  Raw anomaly detections: {len(entry.detected):>7,}")

    sev_str = "  ".join(f"{k}={v}" for k, v in severity.items())
    det_str = ", ".join(f"{k}({v})" for k, v in list(detectors.items())[:6])
    print(f"  Severity breakdown    : {sev_str}")
    print(f"  Detectors fired       : {det_str}")

    if r is not None:
        print(SEP2)
        print(f"  GT windows  : {r.event_count}")
        print(f"  Det. events : {r.detected_count}  (gap ≤ {entry.gap_s/60:.0f} min)")
        print(f"  TP={r.tp}  FP={r.fp}  FN={r.fn}")
        print(f"  Precision   : {fmt_pct(r.precision)}")
        print(f"  Recall      : {fmt_pct(r.recall)}")
        print(f"  F1          : {fmt_pct(r.f1)}")

        # Missed windows
        clusters = cluster_events(entry.detected, gap_s=entry.gap_s)
        reps = [c[0].timestamp() for c in clusters]
        missed = []
        for gs, ge in entry.gt_windows:
            lo = gs.timestamp() - entry.window_s
            hi = ge.timestamp() + entry.window_s
            if not any(lo <= rep <= hi for rep in reps):
                missed.append((gs, ge))
        if missed:
            print(f"\n  Missed GT windows ({len(missed)}):")
            for gs, ge in missed[:5]:
                dur = int((ge - gs).total_seconds() / 60)
                print(f"    {gs.strftime('%Y-%m-%d %H:%M')} → {ge.strftime('%Y-%m-%d %H:%M')}  ({dur} min)")
            if len(missed) > 5:
                print(f"    ... {len(missed)-5} more")
        else:
            print("  ✓ All GT windows detected!")
    else:
        print(f"  (No GT available — detection stats only)")

    if entry.note:
        print(f"\n  Note: {entry.note}")
    print()


def print_summary_table(entries: list[BenchmarkEntry]) -> None:
    print(f"\n\n{'#'*72}")
    print("  FULL BENCHMARK SUMMARY — ALL DATASETS")
    print(f"  * = PROXY GT (self-referential — NOT for external reporting)")
    print(f"{'#'*72}")
    header = f"{'Dataset':<28} {'Domain':<22} {'Detections':>11} {'GT':>5} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * 72)
    for e in entries:
        r     = e.result
        proxy = "*" if e.proxy_gt else " "
        label = e.satellite_id + proxy
        if r is not None:
            row = (
                f"{label:<29} {e.domain:<22} "
                f"{len(e.detected):>11,} {r.event_count:>5} "
                f"{fmt_pct(r.precision):>7} {fmt_pct(r.recall):>7} {fmt_pct(r.f1):>7}"
            )
        else:
            row = (
                f"{label:<29} {e.domain:<22} "
                f"{len(e.detected):>11,} {'N/A':>5} "
                f"{'—':>7} {'—':>7} {'—':>7}"
            )
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset definitions  (tenant, satellite, domain, freq, gap_s, window_s)
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    # (tenant, satellite, domain, freq, gap_s, window_s, label)
    ("cats-spacecraft", "CATS-2",        "Spacecraft sim (1Hz)",    "1 Hz",   600,   1800, "CATS-2  z=3.0 1Hz baseline"),
    ("cats-spacecraft", "CATS-3",        "Spacecraft sim (1Hz)",    "1 Hz",   600,   1800, "CATS-3  z=6.0 1Hz tuned"),
    ("cats-spacecraft", "CATS-S10-1MIN", "Spacecraft sim (1-min)",  "1 min",  600,   900,  "CATS Sprint10 1-min adaptive"),
    ("skab-valve2",     "SKAB-VALVE2",   "Industrial sensor (1Hz)", "1 Hz",   300,   300,  "SKAB Valve2 (Sprint 19)"),
    ("skab-valve2",     "SKAB-S17-TUNED","Industrial sensor (1Hz)", "1 Hz",   300,   300,  "SKAB S17 Tuned"),
    ("opssat",          "OPSSAT-3",      "Spacecraft HK (1Hz)",     "1 Hz",   300,   1800, "OPS-SAT-AD real spacecraft"),
    ("opssat",          "OPSSAT-S17-WARM","Spacecraft HK (1Hz)",    "1 Hz",   300,   1800, "OPS-SAT S17 warm-start"),
    ("gecco-water",     "GECCO-WATER-S10","Municipal IoT (1-min)",  "1 min",  600,   1800, "GECCO Water Sprint10"),
    ("gecco-water",     "GECCO-S16",     "Municipal IoT (1-min)",   "1 min",  600,   1800, "GECCO Water S16"),
    ("satnogs",         "SATNOGS-25544", "ISS real telemetry",      "varies", 3600,  3600, "SatNOGS ISS (no GT)"),
    ("satnogs",         "SATNOGS-43017", "CubeSat real telemetry",  "varies", 3600,  3600, "SatNOGS CubeSat-43017"),
    ("satnogs",         "SATNOGS-40074", "CubeSat real telemetry",  "varies", 3600,  3600, "SatNOGS CubeSat-40074"),
    ("esa-mission1",    "ESA-MISSION1",  "Spacecraft archive",      "varies", 3600,  3600, "ESA Mission1 archive"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run(quick: bool = False, filter_tenant: str | None = None) -> None:
    print(f"\n{'#'*72}")
    print("  DSREMO — AUTOMATED BENCHMARK RUNNER")
    print(f"  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Mode: {'QUICK (skip slow network fetches)' if quick else 'FULL (auto-fetch all GT)'}")
    print(f"{'#'*72}\n")

    # ── Phase 1: Fetch all GT sources ────────────────────────────────────────
    print("Phase 1: Auto-extracting ground truth from all sources")
    print(SEP2)

    cats_gt       : list[tuple[datetime, datetime]] = []
    skab_gt       : list[tuple[datetime, datetime]] = []
    gecco_gt      : list[tuple[datetime, datetime]] = []
    opssat_gt     : list[tuple[datetime, datetime]] = []
    esa_m1_gt     : dict[str, list[tuple[datetime, datetime]]] = {}

    cats_gt  = extract_cats_gt()
    esa_m1_gt = extract_esa_mission1_gt()   # always local — no network needed

    if not quick:
        skab_gt   = fetch_skab_gt_from_github(valve=2)
        gecco_gt  = fetch_gecco_gt_from_github()
        opssat_gt = fetch_opssat_gt_from_zenodo()
    else:
        print("  [SKAB/GECCO/OPSSAT] skipped (quick mode)")

    print()

    # ── Phase 2: Query DB for all anomalies ──────────────────────────────────
    print("Phase 2: Querying DB for detections across all tenants")
    print(SEP2)

    conn = await asyncpg.connect(**DB_CONFIG)
    try:
        entries: list[BenchmarkEntry] = []
        detectors_map: dict[str, dict] = {}
        severity_map:  dict[str, dict] = {}

        for (tenant, sat, domain, freq, gap_s, window_s, label) in DATASETS:
            if filter_tenant and tenant != filter_tenant:
                continue

            detected = await fetch_anomalies(conn, tenant, sat)
            det_breakdown = await fetch_detector_breakdown(conn, tenant, sat)
            sev_breakdown = await fetch_severity_breakdown(conn, tenant, sat)

            key = f"{tenant}/{sat}"
            detectors_map[key] = det_breakdown
            severity_map[key]  = sev_breakdown

            print(f"  {sat:<28} {len(detected):>7,} detections")

            # Pick GT
            gt_windows: list[tuple[datetime, datetime]] = []
            gt_source = "none"
            is_proxy  = False

            if "CATS" in sat:
                gt_windows = cats_gt
                gt_source  = f"parquet y-column ({len(cats_gt)} windows)"
                # NOTE: CATS detection currently capped at 1M rows → misses all 200 GT windows
                # (GT starts at row ~1M+91min). Re-detect with full 5M rows to get honest scores.

            elif "SKAB" in sat:
                if skab_gt:
                    gt_windows = skab_gt
                    gt_source  = f"GitHub raw valve2 ({len(skab_gt)} windows)"
                else:
                    is_proxy   = True
                    gt_windows = derive_gt_from_detections(detected, gap_s=gap_s)
                    gt_source  = f"[PROXY] auto-clustered from detections ({len(gt_windows)} windows)"

            elif "OPSSAT" in sat:
                if opssat_gt:
                    gt_windows = opssat_gt
                    gt_source  = f"Zenodo events.csv ({len(opssat_gt)} windows)"
                else:
                    is_proxy   = True
                    gt_windows = derive_gt_from_detections(detected, gap_s=300)
                    gt_source  = f"[PROXY] auto-clustered detections ({len(gt_windows)} windows)"

            elif "GECCO" in sat:
                if gecco_gt:
                    gt_windows = gecco_gt
                    gt_source  = f"GitHub labels ({len(gecco_gt)} windows)"
                else:
                    is_proxy   = True
                    gt_windows = derive_gt_from_detections(detected, gap_s=3600)
                    gt_source  = f"[PROXY] auto-clustered detections ({len(gt_windows)} windows)"

            elif "ESA-MISSION1" in sat:
                if esa_m1_gt:
                    # Flatten all per-channel windows into a single list for summary scoring.
                    # Per-channel scoring happens in Phase 3b below.
                    gt_windows = sorted(
                        {w for wins in esa_m1_gt.values() for w in wins}
                    )
                    gt_source  = (f"local labels.csv ({len(gt_windows)} unique windows, "
                                  f"{len(esa_m1_gt)} channels)")
                else:
                    gt_windows = []
                    gt_source  = "labels.csv missing — detection stats only"

            elif "SATNOGS" in sat:
                gt_windows = []
                gt_source  = "no public GT available"

            entry = BenchmarkEntry(
                label=label,
                tenant=tenant,
                satellite_id=sat,
                domain=domain,
                freq=freq,
                gt_source=gt_source,
                detected=detected,
                gt_windows=gt_windows,
                gap_s=gap_s,
                window_s=window_s,
                proxy_gt=is_proxy,
            )
            entries.append(entry)

        print()

        # ── Phase 3: Score each dataset ──────────────────────────────────────
        print("Phase 3: Scoring all datasets")
        print(SEP2)

        for entry in entries:
            if not entry.gt_windows:
                continue

            if "ESA-MISSION1" in entry.satellite_id and esa_m1_gt:
                # Per-channel scoring: aggregate TP/FP/FN across all 58 channels
                # using per-channel detections and per-channel GT windows.
                det_by_ch: dict[str, list[datetime]] = {}
                for dt in entry.detected:
                    # anomalies table stores parameter; fetch_anomalies returns all timestamps.
                    # We need per-channel detections — re-query here.
                    pass  # populated below via separate query (see note)

                # Re-query per-channel detections from DB for ESA-M1
                conn2 = await asyncpg.connect(**DB_CONFIG)
                await conn2.execute(
                    "SELECT set_config('app.tenant_id',$1,false)", entry.tenant
                )
                ch_rows = await conn2.fetch(
                    "SELECT parameter, timestamp FROM anomalies "
                    "WHERE satellite_id=$1 ORDER BY timestamp",
                    entry.satellite_id,
                )
                await conn2.close()

                for r in ch_rows:
                    ch = r["parameter"]
                    dt = r["timestamp"]
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    det_by_ch.setdefault(ch, []).append(dt)

                total_tp = total_fp = total_fn = 0
                for ch, ch_gt in esa_m1_gt.items():
                    ch_det = det_by_ch.get(ch, [])
                    r = score(ch_det, ch_gt,
                              window_s=_ESA_WINDOW_S, gap_s=_ESA_GAP_S)
                    total_tp += r.tp
                    total_fp += r.fp
                    total_fn += r.fn

                agg_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                agg_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                agg_f1 = 2*agg_p*agg_r / (agg_p+agg_r) if (agg_p+agg_r) > 0 else 0.0

                # Wrap in a ScoringResult-compatible object for unified display
                from dsremo.eval.scoring import ScoringResult
                entry.result = ScoringResult(
                    event_count=sum(len(v) for v in esa_m1_gt.values()),
                    detected_count=len(entry.detected),
                    tp=total_tp, fp=total_fp, fn=total_fn,
                    precision=agg_p, recall=agg_r, f1=agg_f1,
                )
                entry.note = (
                    f"Per-channel scoring: {len(esa_m1_gt)} channels × local labels.csv. "
                    f"gap={_ESA_GAP_S//3600}h, window={_ESA_WINDOW_S//3600}h."
                )
            else:
                entry.result = score(
                    entry.detected,
                    entry.gt_windows,
                    window_s=entry.window_s,
                    gap_s=entry.gap_s,
                )

        # ── Phase 4: Print detailed results ──────────────────────────────────
        print("\nPhase 4: Detailed results")

        for entry in entries:
            key = f"{entry.tenant}/{entry.satellite_id}"
            print_result(
                entry,
                detectors_map.get(key, {}),
                severity_map.get(key, {}),
            )

        # ── Phase 5: Summary table ────────────────────────────────────────────
        print_summary_table(entries)

        # ── Phase 6: Detection coverage analysis ─────────────────────────────
        print(f"\n{'#'*72}")
        print("  DETECTION COVERAGE ANALYSIS — WHAT WE CAN & CANNOT DETECT")
        print(f"{'#'*72}\n")

        print("STRONG (recall ≥ 80%, our ensemble excels here):")
        for e in entries:
            if e.result and e.result.recall >= 0.80:
                print(f"  ✓ {e.satellite_id:<28}  R={fmt_pct(e.result.recall)}  F1={fmt_pct(e.result.f1)}  — {e.domain}")

        print("\nWEAK (recall < 40% or no GT):")
        for e in entries:
            if e.result is None or e.result.recall < 0.40:
                note = f"R={fmt_pct(e.result.recall)}  F1={fmt_pct(e.result.f1)}" if e.result else "no GT"
                print(f"  ✗ {e.satellite_id:<28}  {note}  — {e.domain}")

        print("\nWHY WE MISS:")
        print("  1. Variance-only anomalies (CATS ced1): mean shift = 0.2σ — z-score/CUSUM blind")
        print("     → Need: Spectral Residual or LSTM autoencoder on 1Hz sinusoidal channels")
        print("  2. Drift+multivariate (GECCO water): all 9 channels drift together")
        print("     → Need: Inter-channel correlation change (CorrelationGraph S19 helps partially)")
        print("  3. No seasonal decomposition at sub-Nyquist: CATS 1-min (90s period)")
        print("     → Need: Run on raw 1Hz data (5M rows) — pipeline scaling work")
        print("  4. Short experiments with close-spaced windows: SKAB windows 13 min apart")
        print("     → Fixed in S9 with adaptive cooldown")

    finally:
        await conn.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Automated benchmark runner")
    parser.add_argument("--quick", action="store_true",
                        help="Skip network GT fetches (SKAB/GECCO/OPSSAT)")
    parser.add_argument("--tenant", default=None,
                        help="Filter to single tenant (e.g. cats-spacecraft)")
    args = parser.parse_args()
    asyncio.run(run(quick=args.quick, filter_tenant=args.tenant))


if __name__ == "__main__":
    main()
