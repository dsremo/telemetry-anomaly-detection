"""Detection pipeline orchestrator — the brain of Sentinel.

Detection order per parameter (post-STL architecture):
    1. Fetch recent window from DB (600 pts, oldest → newest)
    2. STL decompose → residuals  (seasonal removed; trend kept for CUSUM)
    3. Update per-channel calibration (ref_mean, ref_std, k, H, UCL, LCL)
    4. CUSUM  on residuals  — gradual drift accumulation (NASA standard)
    5. EWMA   on residuals  — sudden level shifts
    6. Z-score on residuals — single-point spikes
    7. PELT   on residuals  — abrupt structural breaks
    8. Isolation Forest on raw values — multivariate cross-parameter
    9. Ensemble vote → confidence + severity
   10. Store anomaly, broadcast via WebSocket

Ensemble weights:
    cusum:            0.30   (primary drift detector)
    ewma:             0.25   (level shift detector)
    statistical:      0.20   (spike detector)
    changepoint:      0.15   (structural break detector)
    isolation_forest: 0.10   (multivariate, only for standard parameters)

Severity gate:   derived from ensemble confidence only.
    watch   >= 0.50
    warning >= 0.65
    critical>= 0.85
All gates configurable via sentinel.yaml.  Zero hardcoded thresholds.

State persistence:
    CUSUM and EWMA accumulators are flushed to DB every STATE_FLUSH_EVERY
    detection cycles so they survive server restarts.  Calibration states
    are also persisted whenever a channel transitions to "calibrated".
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import structlog

from sentinel.core.models import Anomaly, DetectorResult, Severity
from sentinel.db import queries
from sentinel.detection.calibration import CalibrationManager
from sentinel.detection.changepoint import ChangePointDetector
from sentinel.detection.cusum import CUSUMDetector
from sentinel.detection.ewma import EWMADetector
from sentinel.detection.isolation import IsolationForestDetector
from sentinel.detection.statistical import StatisticalDetector
from sentinel.detection.stl_decomposer import STLDecomposer
from sentinel.features.engine import FeatureEngine

logger = structlog.get_logger()

# ── Singleton instances — created once at startup via init_detectors() ──────
_feature_engine:    FeatureEngine            = FeatureEngine(window_size=600)
_stl_decomposer:    STLDecomposer            = STLDecomposer()
_calibration_mgr:   CalibrationManager       = CalibrationManager()
_cusum_detector:    CUSUMDetector            = CUSUMDetector()
_ewma_detector:     EWMADetector             = EWMADetector()
_stat_detector:     StatisticalDetector      = StatisticalDetector()
_iso_detector:      IsolationForestDetector  = IsolationForestDetector()
_cp_detector:       ChangePointDetector      = ChangePointDetector()

# Ensemble weights (must sum to 1.0, overridable via config)
WEIGHTS: dict[str, float] = {
    "cusum":            0.30,
    "ewma":             0.25,
    "statistical":      0.20,
    "changepoint":      0.15,
    "isolation_forest": 0.10,
}

# Severity thresholds on the ensemble confidence score.
_severity_thresholds: dict[str, float] = {
    "watch":    0.50,
    "warning":  0.65,
    "critical": 0.85,
}

# Standard parameters for multivariate Isolation Forest (simulator / demo data).
_ALL_PARAMETERS: list[str] = [
    "battery_voltage", "battery_current", "solar_array_current", "bus_voltage",
    "wheel_speed_x", "wheel_speed_y", "wheel_speed_z", "pointing_error",
    "panel_temp_sun", "panel_temp_shade", "battery_temp", "electronics_temp",
    "signal_strength", "bit_error_rate", "link_margin",
]

# State flush to DB: persist CUSUM/EWMA/calibration every N cycles.
_STATE_FLUSH_EVERY: int = 50
_detection_cycle_count: dict[str, int] = {}

# Track samples since last Isolation Forest refit per satellite.
_samples_since_fit: dict[str, int] = {}

# Per-channel last-processed timestamp (epoch seconds).
# Each detection cycle only feeds NEW residuals into the state machines
# (calibration, CUSUM, EWMA) — points with timestamp <= this value were
# already processed in a previous cycle and must not be fed again.
_last_processed_ts: dict[str, float] = {}

# Alert cooldown: suppress repeated alarms for the same channel within N hours.
# Prevents a sustained anomalous regime from generating thousands of records.
# Configurable via sentinel.yaml detection.alert_cooldown_hours.
_alert_cooldown_s: float = 72.0 * 3600.0   # default 72 h; overridden by init
_last_anomaly_ts:  dict[str, float] = {}    # key → epoch of last stored anomaly

# Per-channel threshold overrides — keyed by (satellite_id, parameter).
# NULL/missing fields fall through to global defaults.
# Refreshed at startup and after any PUT/DELETE via load_channel_configs().
_channel_config_cache: dict[tuple[str, str], dict] = {}


# ── Initialisation ───────────────────────────────────────────────────────────

def init_detectors(settings: object) -> None:
    """Wire config values into all detector singletons.  Call once at startup."""
    global _feature_engine, _stl_decomposer, _calibration_mgr
    global _cusum_detector, _ewma_detector, _stat_detector
    global _iso_detector, _cp_detector
    global WEIGHTS, _severity_thresholds
    global _last_processed_ts, _detection_cycle_count, _samples_since_fit
    global _alert_cooldown_s, _last_anomaly_ts

    det  = settings.get("detection", {})   # type: ignore[attr-defined]
    feat = settings.get("features",  {})   # type: ignore[attr-defined]

    # ── Statistical (z-score spike detector) ─────────────────────────────
    z_thresh   = float(det.get("z_score_threshold",  3.0))
    severe_z   = z_thresh * 1.5
    _stat_detector = StatisticalDetector(
        z_threshold=z_thresh,
        severe_z_threshold=severe_z,
    )

    # ── Isolation Forest ──────────────────────────────────────────────────
    _iso_detector = IsolationForestDetector(
        contamination=float(det.get("isolation_contamination", 0.01)),
    )

    # ── PELT changepoint ──────────────────────────────────────────────────
    _cp_detector = ChangePointDetector(
        penalty=float(det.get("changepoint_penalty", 10.0)),
        min_segment_size=int(det.get("changepoint_min_size", 50)),
    )

    # ── STL decomposer ────────────────────────────────────────────────────
    orbital_period_s = int(feat.get("orbital_period", 5400))
    _stl_decomposer  = STLDecomposer(
        orbital_period_s=orbital_period_s,
        recompute_every=int(det.get("stl_recompute_every", 30)),
    )

    # ── Calibration (shared params read by CalibrationManager internals) ──
    import sentinel.detection.calibration as _cal_mod
    _cal_mod.CALIBRATION_WINDOW    = int(det.get("calibration_window", 100))
    _cal_mod.CUSUM_K_FACTOR        = float(det.get("cusum_k_factor", 0.5))
    _cal_mod.CUSUM_H_FACTOR        = float(det.get("cusum_h_factor", 5.0))
    _cal_mod.EWMA_LAMBDA           = float(det.get("ewma_lambda", 0.2))
    _cal_mod.EWMA_SIGMA_FACTOR     = float(det.get("ewma_sigma_factor", 3.0))
    _cal_mod.RECAL_FACTOR          = float(det.get("cusum_recal_factor", 10.0))
    _cal_mod.SIGMA_UPDATE_INTERVAL = int(det.get("sigma_update_interval", 720))
    # Recompute the spread constant in calibration module after lambda change.
    import math
    lam = _cal_mod.EWMA_LAMBDA
    _cal_mod._ewma_spread = math.sqrt(lam / (2.0 - lam))

    _calibration_mgr = CalibrationManager()

    # ── EWMA: update lambda from config ───────────────────────────────────
    _ewma_detector = EWMADetector(lam=float(det.get("ewma_lambda", 0.2)))

    # ── CUSUM / EWMA ──────────────────────────────────────────────────────
    _cusum_detector = CUSUMDetector()

    # ── Ensemble weights ──────────────────────────────────────────────────
    cfg_weights = det.get("ensemble_weights", {})
    if cfg_weights:
        for k in WEIGHTS:
            if k in cfg_weights:
                WEIGHTS[k] = float(cfg_weights[k])

    # ── Severity thresholds ───────────────────────────────────────────────
    sev = det.get("severity_thresholds", {})
    _severity_thresholds.update({
        "watch":    float(sev.get("watch",    0.50)),
        "warning":  float(sev.get("warning",  0.65)),
        "critical": float(sev.get("critical", 0.85)),
    })

    # Alert cooldown from config (default 72 h).
    _alert_cooldown_s = float(det.get("alert_cooldown_hours", 72.0)) * 3600.0

    # Reset per-channel state so a server restart or re-init starts clean.
    _last_processed_ts.clear()
    _last_anomaly_ts.clear()
    _detection_cycle_count.clear()
    _samples_since_fit.clear()

    logger.info(
        "detectors_initialized",
        z_threshold=z_thresh,
        orbital_period_s=orbital_period_s,
        cusum_k_factor=_cal_mod.CUSUM_K_FACTOR,
        cusum_h_factor=_cal_mod.CUSUM_H_FACTOR,
        ewma_lambda=lam,
        severity_thresholds=_severity_thresholds,
        weights=WEIGHTS,
    )


# ── Per-channel config cache + threshold helpers ──────────────────────────────

def load_channel_configs(configs: list[dict]) -> None:
    """Replace the in-memory channel config cache from a list of DB rows.

    Called once at server startup and again after any PUT/DELETE to a channel
    config so the detection pipeline picks up new overrides immediately.
    """
    global _channel_config_cache
    _channel_config_cache = {
        (row["satellite_id"], row["parameter"]): row
        for row in configs
    }
    logger.debug("channel_config_cache_updated", count=len(_channel_config_cache))


def get_effective_thresholds(satellite_id: str, parameter: str) -> dict:
    """Return merged thresholds: per-channel overrides applied on top of globals.

    DRY single source of truth — used by both the detection pipeline AND the
    GET /channels API response. NULL fields fall through to global defaults.
    """
    cfg = _channel_config_cache.get((satellite_id, parameter), {})

    def _get(key: str, default):  # type: ignore[no-untyped-def]
        v = cfg.get(key)
        return v if v is not None else default

    return {
        "z_threshold":      _get("z_threshold",     _stat_detector.z_threshold),
        "cusum_h":          _get("cusum_h",          None),   # None = calibration-computed
        "cusum_k":          _get("cusum_k",          None),   # None = calibration-computed
        "ewma_lambda":      _get("ewma_lambda",      None),   # None = calibration-computed
        "ewma_sigma_mult":  _get("ewma_sigma_mult",  None),   # None = calibration-computed
        "min_confidence":   _get("min_confidence",   0.0),
        "alert_cooldown_s": _get("alert_cooldown_s", _alert_cooldown_s),
    }


def _apply_calibration_overrides(cal: "CalibrationState", eff: dict) -> None:  # type: ignore[name-defined]
    """Mutate a CalibrationState in-place with any non-None override values.

    asyncio is single-threaded — this is race-free. CalibrationState is
    per-channel and not shared across concurrent calls.
    Only applies CUSUM/EWMA overrides when the channel has a valid σ_ref
    (i.e., after calibration completes).
    """
    if eff.get("cusum_h") is not None:
        cal.cusum_h = float(eff["cusum_h"])
    if eff.get("cusum_k") is not None:
        cal.cusum_k = float(eff["cusum_k"])

    # EWMA control limits depend on both lambda and sigma_mult.
    ewma_lambda     = eff.get("ewma_lambda")
    ewma_sigma_mult = eff.get("ewma_sigma_mult")
    if (ewma_lambda is not None or ewma_sigma_mult is not None) and cal.ref_std > 1e-9:
        import math
        import sentinel.detection.calibration as _cal_mod_local
        sigma  = cal.ref_std
        lam    = float(ewma_lambda)     if ewma_lambda     is not None else _cal_mod_local.EWMA_LAMBDA
        smult  = float(ewma_sigma_mult) if ewma_sigma_mult is not None else _cal_mod_local.EWMA_SIGMA_FACTOR
        spread = math.sqrt(lam / (2.0 - lam))
        cal.ewma_ucl = +smult * sigma * spread
        cal.ewma_lcl = -smult * sigma * spread


async def flush_all_states(satellite_id: str | None = None) -> None:
    """Persist CUSUM, EWMA, and calibration states to DB.

    Called periodically (every STATE_FLUSH_EVERY cycles) and at shutdown.
    A full flush covers every satellite seen so far; a targeted flush
    covers only the given satellite.
    """
    cusum_states = _cusum_detector.all_states()
    ewma_states  = _ewma_detector.all_states()

    records: list[dict] = []
    for key, data in cusum_states.items():
        sat, _, param = key.partition(":")
        if satellite_id and sat != satellite_id:
            continue
        records.append({
            "satellite_id":  sat,
            "parameter":     param,
            "detector_name": "cusum",
            "state_data":    data,
        })

    for key, data in ewma_states.items():
        sat, _, param = key.partition(":")
        if satellite_id and sat != satellite_id:
            continue
        records.append({
            "satellite_id":  sat,
            "parameter":     param,
            "detector_name": "ewma",
            "state_data":    data,
        })

    if records:
        try:
            await queries.bulk_upsert_detector_states(records)
            logger.debug("detector_states_flushed", count=len(records))
        except Exception as exc:
            logger.warning("detector_state_flush_failed", error=str(exc))

    # Persist calibration states for newly-calibrated channels.
    for key, cal in _calibration_mgr.all_db_records():
        sat, _, param = key.partition(":")
        if satellite_id and sat != satellite_id:
            continue
        try:
            await queries.upsert_channel_calibration(
                satellite_id=sat,
                parameter=param,
                state=cal["state"],
                ref_mean=cal.get("ref_mean"),
                ref_std=cal.get("ref_std"),
                ref_sample_count=cal.get("ref_sample_count", 0),
            )
        except Exception as exc:
            logger.warning("calibration_persist_failed", key=key, error=str(exc))


# ── Bulk history analysis (direct, no REST API) ──────────────────────────────

async def analyze_channel_history(
    satellite_id: str,
    parameter: str,
    subsystem: str = "",
    batch_size: int = 600,
    progress_cb: "object | None" = None,
) -> list[Anomaly]:
    """Stream ALL stored telemetry for one channel through the detection pipeline.

    Designed for bulk historical analysis (ESA benchmark, backfill jobs).
    Fetches data from the DB in chronological batches, maintains a rolling
    600-row context window for STL decomposition, and feeds every point
    through calibration → CUSUM → EWMA exactly once.

    Anomalies are emitted whenever the ensemble vote fires at any point
    in the batch (not just the last point), enabling precise timestamps
    for historical anomaly events.

    Args:
        satellite_id: Satellite identifier (e.g. "ESA-MISSION1").
        parameter: Channel name (e.g. "channel_12").
        subsystem: Subsystem label for anomaly records.
        batch_size: Rows fetched per DB round-trip.  600 matches the STL
                    context window, so each batch fills the window exactly.
        progress_cb: Optional callable(n_processed) for progress reporting.

    Returns:
        List of Anomaly objects, each with a precise historical timestamp.
    """
    key       = f"{satellite_id}:{parameter}"
    anomalies: list[Anomaly] = []

    # Load per-channel overrides once for the entire analysis run.
    eff = get_effective_thresholds(satellite_id, parameter)

    # Rolling context window: stores the last ≤600 (ts_epoch, value) tuples
    # so STL always has enough history.
    ctx_ts:  list[float] = []
    ctx_val: list[float] = []

    after_ts    = None
    n_processed = 0

    while True:
        batch = await queries.get_telemetry_batch_ordered(
            satellite_id, parameter, after_ts=after_ts, limit=batch_size,
        )
        if not batch:
            break

        # Append new rows to rolling context window.
        for r in batch:
            t = r["timestamp"].timestamp() if hasattr(r["timestamp"], "timestamp") \
                else float(r["timestamp"])
            ctx_ts.append(t)
            ctx_val.append(float(r["value"]))

        # Keep only the last 600 to bound memory.
        if len(ctx_ts) > 600:
            ctx_ts  = ctx_ts[-600:]
            ctx_val = ctx_val[-600:]

        wt = np.array(ctx_ts,  dtype=np.float64)
        wv = np.array(ctx_val, dtype=np.float64)

        if len(wt) < 10:
            after_ts = batch[-1]["timestamp"]
            n_processed += len(batch)
            if progress_cb:
                progress_cb(n_processed)
            continue

        # STL decompose the current window (cached; recomputes every 30 calls).
        decomp    = _stl_decomposer.decompose(key, wv, wt)
        residuals = decomp.residual

        # New indices: points from the current batch that haven't been fed
        # to the state machines yet.
        last_ts     = _last_processed_ts.get(key, 0.0)
        new_indices = np.where(wt > last_ts)[0]

        if len(new_indices) == 0:
            after_ts = batch[-1]["timestamp"]
            n_processed += len(batch)
            if progress_cb:
                progress_cb(n_processed)
            continue

        # ── Advance stateful detectors + check for anomalies at each point ──
        # For historical analysis we emit an anomaly at EVERY point that
        # triggers the ensemble, not just the last one in the batch.
        # Z-score and changepoint are window-based (not per-sample) so they
        # run only on the final new point to save CPU.
        best_cusum = DetectorResult(
            detector_name="cusum", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "warming_up"},
        )
        best_ewma = DetectorResult(
            detector_name="ewma", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "warming_up"},
        )
        alarm_idx: int | None = None  # window index of the highest-confidence alarm

        for idx in new_indices:
            res         = float(residuals[idx])
            calibration = _calibration_mgr.update(key, res)
            _apply_calibration_overrides(calibration, eff)   # Point 1: CUSUM/EWMA overrides
            cr          = _cusum_detector.detect(key, res, calibration)
            er          = _ewma_detector.detect(key, res, calibration)

            if cr.score > best_cusum.score:
                best_cusum = cr
                alarm_idx  = idx
            if er.score > best_ewma.score:
                best_ewma = er
                alarm_idx = idx

        _last_processed_ts[key] = float(wt[new_indices[-1]])

        # Window-based detectors run on the final new point only.
        final_idx      = new_indices[-1]
        final_residual = float(residuals[final_idx])
        final_ts_epoch = float(wt[final_idx])

        feat_res = _feature_engine.compute(f"{parameter}:res", final_residual, final_ts_epoch)
        # Point 2: temporarily patch z_threshold with per-channel override
        _orig_z = _stat_detector.z_threshold
        _stat_detector.z_threshold = eff["z_threshold"]
        try:
            stat_result = _stat_detector.detect(feat_res, residuals)
        finally:
            _stat_detector.z_threshold = _orig_z
        cp_result   = (
            _cp_detector.detect(residuals, parameter)
            if len(residuals) >= 60
            else DetectorResult(
                detector_name="changepoint", is_anomaly=False, score=0.0,
                severity=Severity.NOMINAL, details={"reason": "insufficient_data"},
            )
        )
        _feature_engine.compute(parameter, float(wv[final_idx]), final_ts_epoch)
        iso_result = DetectorResult(
            detector_name="isolation_forest", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "non_standard_parameters"},
        )

        all_results = [best_cusum, best_ewma, stat_result, cp_result, iso_result]
        is_anomaly, confidence, severity = _ensemble_vote(all_results)

        # Point 3: per-channel min_confidence gate
        if is_anomaly and eff["min_confidence"] > 0.0 and confidence < eff["min_confidence"]:
            is_anomaly = False

        if is_anomaly and alarm_idx is not None:
            # Cooldown guard: suppress repeated alarms for the same channel.
            # Point 4: use per-channel cooldown instead of global.
            alarm_ts_epoch = float(wt[alarm_idx])
            last_alarm = _last_anomaly_ts.get(key, 0.0)
            if alarm_ts_epoch - last_alarm < eff["alert_cooldown_s"]:
                # Still within cooldown window — skip but keep advancing state.
                after_ts    = batch[-1]["timestamp"]
                n_processed += len(batch)
                if progress_cb:
                    progress_cb(n_processed)
                continue
            alarm_ts_dt    = datetime.fromtimestamp(alarm_ts_epoch, tz=timezone.utc)
            alarm_value    = float(wv[alarm_idx])

            # Build a lightweight row dict for the explanation builder.
            alarm_row = {
                "value":     alarm_value,
                "unit":      "",
                "subsystem": subsystem,
                "parameter": parameter,
                "timestamp": alarm_ts_dt,
            }
            explanation  = _build_explanation(parameter, feat_res, all_results, alarm_row, decomp.method)
            contributing = _extract_contributions(all_results)

            anomaly = Anomaly(
                satellite_id=satellite_id,
                timestamp=alarm_ts_dt,
                subsystem=subsystem,
                parameter=parameter,
                value=alarm_value,
                severity=severity,
                confidence=confidence,
                detectors_triggered=tuple(r.detector_name for r in all_results if r.is_anomaly),
                explanation=explanation,
                contributing_params=contributing,
            )
            try:
                await queries.insert_anomaly(anomaly)
                anomalies.append(anomaly)
                _last_anomaly_ts[key] = alarm_ts_epoch   # arm the cooldown timer
                logger.debug(
                    "historical_anomaly_found",
                    parameter=parameter,
                    timestamp=str(alarm_ts_dt)[:19],
                    severity=severity.value,
                    confidence=round(confidence, 3),
                )
            except Exception as exc:
                logger.warning("anomaly_store_failed", parameter=parameter, error=str(exc))

        after_ts    = batch[-1]["timestamp"]
        n_processed += len(batch)
        if progress_cb:
            progress_cb(n_processed)

    return anomalies


# ── Main detection loop ──────────────────────────────────────────────────────

async def run_detection_cycle(satellite_id: str) -> list[Anomaly]:
    """Full detection cycle for one satellite.

    Called by the ingest adapter after every batch of telemetry is stored.
    Processes ALL NEW points per channel (not just the latest), so that
    calibration warm-up and CUSUM/EWMA accumulators advance correctly for
    bulk-loaded or batched telemetry as well as live streaming.
    """
    start    = time.monotonic()
    anomalies: list[Anomaly] = []

    latest = await queries.get_latest_values(satellite_id)
    if not latest:
        return anomalies

    for row in latest:
        param    = row["parameter"]
        value    = float(row["value"])
        ts       = row["timestamp"]

        # ── 1. Fetch window (provides STL context + new-point list) ────────
        window_rows = await queries.get_recent_telemetry_window(
            satellite_id, param, window_size=600
        )
        if len(window_rows) < 10:
            continue

        window_values = np.array([r["value"] for r in window_rows], dtype=np.float64)
        window_ts     = np.array(
            [r["timestamp"].timestamp() if hasattr(r["timestamp"], "timestamp")
             else float(r["timestamp"]) for r in window_rows],
            dtype=np.float64,
        )

        # ── 2. STL decomposition → residuals ────────────────────────────
        key    = f"{satellite_id}:{param}"
        decomp = _stl_decomposer.decompose(key, window_values, window_ts)
        residuals = decomp.residual

        # Load per-channel overrides once per parameter (Point 1–4 below).
        eff = get_effective_thresholds(satellite_id, param)

        # ── 3. Identify NEW points (not yet fed to state machines) ───────
        # _last_processed_ts tracks the epoch of the last residual fed to
        # calibration/CUSUM/EWMA.  Only timestamps strictly after this value
        # are new — ensures each data point updates state exactly once, even
        # when the same window rows appear across consecutive detection cycles.
        last_ts      = _last_processed_ts.get(key, 0.0)
        new_indices  = np.where(window_ts > last_ts)[0]

        if len(new_indices) == 0:
            continue   # entire window already processed in previous cycle

        # ── 4. Advance stateful detectors for every new point in order ────
        # Calibration needs N consecutive residuals before it transitions to
        # "calibrated"; CUSUM and EWMA must see every sample to accumulate
        # correctly.  Running only the latest point (old behaviour) caused
        # warmup to take 100× more real-time batches than expected.
        #
        # IMPORTANT: we track the MAX-SCORING result across all new points,
        # not just the last one.  CUSUM resets its accumulators after each
        # alarm, so the final point in a batch often has a lower score than
        # the alarm peak mid-batch.  Using the peak score ensures that any
        # alarm that fired during the batch is visible to the ensemble vote.
        _nominal_cusum = DetectorResult(
            detector_name="cusum", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "warming_up"},
        )
        _nominal_ewma  = DetectorResult(
            detector_name="ewma", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "warming_up"},
        )
        cusum_result = _nominal_cusum
        ewma_result  = _nominal_ewma
        calibration  = None

        for idx in new_indices:
            res         = float(residuals[idx])
            calibration = _calibration_mgr.update(key, res)
            _apply_calibration_overrides(calibration, eff)   # Point 1: CUSUM/EWMA overrides
            cr          = _cusum_detector.detect(key, res, calibration)
            er          = _ewma_detector.detect(key, res, calibration)
            # Promote to best result seen so far in this batch.
            if cr.score > cusum_result.score:
                cusum_result = cr
            if er.score > ewma_result.score:
                ewma_result = er

        # Mark all new window points as processed.
        _last_processed_ts[key] = float(window_ts[new_indices[-1]])

        # ── 5. Z-score on the latest new residual (spike detection) ──────
        current_residual = float(residuals[new_indices[-1]])
        current_ts_epoch = float(window_ts[new_indices[-1]])

        feat_res = _feature_engine.compute(f"{param}:res", current_residual, current_ts_epoch)
        # Point 2: temporarily patch z_threshold with per-channel override
        _orig_z = _stat_detector.z_threshold
        _stat_detector.z_threshold = eff["z_threshold"]
        try:
            stat_result = _stat_detector.detect(feat_res, residuals)
        finally:
            _stat_detector.z_threshold = _orig_z

        # ── 6. PELT on residuals ─────────────────────────────────────────
        cp_result = (
            _cp_detector.detect(residuals, param)
            if len(residuals) >= 60
            else DetectorResult(
                detector_name="changepoint", is_anomaly=False, score=0.0,
                severity=Severity.NOMINAL, details={"reason": "insufficient_data"},
            )
        )

        # ── 7. Isolation Forest on raw values (multivariate) ────────────
        _feature_engine.compute(param, value, current_ts_epoch)

        known_params    = _feature_engine.get_known_parameters()
        has_std_params  = any(p in known_params for p in _ALL_PARAMETERS)
        iso_result = (
            _detect_isolation_forest(satellite_id)
            if has_std_params
            else DetectorResult(
                detector_name="isolation_forest", is_anomaly=False, score=0.0,
                severity=Severity.NOMINAL, details={"reason": "non_standard_parameters"},
            )
        )

        # ── 8. Ensemble vote ─────────────────────────────────────────────
        all_results = [cusum_result, ewma_result, stat_result, cp_result, iso_result]
        is_anomaly, confidence, severity = _ensemble_vote(all_results)

        # Point 3: per-channel min_confidence gate
        if is_anomaly and eff["min_confidence"] > 0.0 and confidence < eff["min_confidence"]:
            is_anomaly = False

        # Cooldown guard: one alarm per channel per cooldown window.
        # Point 4: use per-channel cooldown instead of global _alert_cooldown_s.
        live_key = f"{satellite_id}:{param}"
        if is_anomaly:
            now_epoch = datetime.now(timezone.utc).timestamp()
            if now_epoch - _last_anomaly_ts.get(live_key, 0.0) < eff["alert_cooldown_s"]:
                is_anomaly = False  # still in cooldown — suppress

        if is_anomaly:
            _last_anomaly_ts[live_key] = datetime.now(timezone.utc).timestamp()
            explanation  = _build_explanation(param, feat_res, all_results, row, decomp.method)
            contributing = _extract_contributions(all_results)

            anomaly = Anomaly(
                satellite_id=satellite_id,
                timestamp=ts,
                subsystem=row.get("subsystem", ""),
                parameter=param,
                value=value,
                severity=severity,
                confidence=confidence,
                detectors_triggered=tuple(r.detector_name for r in all_results if r.is_anomaly),
                explanation=explanation,
                contributing_params=contributing,
            )
            # Attach the STL residual for storage in the new DB column.
            object.__setattr__(anomaly, "stl_residual", current_residual) if hasattr(anomaly, "__dataclass_fields__") else None

            try:
                await queries.insert_anomaly(anomaly)
                anomalies.append(anomaly)

                from sentinel.api.websocket import broadcast_anomaly
                await broadcast_anomaly({
                    "id":          anomaly.id,
                    "satellite_id": anomaly.satellite_id,
                    "parameter":   anomaly.parameter,
                    "value":       anomaly.value,
                    "severity":    anomaly.severity.value,
                    "confidence":  anomaly.confidence,
                    "explanation": anomaly.explanation,
                    "timestamp":   anomaly.timestamp.isoformat() if isinstance(anomaly.timestamp, datetime) else str(anomaly.timestamp),
                })

                # Dispatch alert (webhook / email) for WARNING and CRITICAL only.
                # WATCH-level anomalies are informational — no pager alert.
                from sentinel.alerts.service import get_alert_service
                _svc = get_alert_service()
                if _svc is not None:
                    await _svc.process_anomaly(anomaly)
            except Exception as exc:
                logger.error("anomaly_store_failed", error=str(exc), parameter=param)

    # ── Periodic tasks ───────────────────────────────────────────────────
    count = _detection_cycle_count.get(satellite_id, 0) + 1
    _detection_cycle_count[satellite_id] = count

    # Isolation Forest refit
    sat_samples = _samples_since_fit.get(satellite_id, 0) + len(latest)
    _samples_since_fit[satellite_id] = sat_samples
    if _iso_detector.needs_refit(sat_samples):
        await _refit_isolation_forest(satellite_id)
        _samples_since_fit[satellite_id] = 0

    # State flush to DB
    if count % _STATE_FLUSH_EVERY == 0:
        await flush_all_states(satellite_id=satellite_id)

    elapsed_ms = (time.monotonic() - start) * 1000
    if anomalies:
        logger.info(
            "detection_cycle_complete",
            satellite=satellite_id,
            anomalies_found=len(anomalies),
            elapsed_ms=round(elapsed_ms, 1),
        )

    return anomalies


# ── Ensemble voting ──────────────────────────────────────────────────────────

def _ensemble_vote(
    results: list[DetectorResult],
) -> tuple[bool, float, Severity]:
    """Combine detector outputs into a single verdict.

    Confidence is normalised over TRIGGERED detectors only, so a single
    strong detector (e.g. CUSUM alone) can still reach high confidence
    instead of being diluted by the 4 non-triggering detectors scoring 0.

    Agreement factor:
        1/5 triggered → ×0.60
        2/5 triggered → ×0.75
        3/5 triggered → ×0.88
        4/5 triggered → ×0.95
        5/5 triggered → ×1.00
    """
    triggered = [r for r in results if r.is_anomaly]
    if not triggered:
        # No alarm — return weighted average of sub-threshold scores.
        avg = sum(r.score * WEIGHTS.get(r.detector_name, 0.2) for r in results)
        return False, float(avg), Severity.NOMINAL

    # Weighted confidence over triggered detectors only.
    trigger_weight_sum = sum(WEIGHTS.get(r.detector_name, 0.2) for r in triggered)
    if trigger_weight_sum < 1e-9:
        return False, 0.0, Severity.NOMINAL

    signal_score = sum(
        r.score * WEIGHTS.get(r.detector_name, 0.2) for r in triggered
    ) / trigger_weight_sum

    # Agreement factor — logarithmic so 2 detectors is a meaningful jump.
    n_total     = len(results)
    n_triggered = len(triggered)
    agreement   = 0.60 + 0.40 * (n_triggered - 1) / max(n_total - 1, 1)

    confidence = min(1.0, signal_score * agreement)

    # Severity gate — from ensemble confidence only.
    if confidence >= _severity_thresholds["critical"]:
        severity = Severity.CRITICAL
    elif confidence >= _severity_thresholds["warning"]:
        severity = Severity.WARNING
    elif confidence >= _severity_thresholds["watch"]:
        severity = Severity.WATCH
    else:
        # Confidence too low — discard.
        return False, confidence, Severity.NOMINAL

    return True, confidence, severity


# ── Isolation Forest helpers ─────────────────────────────────────────────────

def _detect_isolation_forest(satellite_id: str) -> DetectorResult:
    if not _iso_detector.is_ready:
        return DetectorResult(
            detector_name="isolation_forest", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "model_not_fitted"},
        )
    snapshot = _feature_engine.get_multivariate_snapshot(_ALL_PARAMETERS)
    if snapshot is None:
        return DetectorResult(
            detector_name="isolation_forest", is_anomaly=False, score=0.0,
            severity=Severity.NOMINAL, details={"reason": "incomplete_data"},
        )
    return _iso_detector.detect(snapshot)


async def _refit_isolation_forest(satellite_id: str) -> None:
    matrix = _feature_engine.get_window_matrix(_ALL_PARAMETERS, length=200)
    if matrix is not None:
        _iso_detector.fit(matrix, _ALL_PARAMETERS)


# ── Explanation + contribution ────────────────────────────────────────────────

def _build_explanation(
    parameter: str,
    features,
    results: list[DetectorResult],
    row: dict,
    decomp_method: str,
) -> str:
    """Human-readable explanation of why this anomaly fired."""
    triggered = [r for r in results if r.is_anomaly]
    parts: list[str] = []

    parts.append(
        f"{parameter} = {row['value']:.4f} {row.get('unit', '')} "
        f"(residual: {features.raw_value:.4f}, "
        f"rolling_std: {features.rolling_std:.4f}, "
        f"decomp: {decomp_method})"
    )

    for r in triggered:
        match r.detector_name:
            case "cusum":
                d = r.details
                direction = d.get("direction", "?")
                parts.append(
                    f"CUSUM {direction} drift: S={max(d.get('s_pos',0), d.get('s_neg',0)):.3f} "
                    f"(H={d.get('h',0):.3f}, k={d.get('k',0):.4f}, "
                    f"alarm#{d.get('alarm_count',1)})"
                )
            case "ewma":
                d = r.details
                parts.append(
                    f"EWMA level shift: Z={d.get('z_ewma',0):.4f} "
                    f"vs UCL={d.get('ucl',0):.4f} / LCL={d.get('lcl',0):.4f}"
                )
            case "statistical":
                z = r.details.get("z_score", 0)
                parts.append(f"Z-score spike: {z:.2f}σ (threshold: {r.details.get('threshold', 3.0)})")
                if r.details.get("rate_of_change_anomaly"):
                    parts.append("Rapid rate-of-change detected")
            case "changepoint":
                cps = r.details.get("change_points", [])
                if cps:
                    top = max(cps, key=lambda c: c.get("score", 0))
                    parts.append(
                        f"Structural break detected "
                        f"(mean_shift={top.get('mean_shift',0):.4f}, "
                        f"recency={top.get('recency',0):.2f})"
                    )
            case "isolation_forest":
                contribs = r.details.get("feature_contributions", {})
                if contribs:
                    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    parts.append(
                        "Cross-parameter anomaly — top contributors: "
                        + ", ".join(f"{k}:{v:+.3f}" for k, v in top)
                    )

    parts.append(f"{len(triggered)}/{len(results)} detectors triggered")
    return " | ".join(parts)


def _extract_contributions(results: list[DetectorResult]) -> dict[str, float]:
    for r in results:
        if r.detector_name == "isolation_forest" and r.details.get("feature_contributions"):
            return r.details["feature_contributions"]
    return {}
