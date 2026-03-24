"""Detection pipeline orchestrator — the brain of Dsremo.

Detection order per parameter (post-STL architecture):
    1. Fetch recent window from DB (600 pts, oldest → newest)
    2. STL decompose → residuals + trend
    3. Update per-channel calibration (ref_mean, ref_std, k, H, UCL, LCL)
    4. CUSUM           on residuals  — gradual drift accumulation (NASA standard)
    5. EWMA            on residuals  — sudden level shifts
    6. Z-score         on residuals  — single-point spikes
    7. PELT            on residuals  — abrupt structural breaks
    8. Isolation Forest on raw values — multivariate cross-parameter (≥2 params)
    9. Variance         on residuals  — variance-spike anomalies (CATS-type)
   10. GRU Autoencoder  on residuals  — temporal pattern ML
   11. TCN              on residuals  — dilated causal convolution ML
   12. Trend Velocity   on STL trend  — onset detection (drift acceleration)
   13. Ensemble vote → confidence + severity
   14. Store anomaly, broadcast via WebSocket

Ensemble weights (9 detectors, sum=1.0):
    cusum:            0.19   (primary drift detector)
    ewma:             0.16   (level shift detector)
    statistical:      0.12   (spike detector)
    changepoint:      0.09   (structural break detector)
    isolation_forest: 0.05   (multivariate, ≥2 parameters required)
    variance:         0.08   (variance-spike anomalies)
    lstm:             0.12   (GRU autoencoder — temporal pattern ML)
    tcn:              0.11   (TCN — dilated causal convolutions)
    trend_velocity:   0.08   (STL trend acceleration — onset detection)

Severity gate:   derived from ensemble confidence only.
    watch   >= 0.50
    warning >= 0.65
    critical>= 0.85
All gates configurable via dsremo.yaml.  Zero hardcoded thresholds.

State persistence:
    CUSUM and EWMA accumulators are flushed to DB every STATE_FLUSH_EVERY
    detection cycles so they survive server restarts.  Calibration states
    are also persisted whenever a channel transitions to "calibrated".
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import numpy as np
import structlog

from dsremo.core.models import Anomaly, DetectorResult, Severity
from dsremo.db import queries
from dsremo.detection.calibration import CalibrationManager
from dsremo.detection.changepoint import ChangePointDetector
from dsremo.detection.cusum import CUSUMDetector
from dsremo.detection.ewma import EWMADetector
from dsremo.detection.isolation import IsolationForestDetector
from dsremo.detection.statistical import StatisticalDetector
from dsremo.detection.stl_decomposer import STLDecomposer
from dsremo.detection.autoencoder_detector import AutoencoderDetector
from dsremo.detection.incident_grouper import IncidentGrouper
from dsremo.detection.tcn_detector import TCNDetector
from dsremo.detection.correlation_detector import CorrelationGraphDetector
from dsremo.detection.discord_detector import DiscordDetector
from dsremo.detection.trend_velocity_detector import TrendVelocityDetector
from dsremo.detection.variance_detector import VarianceDetector
from dsremo.core.tenant import get_tenant
from dsremo.features.engine import FeatureEngine

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
_variance_detector:        VarianceDetector         = VarianceDetector()
_trend_velocity_detector:  TrendVelocityDetector    = TrendVelocityDetector()
_discord_detector:         DiscordDetector          = DiscordDetector()
_correlation_detector:     CorrelationGraphDetector = CorrelationGraphDetector()
_incident_grouper:         IncidentGrouper          = IncidentGrouper()

# Directory for persisted ML model checkpoints (warm-start across runs).
# None = persistence disabled (default until configured via init_detectors).
_model_dir: "Path | None" = None  # type: ignore[type-arg]


def _model_path(satellite_id: str, parameter: str, ext: str) -> "Path | None":
    """Return the checkpoint path for a persisted ML model, or None if disabled.

    Args:
        ext: File extension suffix, e.g. "lstm" or "tcn" → "{param}.{ext}.pt"
    """
    from pathlib import Path  # noqa: PLC0415
    if _model_dir is None:
        return None
    safe_sat   = satellite_id.replace("/", "_").replace(":", "_")
    safe_param = parameter.replace("/", "_").replace(":", "_")
    return Path(_model_dir) / safe_sat / f"{safe_param}.{ext}.pt"


# Backward-compatible aliases kept for existing tests (Sprint 16).
def _lstm_model_path(satellite_id: str, parameter: str) -> "Path | None":
    return _model_path(satellite_id, parameter, "lstm")


def _tcn_model_path(satellite_id: str, parameter: str) -> "Path | None":
    return _model_path(satellite_id, parameter, "tcn")


# Per-channel GRU autoencoder models — keyed by "satellite_id:parameter".
# Each AutoencoderDetector accumulates residuals, self-trains, and scores.
_lstm_models: dict[str, AutoencoderDetector] = {}

# Hard cap on in-memory ML model registry size.  When either registry grows
# past this limit, the oldest-inserted entry is evicted (FIFO — Python dicts
# preserve insertion order since 3.7).  At ~100 KB per model, 200 models ≈ 20 MB.
_MAX_ML_MODELS: int = 200

# LSTM config — overridable via init_detectors() from dsremo.yaml.
_lstm_seq_length:      int   = 30
_lstm_hidden_size:     int   = 32
_lstm_bottleneck_size: int   = 8
_lstm_epochs:          int   = 30
_lstm_min_train:       int   = 60
_lstm_retrain_interval: int  = 500
_lstm_threshold_sigma: float = 3.0

# Each TCNDetector accumulates residuals, self-trains, and scores.
_tcn_models: dict[str, TCNDetector] = {}

# TCN config — overridable via init_detectors() from dsremo.yaml.
_tcn_seq_length:       int   = 32
_tcn_n_channels:       int   = 16
_tcn_n_blocks:         int   = 4
_tcn_kernel_size:      int   = 3
_tcn_epochs:           int   = 40
_tcn_min_train:        int   = 64
_tcn_retrain_interval: int   = 500
_tcn_threshold_sigma:  float = 3.0

# TrendVelocityDetector config — overridable via init_detectors() from dsremo.yaml.
_tvel_window:          int   = 20
_tvel_recent_points:   int   = 5
_tvel_threshold_sigma: float = 3.0

# DiscordDetector config — overridable via init_detectors() from dsremo.yaml.
_discord_m:                int   = 20
_discord_window:           int   = 300
_discord_threshold_sigma:  float = 3.0

# CorrelationGraphDetector config — overridable via init_detectors() from dsremo.yaml.
_corr_graph_window:            int   = 60
_corr_graph_min_calibration:   int   = 100
_corr_graph_threshold_sigma:   float = 3.0

# Alert suppression windows (maintenance mode, Sprint 19).
# key = "satellite_id:parameter" → UTC epoch when suppression expires.
# In-memory only — cleared on server restart (by design; maintenance windows
# should not silently persist across deployments).
_suppressed: dict[str, float] = {}

# Ensemble weights (must sum to 1.0, overridable via config)
# 11 detectors: cusum, ewma, statistical, changepoint, isolation_forest, variance,
#               lstm, tcn, trend_velocity, matrix_profile, correlation_graph
WEIGHTS: dict[str, float] = {
    "cusum":              0.17,   # primary drift detector (NASA CUSUM standard)
    "ewma":               0.14,   # level-shift detector
    "statistical":        0.10,   # single-point spike detector (z-score)
    "changepoint":        0.08,   # structural break detector (PELT)
    "isolation_forest":   0.05,   # multivariate cross-parameter anomalies
    "variance":           0.07,   # variance-spike detector (CATS-type oscillatory signals)
    "lstm":               0.10,   # GRU autoencoder — temporal pattern anomalies (ML)
    "tcn":                0.09,   # TCN — dilated causal convolutions, deeper ML patterns
    "trend_velocity":     0.08,   # STL trend acceleration — onset detection (Sprint 14)
    "matrix_profile":     0.06,   # Matrix Profile discord — shape anomaly detection (Sprint 15)
    "correlation_graph":  0.06,   # Correlation graph — relationship breakdown (Sprint 19)
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

# Per-channel detector weights — learned from operator FP feedback.
# Keys: "{tenant}:{satellite_id}:{parameter}" → {detector_name: weight}.
# When present, used instead of global WEIGHTS for that channel.
# Decay rule: each FP verdict decays triggering detectors by FP_WEIGHT_DECAY.
# Weights are re-normalised to sum=1.0 after each update.
_channel_weights: dict[str, dict[str, float]] = {}
_FP_WEIGHT_DECAY: float = 0.95   # 5% per false-positive verdict

# State flush to DB: persist CUSUM/EWMA/calibration every N cycles.
_STATE_FLUSH_EVERY: int = 50
_detection_cycle_count: dict[str, int] = {}

# Track samples since last Isolation Forest refit per satellite.
_samples_since_fit: dict[str, int] = {}

# Per-channel last-processed timestamp (epoch seconds).
# Each detection cycle only feeds NEW residuals into the state machines
# (calibration, CUSUM, EWMA) — points with timestamp <= this value were
# already processed in a previous cycle and must not be fed again.
# Capped at _MAX_CHANNEL_STATE entries; FIFO eviction keeps memory bounded
# in long-running deployments with many transient channels.
_last_processed_ts: dict[str, float] = {}
_MAX_CHANNEL_STATE: int = 10_000   # max tracked channels before FIFO eviction

# Alert cooldown: suppress repeated alarms for the same channel within N hours.
# Prevents a sustained anomalous regime from generating thousands of records.
# Configurable via dsremo.yaml detection.alert_cooldown_hours.
_alert_cooldown_s: float = 72.0 * 3600.0   # default 72 h; overridden by init
_last_anomaly_ts:  dict[str, float] = {}    # key → epoch of last stored anomaly

# Persistence filter: require N consecutive anomalous detections before alerting.
# Production standard (NASA ASIST, YAMCS, SpaceX Doppel) — avoids single-sample
# FPs without needing a data scan.  1 = disabled (backward-compatible default).
# Applied only in run_detection_cycle() (streaming); analyze_channel_history()
# uses the cooldown guard instead (batch analysis).
_alert_persistence_min: int = 1             # set to 2-3 for production streaming
_anomaly_streak:  dict[str, int] = {}       # key → consecutive anomalous windows

# Stale data detection (NASA ASIST / ISRO FOLIO standard, Sprint 18).
# Tracks when each channel last produced a telemetry value.  If the gap
# exceeds stale_threshold_s, a CRITICAL "stale_data" anomaly is raised.
# This catches sensor dropout, instrument power-off, and comms loss —
# scenarios our 10-detector ensemble is completely blind to.
_channel_last_seen:  dict[str, float] = {}  # key=sat:param → UTC epoch
_stale_threshold_s:  float = 300.0          # default 5 min; overridden by config
_ttl_warn_min:       float = 60.0           # default 60 min; warn when ttl < this

# Adaptive context window: auto-scaled per channel based on FFT-detected period.
# window = min(max(600, stl_window_factor × period), stl_max_window)
_stl_window_factor: int = 3      # context_window = factor × detected_period
_stl_max_window:    int = 10000  # absolute upper bound on context window (memory guard)

# Per-channel threshold overrides — keyed by (satellite_id, parameter).
# NULL/missing fields fall through to global defaults.
# Refreshed at startup and after any PUT/DELETE via load_channel_configs().
_channel_config_cache: dict[tuple[str, str], dict] = {}


# ── Initialisation ───────────────────────────────────────────────────────────

def _get_ml_model(satellite_id: str, parameter: str, registry: dict, factory, ext: str):  # type: ignore[no-untyped-def]
    """Lazy-init + checkpoint warm-start shared by all per-channel ML detectors.

    Identical pattern for GRU and TCN: check registry → create via factory → load checkpoint.
    Evicts the oldest entry (FIFO) when registry exceeds _MAX_ML_MODELS to bound memory.
    """
    key = f"{satellite_id}:{parameter}"
    if key not in registry:
        # Evict oldest if at capacity (FIFO — dict insertion order, Python 3.7+).
        if len(registry) >= _MAX_ML_MODELS:
            oldest_key = next(iter(registry))
            del registry[oldest_key]
        det = factory()
        path = _model_path(satellite_id, parameter, ext)
        if path is not None and path.exists():
            det.load(path)
        registry[key] = det
    return registry[key]


def _get_lstm_model(satellite_id: str, parameter: str) -> AutoencoderDetector:
    return _get_ml_model(
        satellite_id, parameter,
        _lstm_models,
        lambda: AutoencoderDetector(
            seq_length=_lstm_seq_length,
            hidden_size=_lstm_hidden_size,
            bottleneck_size=_lstm_bottleneck_size,
            epochs=_lstm_epochs,
            min_train_samples=_lstm_min_train,
            retrain_interval=_lstm_retrain_interval,
            threshold_sigma=_lstm_threshold_sigma,
        ),
        "lstm",
    )


def _get_tcn_model(satellite_id: str, parameter: str) -> TCNDetector:
    return _get_ml_model(
        satellite_id, parameter,
        _tcn_models,
        lambda: TCNDetector(
            seq_length=_tcn_seq_length,
            n_channels=_tcn_n_channels,
            n_blocks=_tcn_n_blocks,
            kernel_size=_tcn_kernel_size,
            epochs=_tcn_epochs,
            min_train_samples=_tcn_min_train,
            retrain_interval=_tcn_retrain_interval,
            threshold_sigma=_tcn_threshold_sigma,
        ),
        "tcn",
    )


def get_incident_grouper() -> IncidentGrouper:
    """Return the singleton IncidentGrouper for testing / API access."""
    return _incident_grouper


def save_channel_models(satellite_id: str, parameter: str) -> None:
    """Persist GRU and TCN checkpoints for one channel to disk.

    Called after each channel finishes in run_bulk_detection() so that
    the next run can warm-start without retraining from scratch.
    No-op when model_dir is not configured or models are not yet fitted.
    """
    key = f"{satellite_id}:{parameter}"
    for ext, registry in (("lstm", _lstm_models), ("tcn", _tcn_models)):
        path = _model_path(satellite_id, parameter, ext)
        if path is not None and key in registry:
            registry[key].save(path)


def init_detectors(settings: object) -> None:
    """Wire config values into all detector singletons.  Call once at startup."""
    global _feature_engine, _stl_decomposer, _calibration_mgr
    global _cusum_detector, _ewma_detector, _stat_detector
    global _iso_detector, _cp_detector, _variance_detector
    global WEIGHTS, _severity_thresholds
    global _last_processed_ts, _detection_cycle_count, _samples_since_fit
    global _alert_cooldown_s, _last_anomaly_ts
    global _alert_persistence_min, _anomaly_streak
    global _stl_window_factor, _stl_max_window
    global _lstm_seq_length, _lstm_hidden_size, _lstm_bottleneck_size
    global _lstm_epochs, _lstm_min_train, _lstm_retrain_interval, _lstm_threshold_sigma
    global _lstm_models
    global _tcn_seq_length, _tcn_n_channels, _tcn_n_blocks, _tcn_kernel_size
    global _tcn_epochs, _tcn_min_train, _tcn_retrain_interval, _tcn_threshold_sigma
    global _tcn_models
    global _tvel_window, _tvel_recent_points, _tvel_threshold_sigma
    global _trend_velocity_detector
    global _discord_m, _discord_window, _discord_threshold_sigma
    global _discord_detector
    global _corr_graph_window, _corr_graph_min_calibration, _corr_graph_threshold_sigma
    global _correlation_detector
    global _suppressed
    global _model_dir
    global _stale_threshold_s, _ttl_warn_min
    global _incident_grouper

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

    # ── STL decomposer + adaptive context window ──────────────────────────
    orbital_period_s  = int(feat.get("orbital_period", 5400))
    _stl_window_factor = int(det.get("stl_window_factor", 3))
    _stl_max_window    = int(det.get("stl_max_window", 10000))
    _stl_decomposer    = STLDecomposer(
        orbital_period_s=orbital_period_s,
        recompute_every=int(det.get("stl_recompute_every", 30)),
        max_fft_samples=int(det.get("stl_max_fft_samples", 600)),
    )

    # ── Calibration (shared params read by CalibrationManager internals) ──
    import dsremo.detection.calibration as _cal_mod
    _cal_mod.CALIBRATION_WINDOW    = int(det.get("calibration_window", 100))
    _cal_mod.CUSUM_K_FACTOR        = float(det.get("cusum_k_factor", 0.5))
    _cal_mod.CUSUM_H_FACTOR        = float(det.get("cusum_h_factor", 5.0))
    _cal_mod.EWMA_LAMBDA           = float(det.get("ewma_lambda", 0.2))
    _cal_mod.EWMA_SIGMA_FACTOR     = float(det.get("ewma_sigma_factor", 3.0))
    _cal_mod.RECAL_FACTOR          = float(det.get("cusum_recal_factor", 10.0))
    _cal_mod.SIGMA_UPDATE_INTERVAL    = int(det.get("sigma_update_interval", 720))
    _cal_mod.SIGMA_UPDATE_THRESHOLD   = float(det.get("sigma_update_threshold", 0.10))
    _cal_mod.GMM_ENABLED              = bool(det.get("gmm_enabled", True))
    _cal_mod.AUTO_Z_THRESHOLD_ENABLED = bool(det.get("auto_z_threshold_enabled", True))
    # Recompute the spread constant in calibration module after lambda change.
    import math
    lam = _cal_mod.EWMA_LAMBDA
    _cal_mod._ewma_spread = math.sqrt(lam / (2.0 - lam))

    _calibration_mgr = CalibrationManager()

    # ── EWMA: update lambda from config ───────────────────────────────────
    _ewma_detector = EWMADetector(lam=float(det.get("ewma_lambda", 0.2)))

    # ── CUSUM / EWMA ──────────────────────────────────────────────────────
    _cusum_detector = CUSUMDetector()

    # ── Variance detector ─────────────────────────────────────────────────
    _variance_detector = VarianceDetector(
        variance_z_threshold=float(det.get("variance_z_threshold", 2.5)),
        window=int(det.get("variance_window", 30)),
    )

    # ── GRU Autoencoder (ML detector, Sprint 10) ──────────────────────────
    _lstm_seq_length       = int(det.get("lstm_seq_length",        30))
    _lstm_hidden_size      = int(det.get("lstm_hidden_size",       32))
    _lstm_bottleneck_size  = int(det.get("lstm_bottleneck_size",    8))
    _lstm_epochs           = int(det.get("lstm_epochs",            30))
    _lstm_min_train        = int(det.get("lstm_min_train_samples", 60))
    _lstm_retrain_interval = int(det.get("lstm_retrain_interval",  500))
    _lstm_threshold_sigma  = float(det.get("lstm_threshold_sigma", 3.0))
    _lstm_models.clear()   # drop stale per-channel models on re-init

    # ── TCN Detector (ML detector, Sprint 13) ────────────────────────────────
    _tcn_seq_length       = int(det.get("tcn_seq_length",        32))
    _tcn_n_channels       = int(det.get("tcn_n_channels",        16))
    _tcn_n_blocks         = int(det.get("tcn_n_blocks",           4))
    _tcn_kernel_size      = int(det.get("tcn_kernel_size",        3))
    _tcn_epochs           = int(det.get("tcn_epochs",            40))
    _tcn_min_train        = int(det.get("tcn_min_train_samples", 64))
    _tcn_retrain_interval = int(det.get("tcn_retrain_interval",  500))
    _tcn_threshold_sigma  = float(det.get("tcn_threshold_sigma", 3.0))
    _tcn_models.clear()   # drop stale per-channel models on re-init

    # ── TrendVelocityDetector (Sprint 14) ─────────────────────────────────
    _tvel_window           = int(det.get("tvel_window",           20))
    _tvel_recent_points    = int(det.get("tvel_recent_points",     5))
    _tvel_threshold_sigma  = float(det.get("tvel_threshold_sigma", 3.0))
    _trend_velocity_detector = TrendVelocityDetector(
        window=_tvel_window,
        recent_points=_tvel_recent_points,
        threshold_sigma=_tvel_threshold_sigma,
    )

    # ── DiscordDetector (Sprint 15) ───────────────────────────────────────
    _discord_m                = int(det.get("matrix_profile_m",     20))
    _discord_window           = int(det.get("matrix_profile_buffer", 300))
    _discord_threshold_sigma  = float(det.get("matrix_profile_sigma", 3.0))
    _discord_detector = DiscordDetector(
        m=_discord_m,
        window=_discord_window,
        threshold_sigma=_discord_threshold_sigma,
    )

    # ── Model persistence directory (warm-start across runs) ─────────────
    raw_model_dir = det.get("model_dir", None)
    if raw_model_dir:
        import os                   # noqa: PLC0415
        from pathlib import Path    # noqa: PLC0415
        _model_dir = Path(os.path.expanduser(str(raw_model_dir)))
    else:
        _model_dir = None

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

    # Persistence filter: require N consecutive anomalous windows before alerting.
    # 1 = disabled (default, backward-compatible). Set to 2-3 for production streaming.
    _alert_persistence_min = int(det.get("alert_persistence_min", 1))

    # ── Incident grouper ──────────────────────────────────────────────────
    incident_window_s     = float(det.get("incident_window_s",     300.0))
    incident_close_after_s = float(det.get("incident_close_after_s", 3600.0))
    _incident_grouper = IncidentGrouper(
        window_s=incident_window_s,
        close_after_s=incident_close_after_s,
    )

    # ── CorrelationGraphDetector (Sprint 19) ──────────────────────────────
    _corr_graph_window           = int(det.get("corr_graph_window",           60))
    _corr_graph_min_calibration  = int(det.get("corr_graph_min_calibration",  100))
    _corr_graph_threshold_sigma  = float(det.get("corr_graph_threshold_sigma", 3.0))
    _correlation_detector = CorrelationGraphDetector(
        window=_corr_graph_window,
        min_calibration=_corr_graph_min_calibration,
        threshold_sigma=_corr_graph_threshold_sigma,
    )

    # ── Suppression windows (Sprint 19) ───────────────────────────────────
    # Clear expired suppressions on re-init (active ones survive restart via
    # the public suppress_channel() API which callers must re-call).
    _suppressed.clear()

    # ── Stale data + TTL config (Sprint 18) ───────────────────────────────
    _stale_threshold_s = float(det.get("stale_threshold_s", 300.0))
    _ttl_warn_min      = float(det.get("ttl_warn_min",      60.0))

    # Reset per-channel state so a server restart or re-init starts clean.
    _last_processed_ts.clear()
    _last_anomaly_ts.clear()
    _anomaly_streak.clear()
    _channel_last_seen.clear()
    _detection_cycle_count.clear()
    _samples_since_fit.clear()
    _channel_weights.clear()   # learned FP weights reset on full re-init

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


# ── Small utilities — eliminate repeated boilerplate ─────────────────────────

def _nominal_result(detector_name: str, reason: str) -> DetectorResult:
    """Factory: return a NOMINAL DetectorResult for warming-up / unavailable cases.

    Replaces 15+ verbatim repetitions of the same 5-line DetectorResult block.
    """
    return DetectorResult(
        detector_name=detector_name,
        is_anomaly=False,
        score=0.0,
        severity=Severity.NOMINAL,
        details={"reason": reason},
    )


@contextmanager
def _z_threshold_override(z_threshold: float) -> Generator[None, None, None]:
    """Temporarily set StatisticalDetector z_threshold for one detect() call.

    asyncio is single-threaded — this context manager is race-free.
    Prefer passing z_threshold as a parameter in future StatisticalDetector API.
    """
    orig = _stat_detector.z_threshold
    _stat_detector.z_threshold = z_threshold
    try:
        yield
    finally:
        _stat_detector.z_threshold = orig


def _gmm_nearest_z(
    residual: float,
    gmm_means: "list[float]",
    gmm_stds: "list[float]",
) -> float:
    """Return z-score of *residual* relative to the nearest GMM component.

    For a 2-component GMM, computes z against each component and returns
    the minimum — i.e. the score relative to whichever mode the residual
    is closest to.  This prevents legitimate bimodal oscillations (e.g. a
    channel that alternates between two normal operating states) from
    triggering false alarms relative to the global mean.
    """
    return float(min(
        abs(residual - mu) / max(sigma, 1e-9)
        for mu, sigma in zip(gmm_means, gmm_stds)
    ))


def _advance_ml_detector(
    det: "AutoencoderDetector | TCNDetector",
    new_indices: "np.ndarray",
    residuals: "np.ndarray",
) -> DetectorResult:
    """Feed new residuals into an ML detector, trigger fit if needed, and detect.

    Extracts the identical 6-line pattern duplicated in both run_detection_cycle()
    and analyze_channel_history() for each of GRU and TCN detectors.
    Training data is capped internally by AbstractMLDetector._MAX_TRAIN_SAMPLES.
    """
    for idx in new_indices:
        det.add_sample(float(residuals[idx]))
    if not det.is_fitted and det.sample_count >= det.min_train_samples:
        det.fit()
    elif det.needs_refit():
        det.fit()   # uses internal buffer (already capped at 2000 samples)
    return det.detect(list(residuals))


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


def record_false_positive(
    satellite_id: str,
    parameter: str,
    detectors_triggered: "tuple[str, ...] | list[str]",
) -> None:
    """Decay per-channel weights for detectors that fired on a false positive.

    Called by the feedback API endpoint when an operator marks an anomaly as FP.
    For each triggering detector, its per-channel weight is multiplied by
    _FP_WEIGHT_DECAY (default 0.95).  Weights are then re-normalised to sum=1.0.

    The per-channel weight dict starts as a copy of the global WEIGHTS, so
    channels that have never received feedback use the standard ensemble weights.
    Only detectors that exist in WEIGHTS are updated — unknown detector names
    (e.g. "stale_data", "hard_limit") are silently skipped.
    """
    if not detectors_triggered:
        return
    from dsremo.core.tenant import get_tenant  # noqa: PLC0415
    key = f"{get_tenant()}:{satellite_id}:{parameter}"
    ch_w = _channel_weights.setdefault(key, dict(WEIGHTS))
    for det_name in detectors_triggered:
        if det_name in ch_w:
            ch_w[det_name] *= _FP_WEIGHT_DECAY
    total = sum(ch_w.values())
    if total > 1e-9:
        for k in ch_w:
            ch_w[k] /= total
    logger.info(
        "channel_weights_updated",
        satellite_id=satellite_id,
        parameter=parameter,
        detectors_decayed=list(detectors_triggered),
        top3=sorted(ch_w.items(), key=lambda x: -x[1])[:3],
    )


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
        "z_threshold":         _get("z_threshold",         _stat_detector.z_threshold),
        "cusum_h":             _get("cusum_h",             None),   # None = calibration-computed
        "cusum_k":             _get("cusum_k",             None),   # None = calibration-computed
        "ewma_lambda":         _get("ewma_lambda",         None),   # None = calibration-computed
        "ewma_sigma_mult":     _get("ewma_sigma_mult",     None),   # None = calibration-computed
        "min_confidence":      _get("min_confidence",      0.0),
        "alert_cooldown_s":      _get("alert_cooldown_s",    _alert_cooldown_s),
        "alert_persistence_min": _alert_persistence_min,
        "variance_z_threshold":  _get("variance_z_threshold",  _variance_detector.variance_z_threshold),
        "velocity_threshold":    _get("velocity_threshold",    None),   # None = calibrated dynamically
        "discord_threshold":     _get("discord_threshold",     None),   # None = uses global threshold_sigma
        "hard_limit_high":       _get("hard_limit_high",       None),   # absolute redline (high)
        "hard_limit_low":        _get("hard_limit_low",        None),   # absolute redline (low)
    }


# ── Alert suppression helpers (Sprint 19) ────────────────────────────────────

def suppress_channel(satellite_id: str, parameter: str, duration_min: float) -> float:
    """Mute anomaly alerts for a channel for *duration_min* minutes.

    Returns the UTC epoch timestamp when the suppression expires.
    Overwrites any existing suppression for the same channel.
    """
    until = datetime.now(timezone.utc).timestamp() + duration_min * 60.0
    _suppressed[f"{get_tenant()}:{satellite_id}:{parameter}"] = until
    logger.info("channel_suppressed", satellite=satellite_id, parameter=parameter,
                duration_min=duration_min)
    return until


def lift_suppression(satellite_id: str, parameter: str) -> bool:
    """Remove the suppression window for a channel.

    Returns True if a suppression was active, False if none existed.
    """
    key = f"{get_tenant()}:{satellite_id}:{parameter}"
    if key in _suppressed:
        del _suppressed[key]
        logger.info("suppression_lifted", satellite=satellite_id, parameter=parameter)
        return True
    return False


def list_suppressions(satellite_id: str) -> list[dict]:
    """Return active (non-expired) suppressions for *satellite_id* in the current tenant."""
    tenant = get_tenant()
    now = datetime.now(timezone.utc).timestamp()
    result = []
    expired = []
    for key, until in _suppressed.items():
        # Key format: "tenant:satellite_id:parameter"
        parts = key.split(":", 2)
        if len(parts) != 3:
            continue
        t, sat, param = parts
        if t != tenant or sat != satellite_id:
            continue
        if until > now:
            result.append({
                "parameter":    param,
                "until_epoch":  until,
                "remaining_min": round((until - now) / 60.0, 1),
            })
        else:
            expired.append(key)
    for k in expired:
        del _suppressed[k]
    return result


def _is_suppressed(key: str) -> bool:
    """Return True if the sat:param key has an active suppression window."""
    until = _suppressed.get(key, 0.0)
    if until == 0.0:
        return False
    if datetime.now(timezone.utc).timestamp() < until:
        return True
    del _suppressed[key]   # clean up expired entry
    return False


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
        import dsremo.detection.calibration as _cal_mod_local
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
        # Retry once on transient DB errors (pool timeout, brief disconnect).
        for _attempt in range(2):
            try:
                await queries.bulk_upsert_detector_states(records)
                logger.debug("detector_states_flushed", count=len(records))
                break
            except Exception as exc:
                if _attempt == 0:
                    import asyncio  # noqa: PLC0415
                    logger.warning("detector_state_flush_retrying", error=str(exc))
                    await asyncio.sleep(0.5)
                else:
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
    max_rows: int | None = None,
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
    key       = f"{get_tenant()}:{satellite_id}:{parameter}"
    anomalies: list[Anomaly] = []

    # Load per-channel overrides once for the entire analysis run.
    eff = get_effective_thresholds(satellite_id, parameter)

    # ── Adaptive context window: pre-scan to detect dominant period ───────
    # Query up to max_fft_samples rows to run FFT period detection before the
    # main analysis loop.  This allows auto-scaling the rolling context window
    # for long-period signals (e.g. 24h diurnal cycles in GECCO water quality).
    # For signals with no detectable period, ctx_limit defaults to 600.
    ctx_limit = 600
    pre_scan = await queries.get_telemetry_batch_ordered(
        satellite_id, parameter,
        after_ts=None,
        limit=_stl_decomposer._max_fft_samples,
    )
    if pre_scan and len(pre_scan) >= 8:
        pre_vals = np.array(
            [float(r["value"]) for r in pre_scan], dtype=np.float64
        )
        detected_period = STLDecomposer._fft_period(pre_vals)
        if detected_period > 0:
            candidate  = _stl_window_factor * detected_period
            ctx_limit  = min(max(600, candidate), _stl_max_window)
            if ctx_limit > 600:
                logger.info(
                    "stl_window_auto_scaled",
                    satellite=satellite_id,
                    parameter=parameter,
                    period_samples=detected_period,
                    ctx_limit=ctx_limit,
                )

    # Rolling context window: stores the last ≤ctx_limit (ts_epoch, value) tuples
    # so STL always has enough history.  ctx_limit ≥ 600 always.
    ctx_ts:  list[float] = []
    ctx_val: list[float] = []

    after_ts    = None
    n_processed = 0

    while True:
        if max_rows is not None and n_processed >= max_rows:
            break
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

        # Keep only the last ctx_limit samples to bound memory.
        # ctx_limit is ≥600 and auto-scaled based on detected signal period.
        if len(ctx_ts) > ctx_limit:
            ctx_ts  = ctx_ts[-ctx_limit:]
            ctx_val = ctx_val[-ctx_limit:]

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
        best_cusum: DetectorResult = _nominal_result("cusum", "warming_up")
        best_ewma:  DetectorResult = _nominal_result("ewma",  "warming_up")
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

        if key not in _last_processed_ts and len(_last_processed_ts) >= _MAX_CHANNEL_STATE:
            del _last_processed_ts[next(iter(_last_processed_ts))]
        _last_processed_ts[key] = float(wt[new_indices[-1]])

        # Window-based detectors run on the final new point only.
        final_idx      = new_indices[-1]
        final_residual = float(residuals[final_idx])
        final_ts_epoch = float(wt[final_idx])

        feat_res = _feature_engine.compute(f"{parameter}:res", final_residual, final_ts_epoch)
        with _z_threshold_override(eff["z_threshold"]):
            stat_result = _stat_detector.detect(feat_res, residuals)

        cp_result = (
            _cp_detector.detect(residuals, parameter)
            if len(residuals) >= 60
            else _nominal_result("changepoint", "insufficient_data")
        )

        _feature_engine.compute(parameter, float(wv[final_idx]), final_ts_epoch)
        _hist_known_params = _feature_engine.get_known_parameters()
        iso_result = (
            _detect_isolation_forest(satellite_id, _hist_known_params)
            if len(_hist_known_params) >= 2
            else _nominal_result("isolation_forest", "single_parameter")
        )

        # Variance + remaining calibration-dependent detectors.
        last_cal = _calibration_mgr.get(key)
        calibrated = last_cal is not None and last_cal.is_calibrated
        var_result = (
            _variance_detector.detect(residuals, last_cal, eff.get("variance_z_threshold"))
            if calibrated
            else _nominal_result("variance", "warming_up")
        )

        # ── ML detectors: GRU Autoencoder + TCN ──────────────────────────
        lstm_result = _advance_ml_detector(_get_lstm_model(satellite_id, parameter), new_indices, residuals)
        tcn_result  = _advance_ml_detector(_get_tcn_model(satellite_id, parameter),  new_indices, residuals)

        # ── Trend Velocity (9th detector) ─────────────────────────────────
        hist_tvel_result = (
            _trend_velocity_detector.detect(decomp.trend, last_cal, eff.get("velocity_threshold"))
            if calibrated
            else _nominal_result("trend_velocity", "warming_up")
        )

        # ── Matrix Profile Discord (10th detector) ────────────────────────
        hist_discord_result = (
            _discord_detector.detect(residuals, last_cal, eff.get("discord_threshold"))
            if calibrated
            else _nominal_result("matrix_profile", "warming_up")
        )

        # Correlation graph update for historical analysis
        _correlation_detector.update(satellite_id, parameter, final_residual)
        hist_corr_result = _correlation_detector.detect(satellite_id, parameter)

        all_results = [best_cusum, best_ewma, stat_result, cp_result, iso_result, var_result, lstm_result, tcn_result, hist_tvel_result, hist_discord_result, hist_corr_result]
        is_anomaly, confidence, severity = _ensemble_vote(
            all_results, weights=_channel_weights.get(key),
        )

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
                # Group into incident (NASA/SpaceX hierarchical routing).
                incident = _incident_grouper.process(anomaly)
                await queries.upsert_incident(incident)
                await queries.link_anomaly_to_incident(anomaly.id, incident.id)
                logger.debug(
                    "historical_anomaly_found",
                    parameter=parameter,
                    timestamp=str(alarm_ts_dt)[:19],
                    severity=severity.value,
                    confidence=round(confidence, 3),
                    incident_id=incident.id,
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

    # ── Stale data check (NASA ASIST / ISRO FOLIO standard, Sprint 18) ────
    # For every channel we've seen before, if the latest known timestamp is
    # older than _stale_threshold_s, fire a CRITICAL stale_data anomaly.
    # This catches sensor dropout, instrument power-off, and comms loss.
    now_epoch = datetime.now(timezone.utc).timestamp()
    for row in latest:
        param    = row["parameter"]
        ts_epoch = row["timestamp"].timestamp() if hasattr(row["timestamp"], "timestamp") else float(row["timestamp"])
        key      = f"{get_tenant()}:{satellite_id}:{param}"
        if key in _channel_last_seen and (now_epoch - ts_epoch) > _stale_threshold_s:
            stale_anomaly = Anomaly(
                satellite_id=satellite_id,
                timestamp=row["timestamp"],
                subsystem=row.get("subsystem", ""),
                parameter=param,
                value=float(row["value"]),
                severity=Severity.CRITICAL,
                confidence=0.95,
                detectors_triggered=("stale_data",),
                explanation=(
                    f"Telemetry silence detected: last sample {round((now_epoch - ts_epoch)/60, 1)} min ago "
                    f"(threshold {round(_stale_threshold_s/60, 1)} min). "
                    "Possible sensor dropout, power-off, or comms loss."
                ),
                contributing_params=(),
            )
            try:
                await queries.insert_anomaly(stale_anomaly)
                anomalies.append(stale_anomaly)
                incident = _incident_grouper.process(stale_anomaly)
                await queries.upsert_incident(incident)
                await queries.link_anomaly_to_incident(stale_anomaly.id, incident.id)
            except Exception:  # noqa: BLE001
                pass
        if key not in _channel_last_seen and len(_channel_last_seen) >= _MAX_CHANNEL_STATE:
            del _channel_last_seen[next(iter(_channel_last_seen))]
        _channel_last_seen[key] = ts_epoch

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
        key    = f"{get_tenant()}:{satellite_id}:{param}"
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
        cusum_result: DetectorResult = _nominal_result("cusum", "warming_up")
        ewma_result:  DetectorResult = _nominal_result("ewma",  "warming_up")
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

        # Mark all new window points as processed (FIFO eviction at cap).
        if key not in _last_processed_ts and len(_last_processed_ts) >= _MAX_CHANNEL_STATE:
            del _last_processed_ts[next(iter(_last_processed_ts))]
        _last_processed_ts[key] = float(window_ts[new_indices[-1]])

        # ── 5. Z-score on the latest new residual (spike detection) ──────
        current_residual = float(residuals[new_indices[-1]])
        current_ts_epoch = float(window_ts[new_indices[-1]])

        feat_res = _feature_engine.compute(f"{param}:res", current_residual, current_ts_epoch)

        # ── GMM nearest-component z-score (Feature: bimodal channels) ────
        # When the channel has a GMM baseline, substitute the z-score with
        # the distance to the nearest component so bimodal oscillations
        # (e.g. two operating modes) don't look like infinite-σ spikes.
        if calibration is not None and calibration.gmm_means is not None:
            import dataclasses  # noqa: PLC0415
            feat_res = dataclasses.replace(
                feat_res,
                z_score=_gmm_nearest_z(
                    current_residual,
                    calibration.gmm_means,
                    calibration.gmm_stds,
                ),
            )

        # ── Per-channel auto z-threshold (Feature: empirical 99th-pct) ──
        # Effective threshold = max(global config, per-channel empirical 99th pct).
        # Channels with wider natural distributions automatically get a
        # higher threshold without any manual per-dataset tuning.
        _base_z = eff["z_threshold"]
        _auto_z = (
            calibration.auto_z_threshold
            if calibration is not None and calibration.auto_z_threshold is not None
            else None
        )
        _eff_z = max(_base_z, _auto_z) if _auto_z is not None else _base_z

        with _z_threshold_override(_eff_z):
            stat_result = _stat_detector.detect(feat_res, residuals)

        # ── 6. PELT changepoint on residuals ─────────────────────────────
        cp_result = (
            _cp_detector.detect(residuals, param)
            if len(residuals) >= 60
            else _nominal_result("changepoint", "insufficient_data")
        )

        # ── 7. Isolation Forest on raw values (multivariate) ────────────
        _feature_engine.compute(param, value, current_ts_epoch)
        known_params = _feature_engine.get_known_parameters()
        iso_result = (
            _detect_isolation_forest(satellite_id, known_params)
            if len(known_params) >= 2
            else _nominal_result("isolation_forest", "single_parameter")
        )

        # Calibration-dependent detectors (variance, ML, trend, discord).
        calibrated = calibration is not None and calibration.is_calibrated

        # ── 8. Variance detector ─────────────────────────────────────────
        var_result = (
            _variance_detector.detect(residuals, calibration, eff.get("variance_z_threshold"))
            if calibrated
            else _nominal_result("variance", "warming_up")
        )

        # ── 9+10. GRU Autoencoder + TCN (ML detectors) ───────────────────
        lstm_result = _advance_ml_detector(_get_lstm_model(satellite_id, param), new_indices, residuals)
        tcn_result  = _advance_ml_detector(_get_tcn_model(satellite_id, param),  new_indices, residuals)

        # ── 11. Trend Velocity (STL trend acceleration onset) ────────────
        tvel_result = (
            _trend_velocity_detector.detect(decomp.trend, calibration, eff.get("velocity_threshold"))
            if calibrated
            else _nominal_result("trend_velocity", "warming_up")
        )

        # ── Time-to-Limit prediction (SpaceX/ISRO pattern, Sprint 18) ──────
        # When hard_limit_high / hard_limit_low are configured for a channel,
        # compute how long until the current trend reaches the redline.
        # ttl_min < ttl_warn_min → escalate tvel_result severity.
        # ── Time-to-Limit prediction (SpaceX/ISRO pattern, Sprint 18) ──────
        # When hard limits are configured, compute minutes until the current
        # trend reaches the redline and annotate/escalate tvel_result.
        _hl_high = eff.get("hard_limit_high")
        _hl_low  = eff.get("hard_limit_low")
        _vel     = tvel_result.details.get("max_velocity", 0.0) if tvel_result.details else 0.0
        if _vel != 0.0 and (_hl_high is not None or _hl_low is not None):
            _ttl_min: float | None = None
            if _vel > 1e-8 and _hl_high is not None:
                _remaining = _hl_high - value
                if _remaining > 0:
                    _ttl_min = round(_remaining / _vel / 60.0, 1)
            elif _vel < -1e-8 and _hl_low is not None:
                _remaining = value - _hl_low
                if _remaining > 0:
                    _ttl_min = round(_remaining / abs(_vel) / 60.0, 1)
            if _ttl_min is not None:
                # Always construct a new DetectorResult — never mutate the dict
                # inside a frozen dataclass (even though Python allows it).
                ttl_details = {**tvel_result.details, "ttl_min": _ttl_min}
                if _ttl_min < _ttl_warn_min:
                    sev = Severity.CRITICAL if _ttl_min < _ttl_warn_min / 3 else Severity.WARNING
                    tvel_result = DetectorResult(
                        detector_name="trend_velocity",
                        is_anomaly=True,
                        score=max(tvel_result.score, 0.75),
                        severity=sev,
                        details={**ttl_details, "ttl_triggered": True},
                    )
                else:
                    tvel_result = DetectorResult(
                        detector_name=tvel_result.detector_name,
                        is_anomaly=tvel_result.is_anomaly,
                        score=tvel_result.score,
                        severity=tvel_result.severity,
                        details=ttl_details,
                    )

        # ── 12. Matrix Profile Discord (10th detector) ────────────────────
        discord_result = (
            _discord_detector.detect(residuals, calibration, eff.get("discord_threshold"))
            if calibrated
            else _nominal_result("matrix_profile", "warming_up")
        )

        # ── 13. Correlation Graph (11th detector — relationship breakdown) ─
        # Update buffer with the current residual, then score.
        _correlation_detector.update(satellite_id, param, current_residual)
        corr_result = _correlation_detector.detect(satellite_id, param)

        # ── 14. Hard Limit Override (zero-FN absolute redline check) ──────
        # If value has crossed a hard limit, force CRITICAL regardless of
        # ensemble confidence.  Redlines are absolute — no statistical gate.
        hl_high = eff.get("hard_limit_high")
        hl_low  = eff.get("hard_limit_low")
        hard_limit_breached = (
            (hl_high is not None and value > hl_high) or
            (hl_low  is not None and value < hl_low)
        )

        # ── 15. Ensemble vote ────────────────────────────────────────────
        all_results = [cusum_result, ewma_result, stat_result, cp_result, iso_result,
                       var_result, lstm_result, tcn_result, tvel_result, discord_result,
                       corr_result]
        if hard_limit_breached:
            is_anomaly, confidence, severity = True, 1.0, Severity.CRITICAL
        else:
            is_anomaly, confidence, severity = _ensemble_vote(
                all_results, weights=_channel_weights.get(live_key),
            )

        # Suppression window (maintenance mode — mutes alerts, not detection).
        # Prefix with tenant_id to prevent cross-tenant cooldown bleed.
        live_key = f"{get_tenant()}:{satellite_id}:{param}"
        if is_anomaly and _is_suppressed(live_key):
            is_anomaly = False

        # Point 3: per-channel min_confidence gate
        if is_anomaly and eff["min_confidence"] > 0.0 and confidence < eff["min_confidence"]:
            is_anomaly = False

        # Cooldown guard: one alarm per channel per cooldown window.
        # Point 4: use per-channel cooldown instead of global _alert_cooldown_s.
        if is_anomaly:
            now_epoch = datetime.now(timezone.utc).timestamp()
            if now_epoch - _last_anomaly_ts.get(live_key, 0.0) < eff["alert_cooldown_s"]:
                is_anomaly = False  # still in cooldown — suppress

        # Persistence filter: require N consecutive anomalous detections before
        # alerting (production standard — NASA ASIST, YAMCS, SpaceX Doppel).
        # Streak increments on anomaly, resets on nominal.  Alert only fires when
        # streak reaches _alert_persistence_min.  Cooldown check runs first so a
        # suppressed alarm still counts toward the streak.
        if is_anomaly:
            _anomaly_streak[live_key] = _anomaly_streak.get(live_key, 0) + 1
            if _anomaly_streak[live_key] < _alert_persistence_min:
                is_anomaly = False  # not enough consecutive anomalous windows yet
        else:
            _anomaly_streak[live_key] = 0  # reset streak on any nominal window

        if is_anomaly:
            if live_key not in _last_anomaly_ts and len(_last_anomaly_ts) >= _MAX_CHANNEL_STATE:
                del _last_anomaly_ts[next(iter(_last_anomaly_ts))]
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

                # Group into incident — no raw alert reaches operator uncorrelated.
                incident = _incident_grouper.process(anomaly)
                await queries.upsert_incident(incident)
                await queries.link_anomaly_to_incident(anomaly.id, incident.id)

                from dsremo.api.websocket import broadcast_anomaly
                await broadcast_anomaly({
                    "id":          anomaly.id,
                    "satellite_id": anomaly.satellite_id,
                    "parameter":   anomaly.parameter,
                    "value":       anomaly.value,
                    "severity":    anomaly.severity.value,
                    "confidence":  anomaly.confidence,
                    "explanation": anomaly.explanation,
                    "timestamp":   anomaly.timestamp.isoformat() if isinstance(anomaly.timestamp, datetime) else str(anomaly.timestamp),
                    "incident_id": incident.id,
                })

                # Dispatch alert (webhook / email) for WARNING and CRITICAL only.
                # WATCH-level anomalies are informational — no pager alert.
                from dsremo.alerts.service import AlertService
                from dsremo.core.tenant import get_tenant
                await AlertService.dispatch(anomaly, get_tenant())
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
    weights: "dict[str, float] | None" = None,
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

    weights: per-channel learned weights (from feedback). Falls back to
             global WEIGHTS when None or a detector name is missing.
    """
    w = weights if weights is not None else WEIGHTS
    triggered = [r for r in results if r.is_anomaly]
    if not triggered:
        # No alarm — return weighted average of sub-threshold scores.
        avg = sum(r.score * w.get(r.detector_name, 0.2) for r in results)
        return False, float(avg), Severity.NOMINAL

    # Weighted confidence over triggered detectors only.
    trigger_weight_sum = sum(w.get(r.detector_name, 0.2) for r in triggered)
    if trigger_weight_sum < 1e-9:
        return False, 0.0, Severity.NOMINAL

    signal_score = sum(
        r.score * w.get(r.detector_name, 0.2) for r in triggered
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

def _detect_isolation_forest(satellite_id: str, known_params: "set[str] | None" = None) -> DetectorResult:
    if not _iso_detector.is_ready:
        return _nominal_result("isolation_forest", "model_not_fitted")
    params   = sorted(known_params) if known_params is not None else _ALL_PARAMETERS
    snapshot = _feature_engine.get_multivariate_snapshot(params)
    if snapshot is None:
        return _nominal_result("isolation_forest", "incomplete_data")
    return _iso_detector.detect(snapshot)


async def _refit_isolation_forest(satellite_id: str) -> None:
    known_params = _feature_engine.get_known_parameters()
    params = sorted(known_params) if known_params else _ALL_PARAMETERS
    matrix = _feature_engine.get_window_matrix(params, length=200)
    if matrix is not None:
        _iso_detector.fit(matrix, params)


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
            case "variance":
                ratio = r.details.get("ratio", 0)
                threshold = r.details.get("threshold", 2.5)
                rolling_std = r.details.get("rolling_std", 0)
                ref_std = r.details.get("ref_std", 0)
                parts.append(
                    f"Variance spike: rolling_std/ref_std = {ratio:.2f}× "
                    f"(threshold: {threshold}, rolling={rolling_std:.4f}, ref={ref_std:.4f})"
                )
            case "lstm":
                mse = r.details.get("mse", 0)
                thr = r.details.get("threshold", 0)
                z   = r.details.get("z_score", 0)
                parts.append(
                    f"Autoencoder: reconstruction MSE={mse:.4f} "
                    f"(threshold={thr:.4f}, z={z:.2f})"
                )
            case "tcn":
                mse = r.details.get("mse", 0)
                thr = r.details.get("threshold", 0)
                z   = r.details.get("z_score", 0)
                parts.append(
                    f"TCN: reconstruction MSE={mse:.4f} "
                    f"(threshold={thr:.4f}, z={z:.2f})"
                )
            case "trend_velocity":
                vel = r.details.get("max_velocity", 0)
                thr = r.details.get("threshold", 0)
                rat = r.details.get("ratio", 0)
                parts.append(
                    f"Trend acceleration: velocity={vel:.4f} "
                    f"(threshold={thr:.4f}, ratio={rat:.2f}×)"
                )
            case "matrix_profile":
                ds  = r.details.get("discord_score", 0)
                thr = r.details.get("threshold", 0)
                z   = r.details.get("z_score", 0)
                parts.append(
                    f"Unusual shape: discord score={ds:.4f} "
                    f"(threshold={thr:.4f}, z={z:.2f})"
                )
            case "correlation_graph":
                cz   = r.details.get("correlation_z", 0)
                peer = r.details.get("peer", "unknown")
                parts.append(
                    f"Correlation breakdown with {peer}: z={cz:.2f} "
                    "(expected peer relationship no longer holds)"
                )

    parts.append(f"{len(triggered)}/{len(results)} detectors triggered")
    return " | ".join(parts)


def _extract_contributions(results: list[DetectorResult]) -> dict[str, float]:
    for r in results:
        if r.detector_name == "isolation_forest" and r.details.get("feature_contributions"):
            return r.details["feature_contributions"]
    return {}
