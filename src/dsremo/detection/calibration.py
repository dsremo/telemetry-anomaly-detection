"""Per-channel calibration — reference distribution for CUSUM and EWMA.

Every channel learns its own reference mean (μ_ref) and standard deviation
(σ_ref) from its first CALIBRATION_WINDOW residual values.  These two numbers
drive ALL threshold calculations for CUSUM and EWMA; there are no global
thresholds anywhere in the detection pipeline.

State machine per channel:
    warming_up    — collecting the first N residuals; CUSUM/EWMA are disabled
    calibrated    — μ_ref and σ_ref are set; detectors are fully active
    recalibrating — μ_ref has drifted by > RECAL_FACTOR × σ_ref, indicating a
                    genuine operational regime change; restarting the reference
                    window (rare: only fires on large permanent step changes)

Derived parameters (recomputed whenever σ_ref changes):
    CUSUM:  k = K_FACTOR × σ_ref   (allowance — half the "shift to detect")
            H = H_FACTOR × σ_ref   (alarm threshold)
    EWMA:   UCL = +SIGMA_FACTOR × σ_ref × sqrt(λ/(2-λ))
            LCL = -SIGMA_FACTOR × σ_ref × sqrt(λ/(2-λ))

All four values are stored on the CalibrationState object so detectors never
need to recompute them — they just read the fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger()

# ── tuneable constants (also overridable via dsremo.yaml / init) ──────────
CALIBRATION_WINDOW:  int   = 100    # residuals before detectors activate
CUSUM_K_FACTOR:      float = 0.5    # k = K × σ_ref
CUSUM_H_FACTOR:      float = 5.0    # H = H × σ_ref
EWMA_LAMBDA:         float = 0.2    # smoothing weight per new sample
EWMA_SIGMA_FACTOR:   float = 3.0    # UCL/LCL = ±SIGMA × σ_ref × spread
RECAL_FACTOR:        float = 10.0   # recalibrate if |drift| > RECAL × σ_ref
SIGMA_UPDATE_INTERVAL: int = 720    # post-calibration samples between σ_ref refreshes
                                    # 720 = 30 days at 1-h resampling; adapts to
                                    # aging / long-term seasonal variance shifts
SIGMA_UPDATE_THRESHOLD: float = 0.10  # minimum σ_ref change ratio (|new/old - 1|)
                                      # to trigger a σ_ref update; avoids noisy
                                      # micro-updates on stable channels

# Pre-computed EWMA spread factor sqrt(λ/(2-λ)) — constant for a given λ
_ewma_spread = math.sqrt(EWMA_LAMBDA / (2.0 - EWMA_LAMBDA))

# ── Adaptive feature flags (overridable via dsremo.yaml / init_detectors) ─
GMM_ENABLED: bool = True            # fit GMM-2 on calibration window to handle
                                    # bimodal channels (two operating modes)
AUTO_Z_THRESHOLD_ENABLED: bool = True  # compute per-channel 99th-pct z from
                                       # calibration window; raises threshold
                                       # automatically for noisy channels


@dataclass
class CalibrationState:
    """All calibration data for one (satellite_id, parameter) pair."""

    state: str = "warming_up"           # warming_up | calibrated | recalibrating

    # Reference distribution (set once calibration window is full)
    ref_mean: float = 0.0
    ref_std:  float = 0.0
    sample_count: int = 0

    # CUSUM thresholds derived from σ_ref
    cusum_k: float = 0.0                # allowance
    cusum_h: float = 0.0                # alarm threshold

    # EWMA control limits derived from σ_ref
    ewma_ucl: float = 0.0
    ewma_lcl: float = 0.0

    # Internal: raw residuals collected during warming_up
    _buffer: list[float] = field(default_factory=list, repr=False)

    # Internal: recent post-calibration residuals for periodic σ_ref refresh.
    # Kept at ≤ CALIBRATION_WINDOW entries — a sliding window of recent noise.
    _recent_buffer: list[float] = field(default_factory=list, repr=False)
    _sigma_update_counter: int = 0

    # ── GMM-2 bimodal reference (set only when BIC(GMM-2) < BIC(Gaussian) - 10) ─
    # When not None, z-scores are computed as min distance to either component,
    # preventing false alarms on channels that oscillate between two normal states.
    gmm_means: list[float] | None = field(default=None, repr=False)
    gmm_stds:  list[float] | None = field(default=None, repr=False)

    # ── Per-channel empirical z-threshold ────────────────────────────────────
    # 99th percentile of |residual|/σ_ref from the calibration window.
    # Stored as the auto-calibrated lower bound for z-threshold overrides.
    # None until calibration completes.  Clamped to [3.0, 10.0].
    auto_z_threshold: float | None = None

    # ── derived-parameter helpers ───────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self.state == "calibrated"

    @property
    def is_warming(self) -> bool:
        return self.state in ("warming_up", "recalibrating")

    def _update_derived(self, sigma: float) -> None:
        """Recompute CUSUM and EWMA thresholds from a new σ_ref."""
        sigma = max(sigma, 1e-6)        # guard against near-zero variance
        self.ref_std  = sigma
        self.cusum_k  = CUSUM_K_FACTOR * sigma
        self.cusum_h  = CUSUM_H_FACTOR * sigma
        spread        = _ewma_spread
        self.ewma_ucl = +EWMA_SIGMA_FACTOR * sigma * spread
        self.ewma_lcl = -EWMA_SIGMA_FACTOR * sigma * spread

    def to_dict(self) -> dict:
        return {
            "state":        self.state,
            "ref_mean":     self.ref_mean,
            "ref_std":      self.ref_std,
            "sample_count": self.sample_count,
        }


def _try_fit_gmm(state: "CalibrationState", arr: "np.ndarray") -> None:
    """Attempt a GMM-2 fit on the calibration window.

    Sets state.gmm_means and state.gmm_stds when a 2-component Gaussian
    Mixture Model fits the data significantly better than a single Gaussian
    (BIC difference > 10).  Handles bimodal channels — e.g. a telemetry
    channel that oscillates between two operating modes — whose residuals
    look like infinite-σ spikes relative to a single-Gaussian baseline.

    Silently does nothing if sklearn is unavailable or the fit fails.
    BIC lower = better; bic1 - bic2 > 10 means GMM-2 is meaningfully better.
    """
    try:
        from sklearn.mixture import GaussianMixture  # noqa: PLC0415
    except ImportError:
        return

    try:
        data = arr.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=0)
        gmm1.fit(data)
        bic1 = gmm1.bic(data)

        gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gmm2.fit(data)
        bic2 = gmm2.bic(data)

        if (bic1 - bic2) > 10.0:
            means = gmm2.means_.flatten().tolist()
            stds  = [float(np.sqrt(c[0, 0])) for c in gmm2.covariances_]
            if all(s > 1e-9 for s in stds):
                state.gmm_means = means
                state.gmm_stds  = stds
    except Exception:  # noqa: BLE001
        pass  # any sklearn failure → fall back to single-Gaussian silently


class CalibrationManager:
    """Registry of per-channel calibration states.  Singleton.

    Call update() on every new residual value.  The state machine advances
    automatically:
        - First CALIBRATION_WINDOW calls → warming_up; returns state with
          is_calibrated=False so callers can skip CUSUM/EWMA.
        - At sample CALIBRATION_WINDOW → transitions to calibrated; μ_ref
          and σ_ref are frozen from the reference buffer.
        - Later: if |current_residual - μ_ref| > RECAL_FACTOR × σ_ref for
          RECAL_STREAK consecutive samples → recalibrating (rare).
    """

    # How many consecutive extreme-deviation samples trigger recalibration.
    _RECAL_STREAK: int = 20

    def __init__(self) -> None:
        self._states: dict[str, CalibrationState] = {}
        # Track consecutive high-deviation counts per channel for recal.
        self._recal_streak: dict[str, int] = {}

    # ── public API ──────────────────────────────────────────────────────

    def update(self, key: str, residual: float) -> CalibrationState:
        """Feed one residual; advance state machine; return current state.

        Args:
            key:     Channel key, e.g. "ESA-MISSION1:channel_047".
            residual: Latest STL residual value for the channel.

        Returns:
            CalibrationState — callers check .is_calibrated before running
            CUSUM/EWMA.  All thresholds (cusum_k, cusum_h, ewma_ucl, ewma_lcl)
            are ready to use when is_calibrated is True.
        """
        state = self._states.setdefault(key, CalibrationState())

        if state.state in ("warming_up", "recalibrating"):
            self._collect(key, state, residual)
        else:
            # Calibrated — check for regime shift
            self._check_regime(key, state, residual)

        return state

    def get(self, key: str) -> CalibrationState | None:
        """Return current calibration state without updating it."""
        return self._states.get(key)

    def load_from_db(self, key: str, data: dict) -> None:
        """Warm the in-memory state from a DB row loaded at startup.

        data keys: state, ref_mean, ref_std, ref_sample_count
        """
        state = self._states.setdefault(key, CalibrationState())
        state.state        = data.get("state", "warming_up")
        state.ref_mean     = float(data.get("ref_mean") or 0.0)
        state.sample_count = int(data.get("ref_sample_count") or 0)
        sigma              = float(data.get("ref_std") or 0.0)
        if state.state == "calibrated" and sigma > 0:
            state._update_derived(sigma)

    def all_db_records(self) -> list[tuple[str, dict]]:
        """Serialize all calibration states for DB persistence.

        Returns list of (key, record_dict) tuples.
        """
        return [
            (
                k,
                {
                    "state":            s.state,
                    "ref_mean":         s.ref_mean,
                    "ref_std":          s.ref_std,
                    "ref_sample_count": s.sample_count,
                },
            )
            for k, s in self._states.items()
        ]

    # ── state machine internals ─────────────────────────────────────────

    def _collect(self, key: str, state: CalibrationState, residual: float) -> None:
        """Accumulate residuals in the warmup buffer; transition when full."""
        state._buffer.append(residual)
        state.sample_count += 1

        if len(state._buffer) >= CALIBRATION_WINDOW:
            arr = np.asarray(state._buffer, dtype=np.float64)
            state.ref_mean = float(np.mean(arr))
            state._update_derived(float(np.std(arr, ddof=1)))

            # ── GMM-2 bimodal fit (resets on recalibration) ───────────────
            state.gmm_means = None
            state.gmm_stds  = None
            if GMM_ENABLED:
                _try_fit_gmm(state, arr)

            # ── Per-channel empirical z-threshold ─────────────────────────
            state.auto_z_threshold = None
            if AUTO_Z_THRESHOLD_ENABLED and state.ref_std > 1e-9:
                z_scores = np.abs(arr - state.ref_mean) / state.ref_std
                raw_pct = float(np.percentile(z_scores, 99))
                state.auto_z_threshold = float(np.clip(raw_pct, 3.0, 10.0))

            state._buffer.clear()
            state.state = "calibrated"
            self._recal_streak[key] = 0
            logger.info(
                "channel_calibrated",
                key=key,
                ref_mean=round(state.ref_mean, 4),
                ref_std=round(state.ref_std, 6),
                samples=state.sample_count,
                gmm_active=state.gmm_means is not None,
                auto_z=round(state.auto_z_threshold, 3) if state.auto_z_threshold else None,
            )

    def _check_regime(self, key: str, state: CalibrationState, residual: float) -> None:
        """Detect if the process has shifted to a new regime.

        Also performs a periodic σ_ref refresh every SIGMA_UPDATE_INTERVAL
        samples.  This prevents the frozen σ_ref (computed from the first
        CALIBRATION_WINDOW residuals) from becoming stale on long time-series
        where signal variance evolves due to component aging or seasonal effects.

        When σ_ref is too small relative to the current noise floor, CUSUM
        accumulates small-but-consistent deviations and fires repeatedly as a
        false positive — this adaptive update is the root-cause fix for that.
        """
        if state.ref_std < 1e-9:
            return
        deviation = abs(residual - state.ref_mean) / state.ref_std
        streak = self._recal_streak.get(key, 0)

        if deviation > RECAL_FACTOR:
            streak += 1
            self._recal_streak[key] = streak
            if streak >= self._RECAL_STREAK:
                # Genuine regime change — restart calibration.
                logger.info(
                    "channel_recalibrating",
                    key=key,
                    deviation_sigma=round(deviation, 2),
                    streak=streak,
                )
                state.state = "recalibrating"
                state._buffer = [residual]
                state._recent_buffer.clear()
                state._sigma_update_counter = 0
                state.sample_count += 1
                self._recal_streak[key] = 0
        else:
            # Normal — reset streak counter.
            self._recal_streak[key] = 0

        # ── Periodic σ_ref refresh ─────────────────────────────────────────
        # Accumulate recent residuals in a sliding window and refresh σ_ref
        # every SIGMA_UPDATE_INTERVAL samples.  Only acts when the new σ
        # differs meaningfully from the current one (>10% change) to avoid
        # noisy micro-updates.
        state._recent_buffer.append(residual)
        if len(state._recent_buffer) > CALIBRATION_WINDOW:
            # Keep only the most recent CALIBRATION_WINDOW residuals.
            state._recent_buffer = state._recent_buffer[-CALIBRATION_WINDOW:]

        state._sigma_update_counter += 1
        if state._sigma_update_counter >= SIGMA_UPDATE_INTERVAL:
            state._sigma_update_counter = 0
            min_samples = CALIBRATION_WINDOW // 2
            if len(state._recent_buffer) >= min_samples:
                arr = np.asarray(state._recent_buffer, dtype=np.float64)
                new_sigma = float(np.std(arr, ddof=1))
                if new_sigma > 1e-9:
                    old_sigma = state.ref_std
                    ratio = new_sigma / max(old_sigma, 1e-9)
                    if abs(ratio - 1.0) > SIGMA_UPDATE_THRESHOLD:
                        state._update_derived(new_sigma)
                        logger.debug(
                            "sigma_refreshed",
                            key=key,
                            old=round(old_sigma, 6),
                            new=round(new_sigma, 6),
                            ratio=round(ratio, 3),
                        )
