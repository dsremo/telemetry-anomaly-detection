"""AbstractMLDetector — shared base for GRU and TCN autoencoder detectors.

Both AutoencoderDetector (GRU, Sprint 11) and TCNDetector (Sprint 13) share
~90% identical logic: buffer management, fit/detect/save/load lifecycle, MSE
scoring, and persistence.  This base class extracts that shared logic once.

Subclasses override three small methods:
    _build_model()             → construct and return an nn.Module
    _model_config() -> dict    → architecture params to persist in checkpoint
    _load_model_from_config()  → rebuild model from saved config dict

Public API (identical for both subclasses):
    det.add_sample(residual)
    det.fit([residuals])
    det.detect(residuals) -> DetectorResult
    det.save(path)
    det.load(path) -> bool
    det.sample_count, det.is_fitted, det.needs_refit()

Design constraints:
    - Lazy torch import: module importable without PyTorch installed.
    - Single-threaded asyncio: no locking needed.
    - Buffer cap: training data capped at 2000 samples so per-channel
      retrain cost stays O(1) regardless of how much history accumulates.
      (Fixes ESA channel_15 bottleneck where unbounded history caused 30s retrains.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Maximum training samples used per fit().  Caps retrain cost for long-running
# channels.  2000 >> seq_length (30-32) so the sliding-window dataset still
# has ~1970 sequences — plenty for a tiny autoencoder.
_MAX_TRAIN_SAMPLES = 2000


class AbstractMLDetector(ABC):
    """Shared base for ML autoencoder anomaly detectors (GRU, TCN).

    Subclasses declare class variables:
        _detector_name: str   — used as DetectorResult.detector_name
        _log_prefix:    str   — prefix for structlog event names
    """

    _detector_name: str
    _log_prefix: str

    def __init__(
        self,
        seq_length:        int,
        epochs:            int,
        lr:                float,
        min_train_samples: int,
        retrain_interval:  int,
        threshold_sigma:   float,
    ) -> None:
        self.seq_length        = seq_length
        self.epochs            = epochs
        self.lr                = lr
        self.min_train_samples = min_train_samples
        self.retrain_interval  = retrain_interval
        self.threshold_sigma   = threshold_sigma

        # Runtime state
        self._buffer: list[float]    = []
        self._samples_since_fit: int = 0
        self._is_fitted: bool        = False
        self._model                  = None   # nn.Module | None

        # Learned from training data
        self._train_mean:     float = 0.0
        self._train_std:      float = 1.0
        self._train_mse_mean: float = 0.0
        self._train_mse_std:  float = 1.0
        self._threshold:      float = float("inf")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def needs_refit(self) -> bool:
        """True when enough new residuals have arrived since last training."""
        return self._is_fitted and self._samples_since_fit >= self.retrain_interval

    # ── Subclass contract ─────────────────────────────────────────────────────

    @abstractmethod
    def _build_model(self):  # type: ignore[no-untyped-def]
        """Construct and return an untrained nn.Module for this detector.

        Called lazily inside fit() — torch is never imported at module load.
        """

    @abstractmethod
    def _model_config(self) -> dict:
        """Return the architecture hyperparameters to persist in checkpoint."""

    @abstractmethod
    def _load_model_from_config(self, cfg: dict):  # type: ignore[no-untyped-def]
        """Rebuild model from a saved config dict (loaded from checkpoint)."""

    # ── Data accumulation ─────────────────────────────────────────────────────

    def add_sample(self, residual: float) -> None:
        """Append one STL residual to the training buffer."""
        self._buffer.append(float(residual))
        if self._is_fitted:
            self._samples_since_fit += 1

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, residuals: list[float] | None = None) -> None:
        """Train the model on the provided or buffered residuals.

        No-op if data is insufficient or PyTorch is not installed.

        Parameters
        ----------
        residuals:
            Optional external list to train on.  If None, trains on the
            internal buffer accumulated via add_sample().
            In both cases, data is capped at the last _MAX_TRAIN_SAMPLES
            points so that retrain cost stays O(1) for long-running channels.
        """
        try:
            import torch                  # noqa: PLC0415
            import torch.nn as nn         # noqa: PLC0415
        except ImportError:
            logger.warning(
                f"{self._log_prefix}_torch_missing",
                reason="torch not installed",
            )
            return

        raw = list(residuals) if residuals is not None else list(self._buffer)
        if len(raw) < self.min_train_samples:
            return

        # Cap at _MAX_TRAIN_SAMPLES to bound retrain cost.
        data = raw[-_MAX_TRAIN_SAMPLES:] if len(raw) > _MAX_TRAIN_SAMPLES else raw

        seqs = [
            data[i: i + self.seq_length]
            for i in range(len(data) - self.seq_length + 1)
        ]
        if not seqs:
            return

        X = torch.tensor(seqs, dtype=torch.float32).unsqueeze(-1)  # (N, seq, 1)

        # Normalise to zero-mean / unit-std for training stability.
        self._train_mean = float(X.mean())
        self._train_std  = max(float(X.std()), 1e-6)
        X = (X - self._train_mean) / self._train_std

        model   = self._build_model()
        opt     = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(model(X), X)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            errors = ((model(X) - X) ** 2).mean(dim=(1, 2)).numpy()

        self._model          = model
        self._train_mse_mean = float(errors.mean())
        self._train_mse_std  = max(float(errors.std()), 1e-6)

        # ── POT threshold (Peak Over Threshold / Extreme Value Theory) ────────
        # The fixed 3σ threshold assumes Gaussian reconstruction errors, but
        # GRU/TCN errors are right-skewed.  POT fits a Generalized Pareto
        # Distribution (GPD) to the tail of training errors and sets the
        # threshold at the 0.1% false-positive rate.
        # Falls back to 3σ if scipy is unavailable or too few tail samples.
        pot_threshold = self._pot_threshold(errors)
        self._threshold      = (
            pot_threshold if pot_threshold is not None
            else self._train_mse_mean + self.threshold_sigma * self._train_mse_std
        )
        self._is_fitted       = True
        self._samples_since_fit = 0

        logger.debug(
            f"{self._log_prefix}_trained",
            seq_length=self.seq_length,
            n_sequences=len(seqs),
            threshold=round(self._threshold, 6),
        )

    # ── POT threshold calibration ─────────────────────────────────────────────

    @staticmethod
    def _pot_threshold(
        errors: "np.ndarray",  # type: ignore[name-defined]
        q: float = 0.001,
        init_percentile: float = 0.85,
        min_tail_samples: int = 15,
    ) -> float | None:
        """Compute anomaly threshold via Peak Over Threshold (EVT).

        Sets the threshold at risk level q (default 0.1% FPR) using the
        Generalized Pareto Distribution fitted to training error tail exceedances.
        Returns None if scipy is unavailable or there are insufficient tail samples
        (falls back to caller's σ-based threshold in that case).

        Uses 85th percentile as the initial threshold (rather than 98th) to
        ensure enough tail samples even with 60-200 training sequences.

        Reference: Siffer et al., KDD 2017 — SPOT algorithm.
        """
        try:
            from scipy.stats import genpareto  # noqa: PLC0415
            import numpy as _np               # noqa: PLC0415
        except ImportError:
            return None

        init_t = float(_np.quantile(errors, init_percentile))
        exceedances = errors[errors > init_t] - init_t
        if len(exceedances) < min_tail_samples:
            return None  # not enough tail data — fall back to σ threshold

        try:
            shape, _, scale = genpareto.fit(exceedances, floc=0)
        except Exception:
            return None

        n_total  = len(errors)
        n_excess = len(exceedances)
        if shape == 0.0:
            # Exponential tail (shape → 0)
            return float(init_t - scale * _np.log(q * n_total / n_excess))
        return float(init_t + (scale / shape) * (
            (q * n_total / n_excess) ** (-shape) - 1.0
        ))

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, residuals: list[float]) -> "DetectorResult":  # type: ignore[name-defined]
        """Score the most recent seq_length residuals.

        Returns a DetectorResult with score in [0, 1].
        Falls back to NOMINAL when not fitted, insufficient data, or no torch.
        """
        from dsremo.core.models import DetectorResult, Severity  # noqa: PLC0415

        name = self._detector_name

        if not self._is_fitted:
            return DetectorResult(
                detector_name=name, is_anomaly=False, score=0.0,
                severity=Severity.NOMINAL, details={"reason": "model_not_fitted"},
            )

        if len(residuals) < self.seq_length:
            return DetectorResult(
                detector_name=name, is_anomaly=False, score=0.0,
                severity=Severity.NOMINAL, details={"reason": "insufficient_data"},
            )

        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return DetectorResult(
                detector_name=name, is_anomaly=False, score=0.0,
                severity=Severity.NOMINAL, details={"reason": "torch_not_available"},
            )

        window = residuals[-self.seq_length:]
        X = torch.tensor([[v] for v in window], dtype=torch.float32).unsqueeze(0)
        X = (X - self._train_mean) / self._train_std

        self._model.eval()
        with torch.no_grad():
            out = self._model(X)
        mse = float(((out - X) ** 2).mean())

        z     = (mse - self._train_mse_mean) / (
            self.threshold_sigma * self._train_mse_std
        )
        score      = float(min(max(z, 0.0), 1.0))
        is_anomaly = mse > self._threshold

        severity = Severity.NOMINAL
        if is_anomaly:
            severity = (
                Severity.CRITICAL if z >= 3.0 else
                Severity.WARNING  if z >= 2.0 else
                Severity.WATCH
            )

        return DetectorResult(
            detector_name=name,
            is_anomaly=is_anomaly,
            score=score,
            severity=severity,
            details={
                "mse":            mse,
                "threshold":      self._threshold,
                "train_mse_mean": self._train_mse_mean,
                "train_mse_std":  self._train_mse_std,
                "z_score":        z,
            },
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist model weights and MSE statistics to a checkpoint file.

        No-op if not fitted or torch is unavailable.
        """
        if not self._is_fitted or self._model is None:
            return
        try:
            import torch  # noqa: PLC0415
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict":     self._model.state_dict(),
                    "train_mean":     self._train_mean,
                    "train_std":      self._train_std,
                    "train_mse_mean": self._train_mse_mean,
                    "train_mse_std":  self._train_mse_std,
                    "threshold":      self._threshold,
                    "sample_count":   len(self._buffer),
                    "config":         self._model_config(),
                },
                path,
            )
            logger.debug(f"{self._log_prefix}_model_saved", path=str(path))
        except Exception as exc:
            logger.warning(
                f"{self._log_prefix}_model_save_failed",
                path=str(path),
                error=str(exc),
            )

    def load(self, path: Path) -> bool:
        """Warm-start from a persisted checkpoint.

        Returns True on success, False on any error (caller continues cold-start).
        """
        try:
            import torch  # noqa: PLC0415
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = self._load_model_from_config(checkpoint.get("config", {}))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._model           = model
            self._train_mean      = float(checkpoint["train_mean"])
            self._train_std       = float(checkpoint["train_std"])
            self._train_mse_mean  = float(checkpoint["train_mse_mean"])
            self._train_mse_std   = float(checkpoint["train_mse_std"])
            self._threshold       = float(checkpoint["threshold"])
            self._is_fitted       = True
            self._samples_since_fit = 0
            logger.debug(f"{self._log_prefix}_model_loaded", path=str(path))
            return True
        except Exception as exc:
            logger.warning(
                f"{self._log_prefix}_model_load_failed",
                path=str(path),
                error=str(exc),
            )
            return False
