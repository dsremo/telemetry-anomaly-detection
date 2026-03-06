"""AutoencoderDetector — GRU sequence autoencoder for temporal anomaly detection.

Complements the statistical ensemble (CUSUM, EWMA, Z-score, PELT, IF, Variance)
by learning the *temporal structure* of STL residuals for each channel.

Principle
---------
A GRU autoencoder is trained to reconstruct normal sequences of residuals.
At inference, reconstruction MSE for the current window is compared to the
learned training MSE distribution:

  anomaly_score = (mse - mean_train_mse) / (threshold_sigma × std_train_mse)
  is_anomaly    = mse > (mean_train_mse + threshold_sigma × std_train_mse)

This catches nonlinear temporal patterns — slowly building oscillation bursts,
correlated multi-step drifts, and regime changes — that single-point
statistical detectors miss even after STL decomposition.

Architecture (tiny, CPU-optimised)
-----------------------------------
  Encoder:   GRU(input=1, hidden=32) → last hidden state h
             Linear(32 → 8)          → latent z

  Decoder:   Linear(8 → 32)          → expand z
             Linear(32 → seq_len)    → reconstruct full sequence at once

Feed-forward decoder is deliberate:
  - No second GRU → full parallelism on CPU, inference < 1 ms
  - GRU encoder handles all temporal modelling; decoder just projects back
  - Trains stably on 60–200 calibration samples without gradient vanishing

Parameter count: ~5 K — well under the "no GPU needed" regime.

Lazy import
-----------
`import torch` is deferred to fit() and detect() so this module can be
imported in environments where PyTorch is not installed.  The detector
returns NOMINAL with reason="torch_not_available" instead of raising.

Integration
-----------
AutoencoderDetector mirrors the IsolationForestDetector API:
  det = AutoencoderDetector(...)
  det.add_sample(residual)   # called per new point in the CUSUM/EWMA loop
  if not det.is_fitted and det.sample_count >= det.min_train_samples:
      det.fit()
  result = det.detect(residuals_list)
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()


# ── Internal PyTorch model (defined lazily inside a function to avoid
#   import-time torch dependency) ────────────────────────────────────────────

def _build_model(seq_len: int, hidden: int, bottleneck: int):  # type: ignore[no-untyped-def]
    """Construct and return a _GRUAutoencoder nn.Module.

    Called lazily inside fit() so torch is never imported at module load time.
    """
    import torch.nn as nn  # noqa: PLC0415

    class _GRUAutoencoder(nn.Module):
        """GRU encoder + feed-forward decoder autoencoder for 1-D sequences."""

        def __init__(self) -> None:
            super().__init__()
            # Encoder: GRU collapses sequence → single hidden vector
            self.encoder_gru = nn.GRU(1, hidden, batch_first=True)
            self.enc_proj    = nn.Linear(hidden, bottleneck)
            # Decoder: expand latent → full sequence (feed-forward, no RNN)
            self.dec_proj    = nn.Linear(bottleneck, hidden)
            self.dec_out     = nn.Linear(hidden, seq_len)

        def forward(self, x):  # type: ignore[override]
            # x: (batch, seq_len, 1)
            _, h = self.encoder_gru(x)              # h: (1, batch, hidden)
            z    = self.enc_proj(h[0]).relu()        # (batch, bottleneck)
            h2   = self.dec_proj(z).relu()           # (batch, hidden)
            out  = self.dec_out(h2)                  # (batch, seq_len)
            return out.unsqueeze(-1)                 # (batch, seq_len, 1)

    return _GRUAutoencoder()


# ── Public detector class ────────────────────────────────────────────────────

class AutoencoderDetector:
    """GRU-based sequence autoencoder for unsupervised temporal anomaly detection.

    One instance per (satellite_id, parameter) pair.  The instance accumulates
    residuals via add_sample(), self-trains when enough data is available, and
    returns a DetectorResult from detect().

    Thread safety: single-threaded asyncio — no locking needed.
    """

    def __init__(
        self,
        seq_length:       int   = 30,
        hidden_size:      int   = 32,
        bottleneck_size:  int   = 8,
        epochs:           int   = 30,
        lr:               float = 0.01,
        min_train_samples: int  = 60,
        retrain_interval: int   = 500,
        threshold_sigma:  float = 3.0,
    ) -> None:
        self.seq_length       = seq_length
        self.hidden_size      = hidden_size
        self.bottleneck_size  = bottleneck_size
        self.epochs           = epochs
        self.lr               = lr
        self.min_train_samples = min_train_samples
        self.retrain_interval = retrain_interval
        self.threshold_sigma  = threshold_sigma

        # Internal state
        self._buffer: list[float]   = []   # residual accumulator
        self._samples_since_fit: int = 0   # ticks up after each fit
        self._is_fitted: bool        = False
        self._model                  = None  # _GRUAutoencoder | None

        # Learned from training data
        self._train_mean: float     = 0.0
        self._train_std:  float     = 1.0
        self._train_mse_mean: float = 0.0
        self._train_mse_std:  float = 1.0
        self._threshold: float      = float("inf")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def needs_refit(self) -> bool:
        """True when enough new residuals have arrived since last training."""
        return self._is_fitted and self._samples_since_fit >= self.retrain_interval

    # ── Data accumulation ────────────────────────────────────────────────────

    def add_sample(self, residual: float) -> None:
        """Append one STL residual to the training buffer.

        Called in the CUSUM/EWMA per-point loop so the buffer grows naturally
        alongside calibration warm-up.
        """
        self._buffer.append(float(residual))
        if self._is_fitted:
            self._samples_since_fit += 1

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, residuals: list[float] | None = None) -> None:
        """Train the GRU autoencoder on the provided or buffered residuals.

        No-op if data is insufficient or PyTorch is not installed.

        Parameters
        ----------
        residuals:
            Optional external list (e.g. full context window) to train on.
            If None, trains on the internal buffer accumulated via add_sample().
        """
        try:
            import torch                  # noqa: PLC0415
            import torch.nn as nn         # noqa: PLC0415
        except ImportError:
            logger.warning("autoencoder_torch_missing", reason="torch not installed")
            return

        data = list(residuals) if residuals is not None else list(self._buffer)
        if len(data) < self.min_train_samples:
            return

        # Build sliding-window sequences: (N, seq_length)
        seqs = [
            data[i: i + self.seq_length]
            for i in range(len(data) - self.seq_length + 1)
        ]
        if not seqs:
            return

        X = torch.tensor(seqs, dtype=torch.float32).unsqueeze(-1)  # (N, seq, 1)

        # Normalise to zero-mean / unit-std (improves training stability)
        self._train_mean = float(X.mean())
        self._train_std  = max(float(X.std()), 1e-6)
        X = (X - self._train_mean) / self._train_std

        model   = _build_model(self.seq_length, self.hidden_size, self.bottleneck_size)
        opt     = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out  = model(X)
            loss = loss_fn(out, X)
            loss.backward()
            opt.step()

        # Record per-sequence reconstruction errors to set anomaly threshold
        model.eval()
        with torch.no_grad():
            out    = model(X)
            errors = ((out - X) ** 2).mean(dim=(1, 2)).numpy()

        self._model          = model
        self._train_mse_mean = float(errors.mean())
        self._train_mse_std  = max(float(errors.std()), 1e-6)
        self._threshold      = (
            self._train_mse_mean + self.threshold_sigma * self._train_mse_std
        )
        self._is_fitted       = True
        self._samples_since_fit = 0

        logger.debug(
            "autoencoder_trained",
            seq_length=self.seq_length,
            n_sequences=len(seqs),
            threshold=round(self._threshold, 6),
        )

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, residuals: list[float]) -> "DetectorResult":  # type: ignore[name-defined]
        """Score the most recent `seq_length` residuals.

        Returns a DetectorResult with detector_name="lstm".  Score is in [0, 1]
        where 1 means the reconstruction error is >> threshold.

        Falls back to NOMINAL (is_anomaly=False) when:
          - Model not yet fitted
          - Residual window is shorter than seq_length
          - PyTorch not installed in the environment
        """
        from sentinel.core.models import DetectorResult, Severity  # noqa: PLC0415

        if not self._is_fitted:
            return DetectorResult(
                detector_name="lstm",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={"reason": "model_not_fitted"},
            )

        if len(residuals) < self.seq_length:
            return DetectorResult(
                detector_name="lstm",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={"reason": "insufficient_data"},
            )

        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return DetectorResult(
                detector_name="lstm",
                is_anomaly=False,
                score=0.0,
                severity=Severity.NOMINAL,
                details={"reason": "torch_not_available"},
            )

        window = residuals[-self.seq_length:]
        # Shape: (1, seq_length, 1)
        X = torch.tensor(
            [[v] for v in window], dtype=torch.float32
        ).unsqueeze(0)
        X = (X - self._train_mean) / self._train_std

        self._model.eval()
        with torch.no_grad():
            out = self._model(X)
        mse = float(((out - X) ** 2).mean())

        # Normalise z-score clamped to [0, 1]
        z     = (mse - self._train_mse_mean) / (
            self.threshold_sigma * self._train_mse_std
        )
        score = float(min(max(z, 0.0), 1.0))
        is_anomaly = mse > self._threshold

        severity = Severity.NOMINAL
        if is_anomaly:
            severity = (
                Severity.CRITICAL if z >= 3.0 else
                Severity.WARNING  if z >= 2.0 else
                Severity.WATCH
            )

        return DetectorResult(
            detector_name="lstm",
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

    # ── Persistence (warm-start across runs) ──────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist model weights and thresholds to disk.

        Saves a small checkpoint (~20 KB) with the PyTorch state_dict plus
        the learned MSE statistics.  Called after each successful retrain so
        the next run can warm-start instead of training from scratch.

        No-op if the model has not been fitted yet or if torch is unavailable.
        """
        if not self._is_fitted or self._model is None:
            return
        try:
            import torch  # noqa: PLC0415
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict":      self._model.state_dict(),
                    "train_mean":      self._train_mean,
                    "train_std":       self._train_std,
                    "train_mse_mean":  self._train_mse_mean,
                    "train_mse_std":   self._train_mse_std,
                    "threshold":       self._threshold,
                    "sample_count":    len(self._buffer),
                    "config": {
                        "seq_length":      self.seq_length,
                        "hidden_size":     self.hidden_size,
                        "bottleneck_size": self.bottleneck_size,
                    },
                },
                path,
            )
            logger.debug("lstm_model_saved", path=str(path))
        except Exception as exc:
            logger.warning("lstm_model_save_failed", path=str(path), error=str(exc))

    def load(self, path: Path) -> bool:
        """Load model weights from a previous run (warm-start).

        Rebuilds the architecture from saved config, loads state_dict, and
        restores the MSE statistics so detection can begin immediately without
        retraining.

        Returns True on success, False on any error (caller continues cold-start).
        """
        try:
            import torch  # noqa: PLC0415
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            cfg = checkpoint.get("config", {})
            model = _build_model(
                cfg.get("seq_length",      self.seq_length),
                cfg.get("hidden_size",     self.hidden_size),
                cfg.get("bottleneck_size", self.bottleneck_size),
            )
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
            logger.debug("lstm_model_loaded", path=str(path))
            return True
        except Exception as exc:
            logger.warning("lstm_model_load_failed", path=str(path), error=str(exc))
            return False
