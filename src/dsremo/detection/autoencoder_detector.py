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

from dsremo.detection.base_ml_detector import AbstractMLDetector


# ── Internal PyTorch model (defined lazily inside a function to avoid
#   import-time torch dependency) ────────────────────────────────────────────

def _build_gru_model(seq_len: int, hidden: int, bottleneck: int):  # type: ignore[no-untyped-def]
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

class AutoencoderDetector(AbstractMLDetector):
    """GRU-based sequence autoencoder for unsupervised temporal anomaly detection.

    One instance per (satellite_id, parameter) pair.  The instance accumulates
    residuals via add_sample(), self-trains when enough data is available, and
    returns a DetectorResult from detect().

    detector_name = "lstm"

    Thread safety: single-threaded asyncio — no locking needed.
    """

    _detector_name = "lstm"
    _log_prefix    = "autoencoder"

    def __init__(
        self,
        seq_length:        int   = 30,
        hidden_size:       int   = 32,
        bottleneck_size:   int   = 8,
        epochs:            int   = 30,
        lr:                float = 0.01,
        min_train_samples: int   = 60,
        retrain_interval:  int   = 500,
        threshold_sigma:   float = 3.0,
    ) -> None:
        super().__init__(
            seq_length=seq_length,
            epochs=epochs,
            lr=lr,
            min_train_samples=min_train_samples,
            retrain_interval=retrain_interval,
            threshold_sigma=threshold_sigma,
        )
        self.hidden_size     = hidden_size
        self.bottleneck_size = bottleneck_size

    def _build_model(self):  # type: ignore[no-untyped-def]
        return _build_gru_model(self.seq_length, self.hidden_size, self.bottleneck_size)

    def _model_config(self) -> dict:
        return {
            "seq_length":      self.seq_length,
            "hidden_size":     self.hidden_size,
            "bottleneck_size": self.bottleneck_size,
        }

    def _load_model_from_config(self, cfg: dict):  # type: ignore[no-untyped-def]
        return _build_gru_model(
            cfg.get("seq_length",      self.seq_length),
            cfg.get("hidden_size",     self.hidden_size),
            cfg.get("bottleneck_size", self.bottleneck_size),
        )
