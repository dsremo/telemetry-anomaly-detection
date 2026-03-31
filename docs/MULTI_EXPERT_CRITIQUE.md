# Dsremo Telemetry Anomaly Detection — Brutal Multi-Expert Critique

*Generated: 2026-03-31*
*"We're not here to be kind. We're here to make it right."*

---

## PANEL 1 — Google Senior SWE (SRE + ML Infra perspective)

### What they'd say after reading the code:

**Good instincts, amateur execution on reliability.**

1. **Single-threaded asyncio is a ticking bomb.** You have ML retraining (`fit()` on GRU/TCN) happening inside the same event loop as API request handling. PyTorch training is CPU-bound and will block your event loop for hundreds of milliseconds. You're calling `asyncio.run_in_executor(None, ...)` nowhere near the ML fit path. This will cause timeouts in production under load. **Fix: train in a ProcessPoolExecutor or a separate worker thread pool.**

2. **No backpressure on ingestion.** If a customer sends 10,000 telemetry points in a burst, your pipeline will process them synchronously, building unbounded in-memory queues. There's no queue depth limit, no rejection policy, no circuit breaker. **This is not production-grade.**

3. **The "FIFO eviction at 200 models" and "10,000 channel state cap" are mentions in comments, not enforced by tests.** I searched — there's no test that actually instantiates 201 models and verifies eviction happens. This is aspirational documentation.

4. **alert_cooldown_hours: 360** (15 days). This is a single global config. In a multi-tenant SaaS system, what happens when Tenant A's satellite has a genuine recurring fault every 14 days? You silently suppress it. This should be configurable per satellite, per parameter, per severity.

5. **The WebhookRouter retries once on 429.** Just once? What about transient 502/503? What about exponential backoff? What about dead-letter queues? At Google we'd use Pub/Sub with acknowledgment. This is fine for a prototype, but not for "production-hardened" as you've documented.

6. **No distributed tracing.** You use structlog (good), but there's no trace_id propagated from ingestion → detection → alert. When an operator files a bug "alert X fired at 14:32", you cannot reconstruct the execution path. OpenTelemetry takes 2 hours to add and would transform your debuggability.

---

## PANEL 2 — SpaceX Software Engineer (Falcon/Starlink Telemetry)

### What they'd say:

**You're detecting anomalies on already-processed telemetry. That's the wrong layer.**

1. **No CCSDS awareness at all.** Real spacecraft telemetry arrives as CCSDS frames. Between the raw frame and your `TelemetryPoint` JSON, there are: frame sync, VCID demuxing, decompression, engineering unit conversion (raw counts → voltage via calibration tables), range/limits checking in the ground system. You're assuming all of that already happened. When it fails, your "stale data" detector fires. That's a coincidence, not design.

2. **Your orbital period is hardcoded to 5400 seconds (90 min LEO).** Starlink shells orbit at different altitudes. ISS is 90 min. Molniya is 12 hours. GEO has no periodicity. Your STL period detection uses FFT — ok — but your fallback, **the entire calibration architecture, assumes LEO**. For MEO GPS satellites (2-hour period), your calibration window of 200 residuals is 200/120 = 1.6 orbital periods. That's dangerously thin for STL to identify the seasonal component properly.

3. **Eclipse transition is handled by removing seasonality via STL. This is necessary but insufficient.** During eclipse entry/exit, you get transient thermal shocks on battery temp, bus voltage sags, and panel current drops — all within 30-90 seconds. Your STL seasonal window is computed as `max(7, period_samples | 1)`. If the period is 5400 samples and you're sampling at 1 Hz, your seasonal window is 5399 samples. The STL smoother won't cleanly subtract a sharp thermal edge — you'll get Gibbs-like ringing in your residuals. At SpaceX we apply separate eclipse-flag-based windowing, not STL alone.

4. **The CUSUM H-factor was raised from 5.0 to 8.0 "for aging tolerance."** That's like widening a speed limit because people keep speeding. Aging IS an anomaly if it crosses safe operating margins. You're adjusting statistical sensitivity to suppress genuine degradation events. **This is exactly backwards for a safety-critical system.**

5. **Matrix Profile Discord detector has weight 0.06 — the lowest non-isolation-forest weight.** Matrix Profile is one of the most powerful exact methods for detecting novel time series patterns (UCR/UCF Keogh group). Giving it 0.06 is almost vestigial. Either invest in it properly or remove it.

---

## PANEL 3 — NASA JPL / GSFC Flight Software Engineer

### What they'd say:

**The architecture lacks the adversarial discipline we require for spacecraft.**

1. **No formal verification of threshold derivation.** Your CUSUM parameters are `k = 0.5σ, H = 8σ`. Where did these come from? The 1954 Page paper? Wald's sequential analysis? The CUSUM ARL (Average Run Length) under these parameters at your specific sampling rates — what is it? For a fault with shift magnitude = 1σ above nominal, with these parameters, how many samples on average before alarm? At H=8, k=0.5, the ARL₁ ≈ 10-11 samples. At 1 Hz, that's ~10 seconds. At 1/3600 Hz (hourly), that's 10 hours. The same threshold behaves completely differently at different sampling rates. **You have no ARL budget in your design.**

2. **The GMM-2 bimodal detection is fundamentally sound, but the BIC difference threshold of 10 is arbitrary.** BIC penalizes model complexity by `k × ln(n)`. For n=100 samples, `ln(100)≈4.6`. Adding 3 parameters (second Gaussian) adds `3×4.6=13.8` to BIC. So your threshold of 10 will frequently fire false bimodality at small sample sizes. At n=200, the threshold should scale as ~14. **This should be `n_samples_in_window × ln(n_samples_in_window) × 0.5` or similar, not a hardcoded 10.**

3. **You don't model measurement noise separately from process noise.** All your detectors operate on the signal. But temperature sensors have ±0.5°C resolution. Voltage sensors have ADC quantization noise. Battery current sensors have ±50 mA shunt noise. None of this is modeled. Your `ref_std` conflates:
   - True physical process variance
   - Sensor noise floor
   - Quantization artifacts
   This means for precise sensors (small noise floor), your effective SNR is high and detection is good. For noisy sensors (large noise floor), `ref_std` is dominated by noise, and genuine drifts get hidden. **A proper system would maintain a noise model separate from process variance, ideally using a Kalman filter.**

4. **The incident grouper uses a 300s window.** In a multi-subsystem cascade failure (e.g., power fault cascading to ADCS desat), the thermal anomaly may appear 600s after the root cause. Your grouper would create two separate incidents. NASA ASIST uses causal graph propagation times, not fixed windows. You have a `CAUSAL_GRAPH` hardcoded in `explainer.py` — use propagation delays from it in the grouper.

5. **No formal FMEA (Failure Mode and Effects Analysis) traceability.** Real flight software anomaly detectors are traceable to specific failure modes in the FMEA. Each detector should have a documented failure mode it covers. "CUSUM detects drift" is not an FMEA entry. "CUSUM detects LiPo cell degradation manifesting as >2%/week capacity fade on channel BAT_CAP_Ah" is.

---

## PANEL 4 — ISRO Scientist (PSLV / Chandrayaan operations)

### What they'd say:

**Operationally, this will create alert fatigue. Operationally, this will miss slow missions.**

1. **The 360-hour (15-day) cooldown is too long for ISRO missions.** Chandrayaan-3's Pragyan rover had a 14-day nominal mission. A 15-day cooldown means you could suppress the ONLY anomaly alert for the entire mission lifetime. For short missions, cooldown should be proportional to mission phase duration.

2. **No mission phase awareness.** ISRO operations distinguish: launch, ascent, orbit insertion, commissioning, nominal ops, eclipse season, maneuver, safe mode, end-of-life. Your system has none of this. During a thruster firing maneuver, attitude rates spike — this is nominal, not anomalous. Your system would alarm. **Without phase-gating, your system cannot be used during dynamic mission phases.**

3. **GAGAN/NavIC signals have different noise characteristics from LEO EPS.** Navigation satellites have thermal cycles every 12 hours (GEO), not 90 minutes. Your STL seasonal detection will fail silently — FFT will find the right period, but your calibration window of 200 samples at 1-hour sampling is only 8 days of data: less than one full GEO thermal cycle (which takes ~10 days to stabilize after eclipse season starts). The `ref_std` will be computed on a partial thermal cycle and will be wrong.

4. **Indian ground stations have intermittent contact windows.** AOS/LOS (Acquisition of Signal / Loss of Signal) creates systematic gaps every 90 minutes. Your stale data detector fires at 300s gaps. Every single orbit, your system generates a "stale data" event that is not anomalous. This will train your operators to ignore stale-data alerts. **Classic boy-who-cried-wolf.**

5. **No multilingual output for operations.** Minor point, but ISRO ground teams work in Hindi and English. Alert messages in English-only with technical jargon ("CUSUM H-factor exceeded") will be misunderstood in time-critical ops. This is a real operational risk.

---

## PANEL 5 — Pure Mathematician (Probability & Statistics, IIT-B / IISc level)

### What they'd say: *(Most brutal section)*

**The mathematics is a collection of heuristics dressed up as a system. Let me be specific.**

### 5.1 The Fundamental Flaw: You're Testing One Hypothesis, Twelve Times

Your ensemble combines 11 detectors, each generating a binary alarm. The ensemble confidence is `sum(triggered_weights) / total_weights`. **This is not a statistical test. It is a voting scheme with no theoretical basis.**

The core problem: Your 11 detectors are NOT independent. CUSUM, EWMA, Statistical, Variance, Trend Velocity — all operate on the same STL residuals. They are correlated. When a genuine anomaly occurs:
- CUSUM fires (accumulates the signal)
- EWMA fires (tracks the same shift)
- Statistical fires (z-score on same residual)
- Variance fires (if std changes too)

You've created a dependency structure but treat it as if each "vote" is independent. The effective number of independent detectors is far less than 11. Your ensemble confidence of 0.85 → CRITICAL is **not calibrated to any false positive rate**. What is P(ensemble > 0.85 | no anomaly)? You don't know. You have no way to know without formal derivation or empirical calibration. This is not engineering. This is numerology.

### 5.2 The STL Residuals Are Not i.i.d. — But You Treat Them As If They Are

Every statistical test in your system (z-score, CUSUM, EWMA) assumes the residuals are independently and identically distributed with some reference distribution. STL residuals are **not** i.i.d. They are:
- **Autocorrelated**: Telemetry has physical inertia. Temperature doesn't jump independently sample-to-sample. Residual at t is correlated with residual at t-1, t-2, ...
- **Heteroskedastic**: Variance changes with orbital phase (even after STL removes the mean seasonal component, the variance may be seasonal — variance seasonality, i.e., `GARCH` effects)

**Consequences:**
- Your z-score critical value of 3.0 corresponds to P=0.0027 under i.i.d. Normal assumption. Under autocorrelation with lag-1 AR coefficient rho=0.5, the effective variance of the mean is inflated by (1+rho)/(1-rho) = 3x. Your effective critical value for the same P-value should be `3.0 * sqrt(3) = 5.2`. You're alarming at 3.0. Your false positive rate is ~10x higher than you think.
- Your CUSUM ARL derivation (H=8, k=0.5) is derived from Wald's paper on **i.i.d. Normal observations**. Under autocorrelation, the ARL is completely different. The accumulated sum doesn't reset to zero — it inherits the autocorrelation. You have no ARL characterization.

### 5.3 MAD-Based Z-Score Is Robust, But Not For What You Think

You use Modified Z-score: `0.6745 * |x - median| / MAD`. The factor 0.6745 = inverse-Normal(0.75) makes MAD consistent for Normal distributions. But:
- Consistency factor only holds asymptotically for large N. For rolling windows of 30-50 samples, the finite-sample bias is significant.
- More critically: **you apply the same threshold (3.0) to the MAD-based z-score as the regular z-score**. This is wrong. The Modified z-score and the classical z-score have different distributions. Iglewicz & Hoaglin (1993) recommend threshold of **3.5**, not 3.0, for the Modified z-score.
- You fall back from MAD to std when MAD~0. But MAD~0 means 50%+ of your window is constant. In that regime, even the std-based z-score is unreliable (the distribution is not Normal — it's quasi-discrete).

### 5.4 BIC for GMM-2: The Threshold Is Dimensionally Inconsistent

BIC = -2*log-likelihood + k*ln(n) where k = parameters, n = samples.

For 1-component Gaussian: k=2 (mean, variance). For 2-component: k=5 (2 means, 2 variances, 1 mixing proportion). DeltaK = 3 parameters.

BIC difference threshold should be on the same scale as the log-likelihood improvement. Your hardcoded threshold of 10 is **not dimensionally consistent** across different n values:
- n=100: penalty for 3 extra params = 3*ln(100) = 13.8 -> threshold 10 < penalty -> bimodality detection is liberal
- n=1000: penalty = 3*ln(1000) = 20.7 -> threshold 10 << penalty -> even more liberal
- n=10000: penalty = 3*ln(10000) = 27.6 -> **at large n, you will massively over-detect bimodality**

The correct null test should be: if |ΔBIC| < penalty, do not reject unimodality. Your threshold of 10 does the opposite of BIC penalization intention at large n.

### 5.5 The Ensemble Weights Are Unjustified

CUSUM: 0.17, EWMA: 0.14, Statistical: 0.10, etc. Where do these come from? From the git history and documentation, they appear to have been set manually and tweaked. **This is not optimal weighting.**

Optimal ensemble weights under independence (Dempster-Shafer, Bayesian model averaging, or simple logistic regression meta-learning) would weight each detector by its log-likelihood ratio: `w_i ~ log(TPR_i / FPR_i)`. Without measured TPR and FPR per detector on your actual data, these weights are fiction.

Furthermore, the PELT changepoint detector has weight 0.08 but PELT is an O(n) exact algorithm that finds *all* changepoints in the window. On a 300-sample lookback at every incoming sample, you're running O(n) at every timestep — O(n^2) total. This is your computational bottleneck and it has one of the lowest weights. That's perverse.

### 5.6 STL Period Detection via FFT: Critical Mathematical Error

```python
peak_freq = np.argmax(np.abs(rfft_vals[1:])) + 1
period = round(n / peak_freq)
validates: period <= n//2
```

**This is wrong.** `rfft` of n samples has n//2+1 frequency bins. Bin k corresponds to frequency k/n. Period = n/k. Your validation `period <= n//2` means `n/k <= n/2`, i.e., `k >= 2`. But you should also validate that the peak has adequate signal-to-noise AND that the period is physically plausible given the mission. You validate against `4 * median_amplitude` threshold — but median amplitude of the rfft spectrum is dominated by the DC component and low-frequency content. For telemetry with strong long-term drift, the median is inflated and your "4x median" criterion may suppress detection of real seasonal components.

**The correct approach**: Use peak prominence above the background spectral estimate (Welch's method) with physical bounds on plausible orbital periods.

### 5.7 The Auto Z-Threshold (99th Percentile) Is Self-Defeating

```python
auto_z = np.percentile(np.abs(residuals / ref_std), 99)
clamped to [3.0, 10.0]
```

If your calibration window contains anomalies (which happens during recalibration after a regime shift), the 99th percentile of `|residual|/sigma_ref` is contaminated. Your threshold will be set HIGH because anomalous samples inflated the 99th percentile, making your system **less sensitive precisely when it was just in an anomalous state**. This is a form of **masking**: the system calibrates high thresholds from contaminated windows and then misses subsequent anomalies.

The correct approach: Use a robust estimator for the threshold (e.g., median + 3*IQR of `|residual|/sigma_ref`) or apply anomaly-trimmed estimation (trim top 5% before computing the 99th percentile). Ironic to use MAD for z-scores but then use non-robust percentile for threshold setting.

---

## PANEL 6 — IIT Professor (Computer Science + Signal Processing)

### What they'd say:

**Good pedagogy, weak engineering discipline in the mathematical scaffolding.**

1. **Savitzky-Golay as STL fallback is fine, but the window length is hardcoded.** SG filter's frequency response depends on polynomial order AND window length. Your window length needs to match the expected trend timescale. For a 90-minute orbit with hourly data, the trend changes over days — a fixed 51-point SG window is arbitrary. **The window should be derived from the lowest expected frequency in the trend (Nyquist for trend = 2x minimum trend period).**

2. **np.gradient for velocity computation uses central differences.** Central difference has O(h^2) error. But your `velocity` is computed on 20-sample windows, and you look at `recent_points = 5` for the maximum. The "trend velocity" is actually the gradient of the STL trend component, not the raw signal trend. This is good. But: STL trend is a smoothed estimate and its gradient inherits the smoothing. The `threshold_sigma * (ref_std / window)` normalization treats the trend gradient standard deviation as `ref_std / window`. This is dimensionally correct (units: signal_unit/sample) only if the trend is linear. For curved trends, this normalization is wrong.

3. **Your GRU architecture (hidden=32, bottleneck=8) has no justification.** Why 32 hidden units? Why 8-dimensional bottleneck? For a univariate time series of residuals, the intrinsic dimensionality is likely 2-3 (drift, variance, phase). An 8-dimensional bottleneck may be severely overparameterized, leading to memorization rather than compression. Classic autoencoder theory says bottleneck should be at or slightly above intrinsic dimensionality of normal behavior.

4. **TCN receptive field of 61 steps is presented as a design feature.** Receptive field = `(kernel_size - 1) * sum(dilations) + 1 = (2-1) * (1+2+4+8) + 1 = 16`. Wait — 4 blocks, dilation 1,2,4,8, but what's the kernel size? If kernel_size=3: RF = (3-1)*(1+2+4+8) + 1 = 2*15+1 = 31, not 61. If kernel_size=4: RF = 3*15+1 = 46. To get RF=61: kernel_size=4, dilations up to 16 (1,2,4,8,16 = 5 blocks), RF = 3*31+1=94. Or kernel_size=5, 4 blocks: 4*15+1=61. But this means your receptive field is defined by architecture parameters that aren't documented together. **A student reading this cannot verify the claim without deriving it.**

5. **The isolation forest contamination=0.01 (1%) is a prior.** It says "I expect 1% of my data to be anomalous." For a healthy satellite in nominal ops, the true anomaly rate might be 0.001% (one anomaly per 100,000 samples). You're tuning your prior 10x too high, which shifts the decision boundary toward more false positives. Contamination in Isolation Forest directly affects the offset in `decision_function`. For spacecraft monitoring, contamination should be much lower (0.001-0.01).

---

## PANEL 7 — Open Source Contributor (Python ecosystem, statsmodels, ruptures, stumpy)

### What they'd say:

**The dependency choices are sensible, the integration is shallow.**

1. **You use `ruptures` for PELT but ignore its `Binseg` and `Dynp` algorithms.** PELT is optimal for many changepoints. For single changepoint detection (single structural break), `Dynp` with n_bkps=1 is exact and faster. For online streaming detection, `ruptures` has no online API — you're re-running PELT on 300 samples every timestep. **This is O(n) per call, every call, which compounds over hours of operation.** Consider `river` library's online changepoint detection (BOCD: Bayesian Online Changepoint Detection) which is O(1) per sample.

2. **Matrix Profile via `stumpy` is correct, but the discord detection threshold is unclear.** Discord = most unusual subsequence (maximum nearest-neighbor distance in matrix profile). Your implementation weight (0.06) and threshold derivation aren't visible in the exploration summary. STUMPY's `stumpy.stump()` is O(n^2) for a single pass — on 300-sample windows at every timestep, this is expensive. Use `stumpy.stumpi()` (online matrix profile update, O(n) per update) instead.

3. **`scipy.signal.savgol_filter` + `statsmodels.tsa.seasonal.STL` + `ruptures.Pelt` + `sklearn.ensemble.IsolationForest` + `torch` (GRU, TCN) + `stumpy` (Matrix Profile).** That's 6 major scientific computing libraries. Your `pyproject.toml` should version-pin these precisely, and your CI should test the minimum supported versions. A single minor-version update to `statsmodels` has broken STL behavior before (the `robust` parameter semantics changed between 0.13 and 0.14).

4. **The lazy import pattern for torch** (`try: import torch except ImportError`) is correct for optional dependencies. But your `pyproject.toml` should use optional dependency groups:
   ```toml
   [project.optional-dependencies]
   ml = ["torch>=2.0", "torchvision"]
   full = ["dsremo[ml]", "stumpy>=1.4"]
   ```
   Currently the boundary between required and optional is only in comments.

5. **No `typing` completeness.** Your dataclasses use `frozen=True` (excellent). But function signatures throughout the detection pipeline are partially typed. `mypy --strict` would reveal significant gaps. For a library that others might build on, this is a barrier to contribution.

---

## PANEL 8 — Space Startup (Series A, building on top of Dsremo as API)

### What they'd say:

**We'd love to use this. Here's why we can't yet.**

1. **The API returns `Anomaly` objects with a fixed schema.** We're building on top of it for a GEO comms satellite constellation. Our satellites have `channel_*` patterns that skip Isolation Forest (you hardcode ESA pattern exclusion). There's no way for us to override this behavior per-satellite-fleet without modifying your source. **The detection pipeline needs a plugin/strategy interface**, not hardcoded domain rules.

2. **Multi-tenancy is implemented but docs say it's untested at scale.** We have 50 satellites per tenant, 10 tenants. That's 500 satellite states, each with 18+ channels = 9,000+ channel calibration states, potentially 9,000 x 11 = 99,000 detector states. Your 10,000-channel cap means we'd FIFO-evict half our fleet's calibration state every orbit. **The cap needs to be per-tenant, not global.**

3. **No SLA definitions in the API.** We're building a commercial product on top. What's the latency guarantee for `/api/v1/ingest`? What's the throughput? You have nginx rate limiting but no published SLA. We can't sign customer contracts without a documented SLA.

4. **The webhook retry on 429 with `Retry-After` handling is thoughtful.** But what happens if our receiving endpoint is down for 2 hours? Your single retry loses the alert forever. We need at least a configurable retry queue with dead-letter storage.

5. **The dashboard is "minimal vanilla JS."** For a B2B space tech product, the dashboard IS the product for most customers. Operators evaluate vendor tools by the dashboard. "Minimal vanilla JS" in a competitive market (Telematica, SpaceSense, Cognitive Space) means no deal. This needs to be a proper React/TypeScript application with:
   - Time-series visualization (Plotly/D3, not HTML tables)
   - Alert management workflow (acknowledge, escalate, close)
   - Per-channel calibration state visibility
   - Detector contribution breakdown per anomaly

---

## PANEL 9 — Space Researcher (Academic, anomaly detection literature)

### What they'd say:

**The literature review is shallow. Key innovations exist that you've reinvented or missed.**

1. **BOCPD (Bayesian Online Changepoint Detection, Adams & MacKay 2007)** is the Bayesian principled version of what you're trying to do with CUSUM + PELT combined. It maintains a posterior distribution over the run length (time since last changepoint) and is provably Bayesian-optimal for changepoint detection under the model. You have PELT (offline, exact) and CUSUM (online, heuristic). BOCPD would give you online, principled changepoint detection with calibrated probabilities — directly replacing both.

2. **MSET (Multivariate State Estimation Technique)** is used extensively by NASA for spacecraft anomaly detection since the 1990s (Gross, Singer, Herzog — NASA TM-2006). It's a kernel-based nonparametric method that learns the normal state manifold and alerts when current state deviates. Your Isolation Forest is the modern ML analog, but MSET has 30 years of flight heritage. The paper trail for certification would be immediate.

3. **OCSVM (One-Class SVM)** is completely absent. For high-dimensional telemetry with known normal-class structure, OCSVM with RBF kernel provides well-understood margin-based separation. It's less interpretable than your statistical methods but has rigorous VC-dimension theory behind it.

4. **Your benchmark is ESA OPS-SAT and synthetic simulator.** There is no evaluation on:
   - **SKAB (Skoltech Anomaly Benchmark)** — you have a `SKAB-S16` folder in your repo but the exploration doesn't show integration into the test harness.
   - **NASA SMAP/MSL (Mars Science Laboratory)** — the gold standard benchmark for spacecraft telemetry anomaly detection (Hundman et al., 2018). Your GRU and TCN are directly comparable to LSTM-AD from that paper. You should publish that comparison.
   - **Numenta NAB** — for streaming detection evaluation with temporal scoring.

5. **You cite "NASA ASIST standard" for persistence filter and "NASA GSFC Pattern" for incident grouper.** These are not citable. The actual NASA standards are:
   - **NPR 7150.2**: NASA Software Engineering Requirements
   - **NASA-STD-8719.13**: Software Safety Standard
   - **ECSS-E-ST-70-41C**: ESA Telemetry and Command standards
   **Using precise citations would dramatically increase credibility for academic/institutional customers.**

---

## PANEL 10 — The MATH PROFESSOR'S Final Verdict (Combinatorial Summary)

### The Three Core Mathematical Problems You Must Fix:

**Problem 1: You have no calibrated false positive rate.**
You cannot answer: "What is P(CRITICAL alert | no anomaly)?" The ensemble confidence of 0.85 -> CRITICAL has no theoretical or empirical grounding. Until you can answer this question, your "severity" levels are labels, not probabilities.

**Problem 2: Your residuals violate the independence assumption of every test.**
All 11 detectors assume either i.i.d. residuals or Markov residuals. Satellite telemetry residuals are: (a) autocorrelated, (b) potentially heteroskedastic, (c) potentially non-Gaussian (bimodal, heavy-tailed). You handle (c) partially with GMM-2. You handle (a) and (b) not at all. The consequences: systematically inflated false positive rates (for autocorrelated residuals) and missed anomalies during high-variance orbital phases (for heteroskedastic residuals).

**Problem 3: Your ensemble weights and thresholds were manually tuned without error bars.**
There is no train/validation/test split for threshold optimization. There is no confidence interval on your ">=95% F1" claim. There is no sensitivity analysis: "if threshold changes by +/-10%, how does F1 change?" A system in this state is in **hyper-parameter debt** — every production incident will require manual tuning with no principle to guide it.

---

## WHAT THIS SYSTEM DOES WELL (Even the Math Professor agrees)

1. **STL decomposition before detection is exactly right.** This is the single most important design decision and it's correct. Many commercial systems skip this and suffer constant eclipse false positives.

2. **Per-channel calibration with adaptive refresh is architecturally sound.** No global thresholds is the right call.

3. **MAD-based z-score (even if threshold is slightly wrong) is more robust than rolling std z-score.** Good production insight.

4. **Frozen dataclasses for core models** prevent mutation bugs that plague long-running detection systems.

5. **The explainability layer (causal chains + counterfactuals) is genuinely thoughtful** and differentiates this from academic implementations.

6. **SKAB-S16 folder in the repo** suggests awareness of the benchmark. Just needs integration.

---

## PRIORITY IMPROVEMENT ROADMAP

| Priority | Problem | Fix |
|---|---|---|
| **P0** | No calibrated false positive rate | Empirical calibration on holdout nominal data; publish P(alarm \| nominal) per severity |
| **P0** | Autocorrelated residuals + wrong ARL | Add Ljung-Box test; if autocorrelated, apply pre-whitening or use ARL tables for correlated CUSUM |
| **P0** | Ensemble weights are fiction | Logistic regression meta-learner on labeled ESA + SKAB data; derive weights from log-likelihood ratios |
| **P1** | GMM BIC threshold is dimensionally inconsistent | Replace 10 with `3 * ln(n_calibration_samples)` |
| **P1** | Auto z-threshold contaminated by anomalies | Use robust estimator (trimmed 99th percentile or MAD-based) for threshold |
| **P1** | PELT O(n^2) total cost | Replace with BOCPD (online, O(1) per sample) for streaming path |
| **P1** | ML training blocks event loop | Move to ProcessPoolExecutor or background worker |
| **P1** | Modified z-score threshold should be 3.5, not 3.0 | Fix per Iglewicz & Hoaglin (1993) |
| **P2** | No mission phase awareness | Add phase-gating config (at minimum: MANEUVER, SAFE_MODE suppress certain detectors) |
| **P2** | Alert cooldown is global | Make cooldown configurable per satellite, per parameter, per severity |
| **P2** | SKAB/SMAP/MSL not benchmarked | Run evaluation on standard benchmarks; publish comparison |
| **P2** | No distributed tracing | Add OpenTelemetry with trace_id propagation |
| **P3** | Dashboard is minimal JS | Proper React/TypeScript dashboard with Plotly time-series |
| **P3** | Missing precise citation (NASA/ESA standards) | Replace informal references with actual standard document numbers |

---

**Bottom line:** The architecture is sound and the engineering instincts are good. The statistical foundations are porous in specific, fixable ways. The path from "promising prototype" to "certifiable spacecraft anomaly detector" runs through the P0 items above — calibrated false positive rates and residual independence testing. Everything else is polish.

*Start with P0. They're not optional. They define whether this is science or engineering.*
