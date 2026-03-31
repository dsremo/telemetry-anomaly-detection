# FMEA Traceability Matrix — Detector ↔ Failure Mode Mapping

Per NPR 7150.2 and NASA-STD-8719.13, each detector should be traceable to
specific failure modes it is designed to catch.

| Detector | FMEA Failure Mode | Example Channel | Detection Mechanism |
|----------|-------------------|-----------------|---------------------|
| CUSUM | Gradual performance degradation | BAT_CAP_Ah (battery capacity fade >2%/week) | Cumulative drift accumulation exceeding H threshold |
| EWMA | Sudden operating point shift | BUS_VOLTAGE (regulator failure, step change) | Level shift detection via exponential smoothing |
| Statistical (z-score) | Single-point transient spike | SIGNAL_STRENGTH (comms link dropout) | Instantaneous deviation from reference distribution |
| BOCPD | Structural regime change | WHEEL_SPEED_X (reaction wheel bearing degradation) | Bayesian posterior P(changepoint) via run-length distribution |
| Isolation Forest | Multi-parameter covariance anomaly | EPS subsystem (correlated voltage/current/temp) | Multivariate isolation in feature space |
| Variance | Noise floor increase / sensor degradation | PANEL_TEMP (thermistor aging, increased noise) | Rolling variance ratio vs calibrated reference |
| GRU Autoencoder | Complex temporal pattern anomaly | POINTING_ERROR (ADCS control loop oscillation) | Reconstruction MSE exceeding trained threshold |
| TCN | Long-range temporal dependency anomaly | SOLAR_ARRAY_CURRENT (eclipse transition pattern break) | Dilated causal convolution reconstruction error |
| Trend Velocity | Onset detection / early warning | BATTERY_TEMP (thermal runaway acceleration) | STL trend gradient exceeding σ-normalized threshold |
| Matrix Profile | Novel subsequence pattern | ANY (never-before-seen operational pattern) | Discord = max nearest-neighbor distance in matrix profile |
| Correlation Graph | Cross-channel decoupling | WHEEL_SPEED_{X,Y,Z} (bearing fault breaks correlation) | Rolling Pearson correlation deviation from calibrated baseline |
| Stale Data | Sensor dropout / comms loss | ANY (no new telemetry for >5 minutes) | Timestamp gap exceeding stale_threshold_s |

## Standards References
- **NPR 7150.2D**: NASA Software Engineering Requirements
- **NASA-STD-8719.13C**: Software Safety Standard
- **ECSS-E-ST-70-41C**: ESA Telemetry and Telecommand Packet Utilization
- **ECSS-Q-ST-30C**: Space Product Assurance — Dependability
