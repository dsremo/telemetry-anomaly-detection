# Dsremo Benchmark Results — Expert Debate
*Run date: 2026-03-31*
*"Don't celebrate numbers you don't understand."*

---

## THE ACTUAL NUMBERS (Context for all panels)

**ESA-MISSION1 — 5-Channel Quick Test (stat-only, 391s)**
| Channel    | Anomalies |
|------------|----------:|
| channel_43 |       130 |
| channel_44 |       176 |
| channel_41 |        87 |
| channel_42 |       101 |
| channel_45 |       132 |
| **TOTAL**  |   **626** |

- Severity: warning=335, watch=291
- Detectors: ewma=624, cusum=623, trend_velocity=144, changepoint=129, matrix_profile=61, bocpd=28, correlation_graph=1
- **No GT available — detection stats only**

**Full Benchmark (quick mode, scoring against existing DB anomalies)**
| Dataset            | Detections | GT  |  P      | R      | F1     | GT Source         |
|--------------------|------------|-----|---------|--------|--------|-------------------|
| CATS-2             | 4,642      | 200 | 0.0%    | 0.0%   | 0.0%   | Parquet y-col     |
| CATS-3             | 4,642      | 200 | 0.0%    | 0.0%   | 0.0%   | Parquet y-col     |
| CATS-S10-1MIN      | 413        | 200 | 7.3%    | 8.0%   | 7.6%   | Parquet y-col     |
| SKAB-VALVE2        | 18         |  4  | 100.0%  | 100.0% | 100.0% | **Proxy (self)**  |
| OPSSAT-3           | 340        | 36  | 100.0%  | 100.0% | 100.0% | **Proxy (self)**  |
| GECCO-WATER-S10    | 1,904      | 430 | 100.0%  | 100.0% | 100.0% | **Proxy (self)**  |
| GECCO-S16          | 799        | 327 | 100.0%  | 100.0% | 100.0% | **Proxy (self)**  |
| SATNOGS-25544      | 32         | N/A | —       | —      | —      | None              |
| ESA-MISSION1       | 10,792     | N/A | —       | —      | —      | None              |

---

## PANEL 1 — ML Researcher / Benchmark Integrity Expert

### "Your 100% scores are mathematically tautological. They prove nothing."

Four of your five scored datasets used **proxy GT** — ground truth derived by clustering your own detections. This means you are scoring your detector against labels it generated itself. The F1=100% is not a result; it is an identity:

```
GT_proxy = cluster(detected_events)
score(detected_events, GT_proxy) = 100%  ← always
```

This is like a student grading their own exam. The only datasets with external, independent ground truth were:
- **CATS** (parquet y-column — 200 real labeled windows) → **F1 = 0.0%**
- **CATS-S10-1MIN** → **F1 = 7.6%**

That is the actual honest performance on independently labeled data: **0–7.6% F1 on real spacecraft simulation data**.

**What this means for your pitch:** If a serious data scientist at Planet Labs or Spire looks at your benchmark page and asks "which GT is independent?", the answer is CATS only — and CATS scores 0-7.6%. Everything else is self-referential.

**What needs to happen before any credibility claim:**
1. Download the real SKAB labels from GitHub (`valve2/1.csv`, `2.csv`, `3.csv`) — you have the downloader, run it in full mode, not quick mode.
2. Download OPSSAT labels from Zenodo (record 8363509). Your downloader exists; the `--quick` flag skipped it.
3. Download GECCO labels from GitHub. Same issue.
4. Fix the CATS scoring pipeline (see Panel 3 below).

**Until those 4 steps are done, your benchmark results are not publishable.**

---

## PANEL 2 — Operations Engineer (SpaceX / Planet Labs perspective)

### "10,792 anomalies on ESA-MISSION1 is a false alarm catastrophe."

The ESA-MISSION1 dataset is **13 years of real ESA spacecraft telemetry** across 57 channels. The dataset contains **200 labeled anomaly events** in the metadata. Your system detected **10,792 anomalies**.

Let's do the arithmetic:

- 57 channels × 122,721 rows/channel = **~7 million data points**
- 10,792 anomalies = **0.15% of all data points flagged**
- Expected anomalies from metadata: ~200 events across 58 channels ≈ 3.4 per channel
- Your system detected: 10,792 / 57 ≈ **189 anomalies per channel**

That is **55× more anomalies than exist in the labeled ground truth**. Even accounting for the fact that the 200 GT labels are metadata-level (not per-channel timestamps), a 55× overdetection rate is a hard sell to any ops team.

A real operator would close your dashboard after the first shift. **Alert fatigue is not a UX problem — it is a correctness problem.**

The 5-channel quick test shows the same pattern: 626 anomalies across 5 channels in data that spans 13 years. That's 125 per channel per 13 years — but with a 60-hour cooldown, meaning you're seeing ~95 cooldown-distinct events per channel in 13 years. Each one requires operator triage. At 15 minutes per triage, that's ~24 hours of operator time per channel per year for what is mostly noise.

**Root cause from the detector breakdown:**
- ewma=624, cusum=623 out of 626 total anomalies
- EWMA and CUSUM fire on **99.7% of all anomalies**
- Every "anomaly" is just EWMA+CUSUM co-firing on a noisy ESA channel
- The ensemble's diversity — the whole point of 11 detectors — is completely absent here

**Fix required:** You need a minimum-detector-agreement threshold for ESA-style slowly-drifting historical data. A single co-firing of EWMA+CUSUM should NOT produce an alert on 13-year archival telemetry without requiring at least one structural detector (PELT, MatrixProfile, BOCPD, CorrelationGraph) to also fire.

---

## PANEL 3 — Statistician / Benchmark Validity Expert

### "CATS 0.0% F1 is a scoring pipeline failure, not a detection failure."

The CATS result is suspicious: 4,642 raw detections, 200 GT windows, F1=0.0%. This requires explanation.

The scoring code in `run_benchmark_auto.py` clusters detections with `gap_s=600` (10 minutes). 4,642 raw detections cluster down to **10 events** with a 10-minute merge gap. Then these 10 events are scored against 200 GT windows with `window_s=1800` (30-minute tolerance).

**The question is: are the detection timestamps from the DB aligned with the GT timestamps from the parquet file?**

The GT windows from the parquet span: `2023-01-12 15:11 → 2023-01-13 ...` (real CATS timestamps).

But the DB telemetry timestamps depend on how CATS data was imported. If the import used `datetime.now()` or synthetic timestamps instead of the original CATS timestamps, the detection timestamps could be entirely in a different time range — e.g., the date the data was loaded rather than the CATS simulation date range.

**Evidence:** The missed GT windows all have dates `2023-01-12` to `2023-01-13`. If the DB detections use 2026 timestamps (insertion dates), there is zero overlap and the scorer correctly returns TP=0.

**This is not a detection quality problem. This is a data pipeline integrity problem.**

**Action required:**
1. Query the DB: `SELECT MIN(timestamp), MAX(timestamp) FROM anomalies WHERE satellite_id='CATS-2'`
2. Compare against the parquet GT range: `2023-01-12`
3. If timestamps don't match → the CATS import used wrong timestamps → re-import with correct timestamps from the parquet index
4. Re-run the full benchmark with correct GT alignment

If this is the cause, the real CATS F1 may be much higher — possibly close to the Sprint 8 reported F1=17.7% for CATS-3. But that still needs to be verified with corrected timestamps.

**Secondary issue — gap parameter mismatch:** Even with correct timestamps, clustering 4,642 detections with gap=600s into only 10 events suggests CATS has very dense, closely spaced detections that cluster into a few massive super-events. This is characteristic of a channel with continuous drift that fires CUSUM/EWMA at every step. The actual GT windows are short (5-83 minutes per the missed window list). A 10-minute merge gap may be too aggressive — it merges distinct anomaly windows into one mega-event.

---

## PANEL 4 — Startup / Product Expert

### "You're one benchmark fix away from a credible pitch, but currently you can't pitch this."

Here's what a VP of Engineering at a potential customer sees:

**The good story (detector health):**
- 5-channel ESA test ran in 391 seconds stat-only — that's 5 × 122K = 610K data points in ~6.5 minutes. Throughput ≈ 1,570 pts/sec. That's respectable for a single-process Python pipeline.
- SatNOGS ISS (25544): 32 anomalies across 4 real channels from Oct 2025–Jan 2026 — this is clean, curated, and believable.
- SKAB performance (F1=100%) is believable once you replace proxy GT with real SKAB labels from GitHub — and historically this dataset has scored F1=90.9% (Sprint 19 notes), which is genuinely excellent.

**The bad story (what will get you rejected in a demo):**
- CATS: 0% F1 — even if it's a scoring bug, "0% F1" on your benchmark page is radioactive
- ESA-MISSION1: 10,792 anomalies with no ground truth — this is your flagship "real spacecraft" dataset and you cannot score it
- The 100% scores on SKAB/OPSSAT/GECCO in quick mode are self-referential — any technical buyer will spot this

**The path to a credible benchmark page:**
1. Fix CATS timestamp alignment (1 hour of work)
2. Run full mode (not --quick) to get real GT from GitHub/Zenodo for SKAB, GECCO, OPSSAT
3. Set a minimum-detector-agreement rule for ESA-MISSION1 (reduces false alarms from 10K to ~200 believable events)
4. Report ESA-MISSION1 as "detection coverage" (% channels with anomalies found) rather than raw counts
5. Update `docs/BENCHMARK_RESULTS.md` with these corrected numbers

**Revenue implication:** Spire Global, Planet Labs, Unseenlabs — these are the buyers. They will send their anomaly detection engineer to evaluate this. That engineer will immediately notice the proxy GT issue and the CATS 0%. Fix these before any outbound sales.

---

## PANEL 5 — ESA / Mission Operations Engineer

### "Your ensemble is not 11 detectors. It's CUSUM+EWMA with 9 decorators."

The Mission1 5-channel breakdown reveals the core structural problem:

```
ewma=624  (99.7% of all anomalies)
cusum=623 (99.5% of all anomalies)
trend_velocity=144 (23%)
changepoint=129 (20.6%)
matrix_profile=61 (9.7%)
bocpd=28 (4.5%)
correlation_graph=1 (0.16%)
```

**CUSUM and EWMA fire together on 99%+ of anomalies.** These two detectors are designed to detect mean shifts. On the ESA Mission1 data — 13 years of slowly evolving EPS telemetry — they are detecting the ordinary long-term aging drift of the satellite's power system. This is not anomalous; this is expected physics.

**The correlation_graph fired once across 626 anomalies** — 0.16%. This is the most sophisticated detector in the ensemble (Sprint 19, based on STGLR paper that achieved F1>0.97 on multi-channel correlation). It is essentially dormant.

**Why:** CUSUM/EWMA dominate the weight budget (0.17 + 0.14 = 0.31 = 31%). Even when they fire on nominal drift, the weighted confidence hits the alarm threshold because these two detectors together contribute nearly a third of total weight. The other 9 detectors cannot suppress a false alarm when CUSUM+EWMA both fire.

**This is a fundamental weighting architecture problem:**

For slowly-drifting archival spacecraft data like ESA-MISSION1, the correct behavior is:
- CUSUM/EWMA detect the drift → **pending, not alarm**
- Only escalate to alarm if at least one structural detector (PELT, BOCPD, MatrixProfile) also finds a structural break
- OR if CorrelationGraph finds multi-channel correlation change (suggesting a real spacecraft event, not aging)

**Recommendation:** Implement a two-tier alarm policy:
- **Tier 1 (sensitivity):** CUSUM or EWMA fires → "watch" status, buffered, not alerted
- **Tier 2 (specificity):** At least one of [PELT, BOCPD, MatrixProfile, CorrelationGraph] also fires → escalate to "warning" or "critical"

This would reduce ESA-MISSION1 false alarms by an estimated 80% while preserving sensitivity on datasets like SKAB and OPSSAT where structural detectors also fire.

---

## PANEL 6 — Data Engineering / Infrastructure Expert

### "The benchmark infrastructure itself has three correctness bugs."

**Bug 1: run_benchmark_auto.py scores against stale detections.**

The `run_benchmark_auto.py --quick` script queries the DB for existing anomalies and scores them. It does NOT re-run detection. This means you're scoring whatever anomalies happen to be in the DB right now — which could be from any previous sprint, with any parameter settings. The benchmark does not reflect current code behavior unless you explicitly clear and re-detect first.

Compare with `run_full_benchmark.py` which correctly calls `_clear_anomalies()` + `_clear_detector_state()` before each run. The auto runner has no such step.

**Action:** Add a `--fresh` flag to `run_benchmark_auto.py` that clears and re-detects before scoring. Or always mention in the output that these are "current DB detections" and when detection was last run.

**Bug 2: SKAB-S17-TUNED has 0 detections.**

The variant `SKAB-S17-TUNED` in the DATASETS list returned 0 detections and 0 GT windows. Either:
(a) This satellite was never detected against (DB is empty for this satellite_id)
(b) The satellite_id string doesn't match what's in the DB

This produces a silent omission in the summary table. **Add a check: if `len(detected) == 0 and len(gt_windows) == 0`, flag this as "NOT RUN" not "0 detections" to distinguish missing data from clean detection.**

**Bug 3: OPSSAT proxy GT is 36 windows, but the Sprint 19 notes say 17 GT windows from the OPS-SAT paper.**

When `--quick` mode skips the Zenodo download, the code falls back to:
```python
gt_windows = derive_gt_from_detections(detected, gap_s=300)
```
This creates **36 proxy GT windows** from 340 detections. The real paper reports **17 labeled events**. Your proxy-scored F1=100% because 36 proxy windows derived from 340 detections will trivially map back to those 340 detections. The Sprint 19 F1=75.6% with 17 real GT windows is the correct number. **Quick mode is showing 100% on a self-referential test instead of the real 75.6%.**

This is directly misleading. The output should clearly label "PROXY GT — DO NOT USE FOR REPORTING" on any dataset where GT was derived from detections.

---

## UPDATE — Fixes Applied (2026-03-31)

| Fix | Status |
|-----|--------|
| ESA-MISSION1 scored with local `labels.csv` (200 events, 58 ch) | ✅ Done — F1=5.1% (P=3.4%, R=10.1%) |
| Proxy GT rows flagged with `*` in benchmark summary table | ✅ Done — SKAB/OPSSAT/GECCO in quick mode marked `[PROXY]` |
| CATS re-detection started (ced1+cso1, full 5M rows, stat-only) | ⏳ Running background (PID 87120, ~30–40 min) |
| CATS timestamp alignment confirmed: 1M row cap = 11.57 days; GT starts day 11.57+91min | ✅ Root cause confirmed |

**Real honest scores (quick mode after fixes):**
| Dataset | P | R | F1 | GT Source | Reportable? |
|---|---|---|---|---|---|
| CATS-2 | 0.0% | 0.0% | 0.0% | Parquet (row cap bug — re-detecting) | ⏳ Pending |
| CATS-3 | 0.0% | 0.0% | 0.0% | Parquet (row cap bug — re-detecting) | ⏳ Pending |
| SKAB-VALVE2 | 75.0% | 100% | 85.7% | 3 real windows from GitHub | ✅ Yes |
| ESA-MISSION1 | 3.4% | 10.1% | 5.1% | Local labels.csv, 200 events | ✅ Yes |

---

## CONSENSUS — What Must Change Before Next Demo

| Priority | Action                                                                 | Estimated effort |
|----------|------------------------------------------------------------------------|------------------|
| **P0**   | Fix CATS timestamp alignment (query DB timestamps vs parquet dates)    | 30 min           |
| **P0**   | Run `run_full_benchmark.py` (not auto+quick) with fresh detection      | 2-6 hours        |
| **P0**   | Download real SKAB/OPSSAT/GECCO GT (remove --quick from benchmark)     | 30 min (network) |
| **P1**   | Add "PROXY GT — NOT FOR REPORTING" warning in benchmark output         | 1 hour           |
| **P1**   | Two-tier alarm policy for archival data (reduce ESA false alarms 80%)  | Sprint work      |
| **P1**   | Min-agreement rule: require ≥1 structural detector to co-fire          | Sprint work      |
| **P2**   | Add `--fresh` flag to `run_benchmark_auto.py`                          | 2 hours          |
| **P2**   | SKAB-S17-TUNED: investigate 0 detections (DB missing or wrong ID)      | 30 min           |
| **P2**   | ESA-MISSION1: report "detection coverage" not raw anomaly count        | 1 hour           |

**Bottom line:** The system detects real anomalies on clean industrial/spacecraft HK data (SKAB, OPSSAT) with good quality. The benchmark infrastructure obscures this with proxy GT inflation and CATS timestamp bugs. Fix the benchmark first, then the scores will tell the real story.
