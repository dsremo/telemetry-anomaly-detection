# Sentinel — Cold Outreach Email Templates

---

## Template A — Space Startup (Series A/B, operating constellation)

**Subject:** Catching the anomaly your engineer missed last Thursday

---

Hi [Name],

I noticed [Company] is now operating [X] satellites. Congratulations — that's a real scaling milestone.

Here's the problem that hits every operator at your stage: your telemetry volume just outgrew your team's ability to watch it manually. The anomaly that causes your first in-orbit failure probably won't be dramatic — it'll be a 0.3-sigma voltage drift that nobody flagged because the dashboard looked "normal."

**I built Sentinel to catch exactly that.**

Sentinel is a multi-tenant anomaly detection engine for satellite telemetry. You push JSON (or connect YAMCS / InfluxDB / SatNOGS), and within 30 minutes you get:

- Real-time anomaly detection across all parameters, all satellites
- Confidence scores + root-cause explanations (not just alert noise)
- Webhook/email routing so the right engineer gets paged — not everyone

**What it found on real data:**

Blind test on ISS telemetry (5,000 frames, no prior knowledge):
- Detected the Oct 19 post-EVA RF power cycling the same day it happened
- Caught the Jan 2026 medical emergency event — all 3 detectors fired
- Zero false positives on normal orbital operations

On ESA Mission 1: 8,795 anomalies across 7.1M telemetry points. 39 critical. Would your team have caught all 39?

I'd love to show you a 20-minute live demo with your own satellite's telemetry format. No commitment, no pitch deck — just the tool running on real data.

Worth a look?

[Your name]
[Email] | [LinkedIn]

P.S. It runs on a $50/mo VPS. Your ops budget won't feel it.

---

## Template B — Traditional Satellite Operator (Tier 1/2)

**Subject:** Your anomaly detection is probably 10 years behind your satellites

---

Hi [Name],

[Company]'s [fleet/mission] has been operating for [X] years. The ground segment tools that got you here may not be the right ones for the next decade.

Most operators I talk to are still running threshold-based alerting with Excel runbooks. That works until you have a slow battery cell degradation that stays inside nominal bounds for 6 weeks before it becomes a mission risk.

**Sentinel uses a 5-algorithm ensemble** (Z-score, Isolation Forest, CUSUM, changepoint detection, rolling variance) to find what thresholds miss. Anomalies are flagged only when 2+ detectors agree — meaning your on-call engineer gets paged when something is actually wrong.

Three things that differentiate us from basic monitoring tools:

1. **Root-cause explanation** — not just "anomaly detected" but "battery_voltage dropped 2.8σ, CUSUM drift S=0.34, possible cell degradation or thermal event"
2. **Multi-tenant from the ground up** — run one instance for your entire fleet, full data isolation between missions
3. **Validated on real spacecraft** — ISS, ESA Mission 1, SatNOGS constellation

I can have you ingesting telemetry and seeing results within a single afternoon. Happy to do a trial on a non-critical mission segment first.

Do you have 20 minutes this week or next?

[Your name]
[Email]

---

## Template C — Space Agency / Research Institution

**Subject:** Open to trying an anomaly detection engine on your archive data?

---

Hi [Name / Team],

I'm reaching out because [Agency/Lab] has one of the most interesting telemetry archives in the field. I'd like to propose something: let us run Sentinel against a portion of your historical data and share the results with you — at no cost.

Sentinel is a production anomaly detection platform for satellite telemetry. We use a 5-detector ensemble that recently achieved 100% detection rate on ISS anomaly events (blind test, results cross-referenced with NASA/AMSAT public records).

We're validating against NASA SMAP/MSL telemetry right now (the Hundman et al. benchmark) and would welcome the chance to run against a mission with your team's domain knowledge to help interpret results.

What we'd need: a CSV export or API access to telemetry from any completed or non-operational mission.
What you'd get: a full anomaly report with severity classification, root-cause grouping, and comparison against any known anomaly log you're willing to share.

No strings attached — this is validation work that benefits both sides.

Is there someone on your team who handles telemetry data sharing agreements?

[Your name]
[Email] | [Organization]

---

## Template D — VC / Accelerator (If raising)

**Subject:** Sentinel — anomaly detection for the 10,000-satellite decade

---

Hi [Name],

The number of operational satellites doubles every 18 months. The number of satellite engineers grows at roughly the rate of any other technical specialty — maybe 10% per year.

That gap is Sentinel's market.

Sentinel is a SaaS anomaly detection engine for satellite telemetry. We sit between the spacecraft and the operations team — ingesting telemetry from any source (YAMCS, InfluxDB, SatNOGS, direct API) and surfacing real anomalies with explanations before they become failures.

**Traction:**
- Working product, deployed on real ISS and ESA data
- 7.1M telemetry points ingested, 8,795 anomalies detected (validated)
- Multi-tenant architecture supports constellation operators natively
- Pricing designed for the NewSpace budget (not the traditional defense contractor budget)

**Why now:**
- NewSpace operators cannot afford dedicated FDIR engineers per satellite
- Insurance underwriters are starting to require anomaly logs
- In-orbit servicing missions (Astroscale, ClearSpace) need predictive targeting

I'm raising a [pre-seed / seed] round to fund the first 10 paying customers and the sales motion. Happy to share the deck and a live demo.

10 minutes to talk?

[Your name]
[Email] | [LinkedIn]

---

## Follow-up Template (7 days after no reply)

**Subject:** Re: [original subject]

---

Hi [Name],

Quick follow-up — I realize my previous note landed during what was probably a busy week.

One specific thing that might be relevant to [Company]: we recently detected a propulsion anomaly pattern on ESA Mission 1 data that standard threshold alerting had been ignoring for months. The signature was in the correlation between two parameters, not in either one individually.

If that's a problem you recognize, Sentinel is worth 20 minutes of your time.

If the timing is wrong, happy to check back in [month]. Just let me know.

[Your name]

---

## Subject Line Variations (A/B test these)

**Pain-focused:**
- "The anomaly your team will miss this week"
- "Your satellite's next failure has already started"
- "Catching the 0.3-sigma drift before it becomes a mission loss"

**Curiosity-focused:**
- "We detected 4 ISS events before they were publicly confirmed"
- "What 7 million telemetry points taught us about battery degradation"
- "Why your threshold alerts are 6 weeks too late"

**Direct:**
- "Anomaly detection for [Company]'s constellation — worth 20 minutes?"
- "Satellite telemetry AI — pilot offer for [Company]"
- "Sentinel: real-time anomaly detection, live in 30 minutes"
