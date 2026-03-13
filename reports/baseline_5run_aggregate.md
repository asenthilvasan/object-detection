# Baseline 5-Run Aggregate Report
## Ray Serve YOLOv5s Object Detection — Open-Loop Profiling

**Date:** 2026-03-10
**Cluster:** Nautilus/NRP (`ml-pipelines`)
**Branch:** `kubernetes-deployment`
**Runs:** 5 identical open-loop load tests, 9 RPS stages each (5–100), 210 s cooldown.

---

## 5A. Aggregate Table A — k6 Summary Across Runs

| rps_target | p50_mean (ms) | p50_std | p99_mean (ms) | p99_std | throughput_mean | throughput_std | drain_mean (s) | drain_std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 55.8 | 1.3 | 79.0 | 6.0 | 5.02 | 0.01 | 0.0 | 0.0 |
| 10 | 55.4 | 1.3 | 77.2 | 8.6 | 10.02 | 0.01 | 0.0 | 0.0 |
| 20 | 63.6 | 1.1 | 100.2 | 54.7 | 20.01 | 0.01 | 0.0 | 0.0 |
| 30 | 56.6 | 0.9 | 213.7 | 38.3 | 30.01 | 0.01 | 0.0 | 0.0 |
| 40 | 67.4 | 4.0 | 246.8 | 52.5 | 40.01 | 0.01 | 0.0 | 0.0 |
| 50 | 3456.9 | 460.2 | 9883.8 | 2329.6 | 50.01 | 0.01 | 0.0 | 0.0 |
| 60 | 7154.9 | 505.5 | 28859.0 | 1872.2 | 60.01 | 0.01 | 0.0 | 0.0 |
| 80 | 18432.0 | 814.9 | 60371.3 | 5233.5 | 75.98 | 4.12 | 3.3 | 3.5 |
| 100 | 29472.3 | 1473.7 | 85962.0 | 4154.7 | 68.24 | 3.51 | 28.1 | 4.5 |

All 5 runs had **zero dropped iterations** and **100% OK rate** at all stages. No run had timeout-censored data (max_ms < 180,000 for all rows).

---

## 5B. Aggregate Table B — Latency Decomposition

| rps_target | proxy_queue_p50 (mean +/- std) | handle_roundtrip_p50 (mean +/- std) | gpu_duty_pct (mean +/- std) | gap_ms (mean +/- std) |
|---:|---:|---:|---:|---:|
| 5 | 15.2 +/- 0.0 | 38 +/- 0 | 3.9 +/- 0.1 | -0 +/- 1 |
| 10 | 15.0 +/- 0.0 | 38 +/- 0 | 7.7 +/- 0.2 | -0 +/- 1 |
| 20 | 15.0 +/- 0.0 | 38 +/- 0 | 15.6 +/- 0.2 | 7 +/- 1 |
| 30 | 15.3 +/- 0.0 | 39 +/- 0 | 23.2 +/- 0.7 | -0 +/- 1 |
| 40 | 15.0 +/- 0.0 | 71 +/- 58 | 32.1 +/- 0.6 | -22 +/- 54 |
| 50 | 155.3 +/- 302.6 | 1881 +/- 732 | 35.8 +/- 0.6 | 1417 +/- 683 |
| 60 | 599.7 +/- 249.6 | 2755 +/- 190 | 41.7 +/- 0.5 | 3796 +/- 578 |
| 80 | 1469.7 +/- 460.2 | 2954 +/- 95 | 42.4 +/- 1.1 | 14004 +/- 541 |
| 100 | 2072.8 +/- 556.8 | 2864 +/- 203 | 42.8 +/- 1.8 | 24531 +/- 1605 |

**gap_ms at rps=5 (validation):** -0.4 +/- 1.3 ms — PASS (< 20ms).

---

## 5C. Stability Assessment

| rps_target | p50_mean | p50_std | CV | Status |
|---:|---:|---:|---:|---|
| 5 | 55.8 | 1.3 | 0.023 | Stable |
| 10 | 55.4 | 1.3 | 0.024 | Stable |
| 20 | 63.6 | 1.1 | 0.018 | Stable |
| 30 | 56.6 | 0.9 | 0.016 | Stable |
| 40 | 67.4 | 4.0 | 0.060 | Stable |
| 50 | 3456.9 | 460.2 | 0.133 | Stable |
| 60 | 7154.9 | 505.5 | 0.071 | Stable |
| 80 | 18432.0 | 814.9 | 0.044 | Stable |
| 100 | 29472.3 | 1473.7 | 0.050 | Stable |

All stages have CV < 0.15. Results are stable across the 5 runs.

---

## 5D. Service Rate Estimate

### Method 1: STAGES Log (weighted average)
mu = **48.2 +/- 0.8 RPS** (across 5 runs)

Computed as `sum(batch_sizes) / sum(total_ms) * 1000` from all STAGES log entries per run.

### Method 2: Batch Size 10 (peak throughput)
mu = **43.7 +/- 4.9 RPS** (across 5 runs)

Computed as `10 / mean_total_ms_at_batch_10 * 1000`.

The two methods agree within their error bars. The service rate ceiling is **~48 RPS**.

---

## 5E. GPU Duty Cycle (Corrected)

GPU duty at saturation (rps 50–80), queried at `stage_start + 90s` (corrected timing per METRICS_SPEC.md):

**40.0 +/- 0.6%**

| Run | GPU duty at saturation (%) |
|---:|---:|
| 1 | 39.8 |
| 2 | 39.7 |
| 3 | 40.6 |
| 4 | 40.6 |
| 5 | 39.2 |

This confirms the GPU is active only ~40% of the time at saturation. The remaining ~60% is idle time caused by the serial pipeline (`max_concurrent_batches=1`).

---

## 5F. Summary Finding

The 5-run baseline confirms the following with statistical confidence:

- **Service rate mu = 48.2 +/- 0.8 RPS** (Method 1) / **43.7 +/- 4.9 RPS** (Method 2). The pipeline saturates at approximately **48 RPS** with `max_concurrent_batches=1`.

- **Saturation knee at RPS 80** — the first stage where `drain_time_s` exceeds 1 s. Below this point, all requests complete within the 60 s injection window. Above it, the dispatch queue builds unboundedly.

- **GPU duty cycle at saturation: 40.0 +/- 0.6%** (corrected, queried at `stage_start + 90s`). The GPU is idle > 60% of the time even at full load.

- **Primary bottleneck: `max_concurrent_batches=1` serializes the GPU-to-CPU pipeline.** GPU inference cannot overlap with CPU postprocess. At batch size 10, inference takes ~130 ms and postprocess ~75 ms. Total ~210 ms per batch = ~48 RPS max. Setting `max_concurrent_batches=2` would allow batch N+1 to begin GPU inference while batch N completes CPU postprocess, predicted to increase throughput by 30–50% (to ~65–72 RPS) and raise GPU duty to ~55–65%.

---

## Validation Checklist

| Check | Status |
|---|---|
| 8 charts in each run_N_charts/ directory | PASS |
| run_1_baseline.md through run_5_baseline.md exist | PASS |
| baseline_5run_aggregate.md exists | PASS (this file) |
| No run has dropped > 0 | PASS (all 5 runs, all stages) |
| No TIMEOUT-CENSORED stages | PASS (max_ms < 180,000 all rows all runs) |
| Table B gap_ms < 20ms at rps=5 | PASS (-0.4ms) |
| GPU duty > 20% at rps >= 50 (corrected timing) | PASS (40.0%) |

---

*Individual run reports: `reports/run_1_baseline.md` through `reports/run_5_baseline.md`*
*Individual run charts: `reports/run_1_charts/` through `reports/run_5_charts/`*
*Raw data: `reports/raw/run_1/` through `reports/raw/run_5/`*
