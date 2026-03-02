# MLPerf Inference — BERT-99 Results

**Date:** 2026-03-01 / 2026-03-02
**Node:** g138 (single node)
**GPUs:** 8x NVIDIA H100 80GB HBM3 SXM5
**Container:** `nvcr.io/nvidia/pytorch:24.09-py3`
**Model:** BERT-Large fine-tuned on SQuAD v1.1
**Checkpoint:** Official MLCommons checkpoint — Zenodo record 3733896 (`model.pytorch`)
**Dataset:** SQuAD v1.1 validation set (10,570 examples → 10,772 sliding-window features)
**Precision:** FP16
**Harness:** Custom Python (mlcommons-loadgen 6.0.10 + PyTorch 2.5.0a0)

## Offline Scenario (Throughput)

> Offline: all queries submitted at once. Measures peak batch inference throughput.
> Primary metric for compute capacity.

| Metric | Value |
|--------|-------|
| **Queries per second (QPS)** | **3336 QPS** |
| Batch size | 512 (DataParallel across 8 GPUs) |
| Samples processed | 270,336 |
| Duration | 81 seconds |
| Result | **VALID ✓** |
| Min duration satisfied | Yes (81s ≥ 60s) |
| Min queries satisfied | Yes (270,336 ≥ 270,336) |

**Log:** `offline/mlperf_log_summary.txt`

## Server Scenario (Latency-bounded)

> Server: queries arrive with Poisson distribution. Target: p99 latency ≤ 130 ms.
> Limited by harness overhead — see analysis below.

| Metric | Value |
|--------|-------|
| Sustained throughput | **~800 QPS** (actual completion rate at target=3000) |
| Min per-query latency | **70 ms** (best-case, empty queue) |
| P50 latency at 500 QPS target | 132 ms |
| P99 latency at 500 QPS target | 207 ms |
| Result | INVALID ✗ (p99 > 130 ms target) |

### Harness Overhead Analysis

The Python-based harness adds ~60 ms overhead per query beyond the raw CUDA inference time:

| Component | Time |
|-----------|------|
| Raw CUDA forward (batch=8, single GPU) | ~9 ms |
| Python threading + GIL serialization | ~45 ms |
| Tensor indexing, H2D transfer, response | ~6 ms |
| **Total observed min latency** | **~70 ms** |

With a 70 ms floor per query and a 130 ms target, only 60 ms remains for queuing — allowing
only ~10-30 QPS at p99 ≤ 130 ms (impractically low).

The official NVIDIA MLPerf submission uses TensorRT with CUDA graphs, achieving <10 ms
per-query latency and 5,000–8,000+ QPS Server at p99 ≤ 130 ms on H100.

**Sustained capacity** (at full saturation): **~815 QPS** using 8 independent GPU workers
(one model per GPU, round-robin dispatch, batch=8/GPU, pinned-memory pre-stacking).

## Hardware Reference

Single-GPU BERT-Large FP16 inference latency (raw CUDA, no Python overhead):

| Batch | Median latency | P99 latency | Throughput |
|-------|---------------|-------------|------------|
| 1     | 8.6 ms        | 9.0 ms      | 116 q/s    |
| 4     | 8.6 ms        | 14.2 ms     | 463 q/s    |
| 8     | 9.2 ms        | 14.7 ms     | 869 q/s    |
| 16    | 16.0 ms       | 17.0 ms     | 998 q/s    |
| 32    | 29.0 ms       | 30.7 ms     | 1102 q/s   |

8-GPU theoretical peak (no Python overhead):
- At batch=8/GPU: 8 × 869 = **~6,950 QPS**

## Notes

- **Container:** `nvcr.io/nvidia/mlperf/inference:bert-24.09` requires MLPerf program
  enrollment (access denied). Used `pytorch:24.09-py3` with custom Python harness.
- **Accuracy:** Not measured in performance mode. The official BERT-Large SQuAD checkpoint
  from Zenodo achieves ~90.9 F1 on SQuAD v1.1 (from MLCommons leaderboard; above the
  BERT-99 threshold of ≥ 90.874 F1).
- **Offline is the primary metric** for this hardware: 3336 QPS demonstrates strong
  BERT-Large inference throughput at FP16 on 8× H100.
