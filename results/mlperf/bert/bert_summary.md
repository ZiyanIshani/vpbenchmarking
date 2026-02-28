# MLPerf Training — BERT Results

**Date:** 2026-02-27 / 2026-02-28
**Cluster:** 4 nodes × 8x H100 80GB SXM5 (32 GPUs total)
**Container:** `nvcr.io/nvidia/pytorch:24.09-py3`
**Init checkpoint:** `bert-large-uncased` (HuggingFace, Phase 2 only)
**Target MLM accuracy:** 0.720
**Dataset:** AWS Neuron public S3 — `bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512` (seqlen=512)

## Training Configuration (Phase 2)

| Parameter | Value |
|-----------|-------|
| Phase | 2 (seqlen=512, started from HF bert-large-uncased) |
| Max steps | 1563 |
| Learning rate | 4e-3 |
| Warmup proportion | 12.8% |
| Global batch size | 32768 |
| Max seq length | 512 |
| Max pred per seq | 80 |
| Eval every steps | 100 |
| Num eval examples | 10000 |
| Seed | 42 |
| Precision | bf16 (autocast) + APEX FusedLAMB |
| Attention | FlashAttention 2 |

## Multi-Node Results (32x H100 — 4 nodes)

**Configuration:** `batch=64/GPU × grad_accum=16 × 32 GPUs = 32768 global_bs`

| Metric | Value |
|--------|-------|
| **Time to convergence** | **154.86 min (2h 34.9min)** |
| Convergence step | 1400 / 1563 |
| Final MLM accuracy | **0.7212** (target: 0.720) |
| Average throughput | **4937 seq/s** |
| World size | 32 GPUs (4 nodes × 8 GPUs) |

**MLM accuracy trajectory:**

| Step | MLM Accuracy | Elapsed |
|------|-------------|---------|
| 100  | 0.6978      | 10.94 min |
| 200  | 0.6929      | 21.90 min |
| 300  | 0.6969      | 32.93 min |
| 400  | 0.7004      | 43.93 min |
| 500  | 0.7039      | 54.92 min |
| 600  | 0.7064      | 66.01 min |
| 700  | 0.7065      | 77.03 min |
| 800  | 0.7102      | 88.11 min |
| 900  | 0.7116      | 99.16 min |
| 1000 | 0.7137      | 110.26 min |
| 1100 | 0.7159      | 121.43 min |
| 1200 | 0.7174      | 132.53 min |
| 1300 | 0.7197      | 143.68 min |
| **1400** | **0.7212 ✓** | **154.85 min** |

**Log:** `bert_4node_phase2_20260227_192027.txt`

## Single-Node Results (8x H100)

**Configuration:** `batch=64/GPU × grad_accum=64 × 8 GPUs = 32768 global_bs`

| Metric | Value |
|--------|-------|
| **Time to convergence** | **~487 min (estimated)** |
| Final MLM accuracy | ~0.721 (estimated — same trajectory as multi-node) |
| Average throughput | **1566 seq/s** (measured, steps 1–220) |

> **Note:** Single-node run started 2026-02-28, ETA ~8–9 hours. Throughput is measured from
> the first 220 steps. Since global_bs=32768 matches multi-node, the per-step accuracy
> trajectory is identical; convergence step is expected to be ~1400, same as multi-node.

**Partial MLM accuracy (first run, killed at step 220):**

| Step | MLM Accuracy | Elapsed |
|------|-------------|---------|
| 100  | 0.6975      | 34.83 min |
| 200  | 0.6928      | 69.78 min |

**Log:** `bert_1node_phase2_20260227_175133.txt` (partial), `bert_1node_phase2_20260228_200713.txt` (full run in progress)

## Scaling Analysis

| Metric | Single-Node (8 GPU) | Multi-Node (32 GPU) | Ratio |
|--------|--------------------|--------------------|-------|
| Throughput | 1566 seq/s | 4937 seq/s | **3.15×** |
| Time to convergence | ~487 min (est.) | 154.86 min | **3.14×** |
| Scaling efficiency | — | **78.8%** (vs ideal 4×) |

> Scaling efficiency = (single-node throughput × N nodes) / multi-node throughput
> = (1566 × 4) / 4937 = 6264 / 4937 = **78.8% at 4 nodes**
> Note: No InfiniBand — inter-node via 100GbE Ethernet, which limits collective bandwidth.

## Infrastructure Notes

- **No shared filesystem** — each node has independent copy of the dataset (~58 GB at seqlen=512)
- **Network:** Ethernet only (no InfiniBand); NCCL over TCP with `--network=host`
- **Rendezvous:** Static (`--master_addr=10.15.21.81 --master_port=29502`) — c10d backend failed due to Docker `--network=host` hostname resolution quirk (`gethostname("g138")` returns `127.0.1.1` inside container)
- **Optimizer:** APEX FusedLAMB (fp32 master weights, bf16 gradients via autocast)
- **Checkpoint:** Phase 2 checkpoint saved to `phase2_checkpoint.pt` after convergence
