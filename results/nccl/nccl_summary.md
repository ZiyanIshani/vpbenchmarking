# NCCL Benchmark Results
Date: 2026-02-24 22:34:44
Node: g138
GPUs: 8 x NVIDIA H100 80GB HBM3
CUDA: 12.6 (driver 560.35.03)
NCCL: 2.24.3

## Notes
- Benchmarks run single-process with 8 GPUs (`-g 8`) due to MPI process-discovery issues in this environment
- Message range: 8B to 8G, step factor 2, 20 iterations per size

| Test      | Message Size | Algo BW (GB/s) | Bus BW (GB/s) | Latency (µs) |
|-----------|-------------|----------------|---------------|-------------|
| AllReduce | 8 GiB       | 274.13         | 479.72        | 32.55        |
| AllGather | 256 MiB     | 209.25         | 183.09        | 46.42        |
| Broadcast | 128 MiB     | 212.86         | 212.86        | 46.00        |

## Detailed Results

### AllReduce (peak at largest message)
Peak algo BW at 8 GiB: 274.13 GB/s (out-of-place), bus BW: 479.72 GB/s
Min latency at 8 B: 32.55 µs (in-place)
Avg bus bandwidth: 146.21 GB/s

### AllGather (peak at 256 MiB)
Peak algo BW at 256 MiB: 209.25 GB/s (out-of-place), bus BW: 183.09 GB/s
Min latency at 4096 B: 46.42 µs (out-of-place)
Avg bus bandwidth: 27.51 GB/s

### Broadcast (peak at 128 MiB)
Peak algo BW at 128 MiB: 212.86 GB/s (out-of-place), bus BW: 212.86 GB/s
Min latency at 8 B: 46.00 µs (out-of-place)
Avg bus bandwidth: 40.74 GB/s
