# STREAM / NVBench Memory Bandwidth Results
Date: 2026-02-25 18:31
Node: g138 (10.15.21.81)
GPUs: 8x NVIDIA H100 80GB HBM3
CPUs: 104x Intel Xeon Platinum 8470 (104 threads)
CUDA: 12.6

## STREAM — CPU Host Memory Bandwidth
| Operation | Bandwidth (GB/s) |
|-----------|-----------------|
| Copy      | 450.1           |
| Scale     | 431.1           |
| Add       | 485.1           |
| Triad     | 486.7           |

**Theoretical peak:** Intel Xeon Platinum 8470 = 307.2 GB/s per socket,
2 sockets = ~614 GB/s total DDR5 bandwidth

## NVBench — GPU Transfer Bandwidth
| Transfer Direction     | Peak Bandwidth (GB/s) |
|------------------------|----------------------|
| Host to Device (H2D)   | 54.8                 |
| Device to Host (D2H)   | 46.8                 |
| Device to Device (D2D) | 2903.4               |

**H100 HBM3 theoretical peak:** 3.35 TB/s device memory bandwidth

**Note on D2H:** D2H peaks at 46.8 GB/s for small transfers (~900 KB) but falls to ~20.5 GB/s at large sizes (>2 MB), likely due to NUMA effects (pinned host memory allocated on a remote NUMA node relative to the GPU).

## Analysis
- STREAM Triad efficiency: 486.7 / 614 GB/s = **79.3%**
- H100 D2D efficiency: 2903.4 / 3350 GB/s = **86.7%** (peak at 13 MB transfer size)
- H2D PCIe efficiency: ~54.8 GB/s — consistent with PCIe 5.0 x16 (63 GB/s theoretical), ~87% utilization
- D2H large-transfer drop from ~47 to ~20.5 GB/s suggests NUMA pinned-memory allocation mismatch; H2D is unaffected

## Raw Log Files
- stream_20260225_183010.txt
- bandwidth_h2d_d2h_20260225_183119.txt
- bandwidth_d2d_20260225_183138.txt

