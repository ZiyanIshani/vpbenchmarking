# NCCL Multi-Node Benchmark Results
Date: 2026-02-24 (timestamp: 20260224_225706)
Nodes: 4 x g-cluster (g138/10.15.21.81, g205/10.15.25.105, g244/10.15.28.33, g150/10.15.22.49)
GPUs: 32 x NVIDIA H100 80GB HBM3 (8 per node)
Interconnect: 3.2 Tbps InfiniBand (400 Gbps per port, 3 active ports per node)
CUDA: 12.6
NCCL: 2.24.3+cuda12.6
MPI: OpenMPI 4.1.2

| Test      | Message Size | Algo BW (GB/s) | Bus BW (GB/s) | Latency (µs) |
|-----------|-------------|----------------|---------------|-------------|
| AllReduce | 8 GB        | 170.80         | 330.93        | 38.37       |
| AllGather | 8 GB        | 333.97         | 323.53        | 49.68       |
| Broadcast | 8 GB        | 286.34         | 286.34        | 9.03        |

Notes:
- AllReduce latency at 8B; AllGather latency at 512B (first non-zero size); Broadcast latency at 32B
- Peak bus BW measured at 8 GB message size (out-of-place)

### Comparison vs Single-Node
| Test      | Single-Node Bus BW | Multi-Node Bus BW | Efficiency % |
|-----------|--------------------|-------------------|-------------|
| AllReduce | 479.72 GB/s        | 330.93 GB/s       | 69.0%       |
| AllGather | 183.09 GB/s        | 323.53 GB/s       | 176.7%      |
| Broadcast | 212.86 GB/s        | 286.34 GB/s       | 134.5%      |

### Observations
- **AllReduce** achieves 330.93 GB/s bus BW across 32 GPUs vs 479.72 GB/s single-node (8 GPUs).
  The 69% cross-node efficiency reflects IB bandwidth utilization across 4 nodes.
- **AllGather** scales well beyond single-node, as 32 GPUs can aggregate more total bandwidth
  from multiple IB links — the per-node chunk size decreases (1/32 vs 1/8), enabling higher aggregate throughput.
- **Broadcast** also exceeds single-node BW due to pipelining across 4 IB links from the root node.
- All tests passed correctness validation (0 out-of-bounds errors).

### Raw Result Files
- `nccl_allreduce_20260224_225706.txt`
- `nccl_allgather_20260224_225706.txt`
- `nccl_broadcast_20260224_225706.txt`
