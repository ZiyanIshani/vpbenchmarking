# Voltage Park Benchmark — Multi-Node NCCL Tests

## Goal
Run cross-node NCCL collective benchmarks across 4 nodes x 8 H100 GPUs = 32 GPUs total.
Measure bandwidth (GB/s) and latency (µs) for AllReduce, AllGather, and Broadcast over
3.2 Tbps InfiniBand.

## Cluster
| Node | Internal IP   |
|------|--------------|
| g138 | 10.15.21.81  |
| n2   | 10.15.25.105 |
| n3   | 10.15.28.33  |
| n4   | 10.15.22.49  |

Primary node (run all commands from here): 10.15.21.81
SSH access to all other nodes is passwordless as ubuntu@<ip>.

## Project Structure
~/vp-benchmarks/
├── CLAUDE.md
├── hostfile
├── results/
│   └── nccl_multinode/
│       ├── nccl_allreduce_<timestamp>.txt
│       ├── nccl_allgather_<timestamp>.txt
│       ├── nccl_broadcast_<timestamp>.txt
│       └── nccl_multinode_summary.md
└── scripts/
    └── nccl/
        └── nccl-tests/               ← already built on primary node

## Step 1: Create hostfile
```bash
cat > ~/vp-benchmarks/hostfile << 'HOSTEOF'
10.15.21.81 slots=8
10.15.25.105 slots=8
10.15.28.33 slots=8
10.15.22.49 slots=8
HOSTEOF
```

## Step 2: Install nccl-tests on all other nodes
SSH into each of the 3 remote nodes and run the following.
nccl-tests is already built on 10.15.21.81 — replicate to others:
```bash
for NODE in 10.15.25.105 10.15.28.33 10.15.22.49; do
  echo "Setting up $NODE..."
  ssh ubuntu@$NODE "mkdir -p ~/vp-benchmarks/scripts/nccl"
  scp -r ~/vp-benchmarks/scripts/nccl/nccl-tests ubuntu@$NODE:~/vp-benchmarks/scripts/nccl/
  # Verify GPU visibility on each node
  ssh ubuntu@$NODE "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
done
```

If scp fails or binaries are incompatible, build from source on each node instead:
```bash
for NODE in 10.15.25.105 10.15.28.33 10.15.22.49; do
  ssh ubuntu@$NODE "
    sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev libnccl2 libnccl-dev &&
    mkdir -p ~/vp-benchmarks/scripts/nccl &&
    cd ~/vp-benchmarks/scripts/nccl &&
    git clone https://github.com/NVIDIA/nccl-tests.git &&
    cd nccl-tests &&
    make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu
  "
done
```

## Step 3: Verify cluster connectivity
```bash
# Verify all 32 GPUs visible across nodes
for NODE in 10.15.21.81 10.15.25.105 10.15.28.33 10.15.22.49; do
  echo -n "$NODE GPUs: "
  ssh ubuntu@$NODE "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
done

# Test MPI across all nodes
mpirun -np 4 --hostfile ~/vp-benchmarks/hostfile \
  -x NCCL_DEBUG=WARN \
  hostname
```

If MPI fails, try adding: `--mca btl_tcp_if_include enp157s0f0np0`
(this is the active network interface on the primary node)

## Step 4: Run benchmarks
```bash
mkdir -p ~/vp-benchmarks/results/nccl_multinode
cd ~/vp-benchmarks
TS=$(date +%Y%m%d_%H%M%S)

MPIFLAGS="-np 32 \
  --hostfile hostfile \
  --map-by ppr:8:node \
  -x NCCL_DEBUG=WARN \
  -x NCCL_SOCKET_IFNAME=enp157s0f0np0 \
  -x NCCL_IB_DISABLE=0 \
  -x LD_LIBRARY_PATH \
  --mca btl_tcp_if_include enp157s0f0np0"

# AllReduce
mpirun $MPIFLAGS \
  scripts/nccl/nccl-tests/build/all_reduce_perf \
  -b 8 -e 8G -f 2 -g 1 2>&1 | tee results/nccl_multinode/nccl_allreduce_${TS}.txt

# AllGather
mpirun $MPIFLAGS \
  scripts/nccl/nccl-tests/build/all_gather_perf \
  -b 8 -e 8G -f 2 -g 1 2>&1 | tee results/nccl_multinode/nccl_allgather_${TS}.txt

# Broadcast
mpirun $MPIFLAGS \
  scripts/nccl/nccl-tests/build/broadcast_perf \
  -b 8 -e 8G -f 2 -g 1 2>&1 | tee results/nccl_multinode/nccl_broadcast_${TS}.txt
```

## Step 5: Write summary
Parse peak bandwidth and minimum latency from each result file and write to
~/vp-benchmarks/results/nccl_multinode/nccl_multinode_summary.md:

# NCCL Multi-Node Benchmark Results
Date: <timestamp>
Nodes: 4 x g138-cluster (10.15.21.81, 10.15.25.105, 10.15.28.33, 10.15.22.49)
GPUs: 32 x NVIDIA H100 80GB HBM3 (8 per node)
Interconnect: 3.2 Tbps InfiniBand
CUDA: 12.6
NCCL: 2.24.3

| Test      | Message Size | Algo BW (GB/s) | Bus BW (GB/s) | Latency (µs) |
|-----------|-------------|----------------|---------------|-------------|
| AllReduce | ...         | ...            | ...           | ...         |
| AllGather | ...         | ...            | ...           | ...         |
| Broadcast | ...         | ...            | ...           | ...         |

### Comparison vs Single-Node
| Test      | Single-Node Bus BW | Multi-Node Bus BW | Efficiency % |
|-----------|--------------------|-------------------|-------------|
| AllReduce | 479.72 GB/s        | ...               | ...         |
| AllGather | 183.09 GB/s        | ...               | ...         |
| Broadcast | 212.86 GB/s        | ...               | ...         |

## Step 6: Commit results
```bash
cd ~/vp-benchmarks
git add results/nccl_multinode/nccl_multinode_summary.md hostfile
git commit -m "results: multi-node NCCL benchmarks $(date +%Y%m%d_%H%M%S)"
git push origin master
```

## Error Handling
- If MPI can't find hosts: check hostfile formatting, verify SSH works manually
- If NCCL falls back to TCP: set NCCL_IB_DISABLE=0 and verify ibstat shows active IB ports
- If bandwidth is unexpectedly low (<50 GB/s): run ibstat to check IB link speed
- If a node is unreachable mid-run: verify with ping, do not proceed until all 4 nodes respond
- Fix all errors before running the next collective

## Done
When nccl_multinode_summary.md is written and committed, stop and report results.
Do not run any other benchmarks.
