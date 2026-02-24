# Voltage Park Benchmark — NCCL Tests

## Goal
Run NCCL collective communication benchmarks across all available GPUs on this node.
Measure bandwidth (GB/s) and latency (µs) for AllReduce, AllGather, and Broadcast.

## Project Structure
All work lives in ~/vp-benchmarks/ which is a git repo.

## Environment Detection
Run these first and adapt all subsequent commands to what you find:
    nvidia-smi
    nvcc --version
    which mpirun
    ls /usr/lib/x86_64-linux-gnu/libnccl* 2>/dev/null || ls /usr/local/cuda/lib64/libnccl* 2>/dev/null

## Steps

### 1. Install dependencies (if missing)
    sudo apt-get install -y build-essential git openmpi-bin libopenmpi-dev
    sudo apt-get install -y libnccl2 libnccl-dev

### 2. Clone and build nccl-tests
    git clone https://github.com/NVIDIA/nccl-tests.git scripts/nccl/nccl-tests
    cd scripts/nccl/nccl-tests
    make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu
If build fails, locate correct paths with find /usr -name "libnccl*" and find /usr -name "mpi.h", then retry.

### 3. Run benchmarks
    cd ~/vp-benchmarks
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    TS=$(date +%Y%m%d_%H%M%S)
    mpirun -np $N_GPUS scripts/nccl/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 2>&1 | tee results/nccl/nccl_allreduce_${TS}.txt
    mpirun -np $N_GPUS scripts/nccl/nccl-tests/build/all_gather_perf -b 8 -e 8G -f 2 -g 1 2>&1 | tee results/nccl/nccl_allgather_${TS}.txt
    mpirun -np $N_GPUS scripts/nccl/nccl-tests/build/broadcast_perf -b 8 -e 8G -f 2 -g 1 2>&1 | tee results/nccl/nccl_broadcast_${TS}.txt

### 4. Write summary
Parse peak bandwidth and minimum latency from each output file and write to results/nccl/nccl_summary.md with this format:

# NCCL Benchmark Results
Date: <timestamp>
Node: <hostname>
GPUs: <count> x <model>
CUDA: <version>
NCCL: <version>

| Test      | Message Size | Algo BW (GB/s) | Bus BW (GB/s) | Latency (µs) |
|-----------|-------------|----------------|---------------|-------------|
| AllReduce | ...         | ...            | ...           | ...         |
| AllGather | ...         | ...            | ...           | ...         |
| Broadcast | ...         | ...            | ...           | ...         |

### 5. Commit results
    git add results/nccl/nccl_summary.md
    git commit -m "results: NCCL benchmarks $(date +%Y%m%d_%H%M%S)"

## Error Handling
- If mpirun fails with permission denied: add --allow-run-as-root
- If NCCL throws ncclSystemError: run nvidia-smi -L to verify all GPUs visible
- If build fails with missing headers: adjust MPI_HOME and NCCL_HOME paths
- Fix errors before moving to the next test

## Done
When nccl_summary.md is written and committed, stop and report the results.
Do not start any other benchmarks.
