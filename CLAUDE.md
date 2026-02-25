# Voltage Park Benchmark — STREAM / NVBench Memory Bandwidth

## Goal
Measure memory bandwidth across three channels:
1. CPU host memory bandwidth (STREAM)
2. Host-to-Device and Device-to-Host GPU transfer bandwidth (cuda-samples bandwidthTest)
3. Device-to-Device GPU memory bandwidth (cuda-samples bandwidthTest)

## Cluster Context
- Node: g138 (10.15.21.81)
- GPUs: 8x NVIDIA H100 80GB HBM3
- CPUs: 104x Intel Xeon Platinum 8470
- CUDA: 12.6

## Project Structure
All results go to ~/vp-benchmarks/results/stream_nvbench/
~/vp-benchmarks/
├── CLAUDE.md
├── results/
│   └── stream_nvbench/
│       ├── stream_<timestamp>.txt
│       ├── bandwidth_h2d_d2h_<timestamp>.txt
│       ├── bandwidth_d2d_<timestamp>.txt
│       └── stream_nvbench_summary.md
└── scripts/
    └── stream_nvbench/

## Setup
```bash
mkdir -p ~/vp-benchmarks/results/stream_nvbench
mkdir -p ~/vp-benchmarks/scripts/stream_nvbench
```

## Step 1: STREAM (CPU Host Memory Bandwidth)

### Build
```bash
cd ~/vp-benchmarks/scripts/stream_nvbench
git clone https://github.com/jeffhammond/STREAM.git
cd STREAM
gcc -O3 -march=native -fopenmp \
    -DSTREAM_ARRAY_SIZE=80000000 \
    -DNTIMES=20 \
    stream.c -o stream
```

If gcc is missing: `sudo apt-get install -y build-essential`

### Run
```bash
TS=$(date +%Y%m%d_%H%M%S)
OMP_NUM_THREADS=$(nproc) ./stream 2>&1 | \
    tee ~/vp-benchmarks/results/stream_nvbench/stream_${TS}.txt
```

### What to record
From the output, extract all 4 operation bandwidths:
- Copy (GB/s)
- Scale (GB/s)
- Add (GB/s)
- Triad (GB/s) ← primary metric per proposal

---

## Step 2: NVBench — Host-Device Bandwidth (cuda-samples bandwidthTest)

### Build
```bash
cd ~/vp-benchmarks/scripts/stream_nvbench
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/bandwidthTest
make
```

If make fails, check CUDA path:
```bash
which nvcc
ls /usr/local/cuda/bin/nvcc
```
Then retry with: `make CUDA_PATH=/usr/local/cuda`

### Run H2D and D2H
```bash
TS=$(date +%Y%m%d_%H%M%S)
./bandwidthTest --mode=shmoo --memory=pinned --htod --dtoh 2>&1 | \
    tee ~/vp-benchmarks/results/stream_nvbench/bandwidth_h2d_d2h_${TS}.txt
```

### What to record
- Peak H2D bandwidth (GB/s)
- Peak D2H bandwidth (GB/s)

---

## Step 3: NVBench — Device-Device Bandwidth

### Run D2D
```bash
TS=$(date +%Y%m%d_%H%M%S)
./bandwidthTest --mode=shmoo --memory=pinned --dtod 2>&1 | \
    tee ~/vp-benchmarks/results/stream_nvbench/bandwidth_d2d_${TS}.txt
```

### What to record
- Peak D2D bandwidth (GB/s)

---

## Step 4: Write Summary
Parse all result files and write to
~/vp-benchmarks/results/stream_nvbench/stream_nvbench_summary.md:

# STREAM / NVBench Memory Bandwidth Results
Date: <timestamp>
Node: g138
GPUs: 8x NVIDIA H100 80GB HBM3
CPUs: 104x Intel Xeon Platinum 8470 ($(nproc) threads)
CUDA: 12.6

## STREAM — CPU Host Memory Bandwidth
| Operation | Bandwidth (GB/s) |
|-----------|-----------------|
| Copy      | ...             |
| Scale     | ...             |
| Add       | ...             |
| Triad     | ...             |

**Theoretical peak:** Intel Xeon Platinum 8470 = 307.2 GB/s per socket,
2 sockets = ~614 GB/s total DDR5 bandwidth

## NVBench — GPU Transfer Bandwidth
| Transfer Direction | Peak Bandwidth (GB/s) |
|-------------------|----------------------|
| Host to Device (H2D) | ...              |
| Device to Host (D2H) | ...              |
| Device to Device (D2D) | ...            |

**H100 HBM3 theoretical peak:** 3.35 TB/s device memory bandwidth

## Analysis
- STREAM Triad efficiency: <measured> / 614 GB/s = <pct>%
- H100 D2D efficiency: <measured> / 3350 GB/s = <pct>%
- Note any bottlenecks observed

## Raw Log Files
- stream_<timestamp>.txt
- bandwidth_h2d_d2h_<timestamp>.txt
- bandwidth_d2d_<timestamp>.txt

---

## Step 5: Commit Results
```bash
cd ~/vp-benchmarks
git add results/stream_nvbench/
git commit -m "results: STREAM/NVBench memory bandwidth $(date +%Y%m%d_%H%M%S)"
git push origin master
```

## Error Handling
- If STREAM array size causes OOM: reduce STREAM_ARRAY_SIZE to 40000000
- If bandwidthTest make fails: check CUDA_PATH and nvcc location
- If D2D bandwidth seems low: verify GPUs are on same NVLink domain with nvidia-smi topo -m
- Fix errors before moving to next test

## Done
When stream_nvbench_summary.md is written and committed, stop and report results.
Do not run any other benchmarks.
