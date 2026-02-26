# Voltage Park Benchmark — HPL-AI / HPCG (GPU-Optimized)

## Goal
Measure raw floating-point compute efficiency and tensor core utilization across
all 8 H100 GPUs using NVIDIA's official GPU-optimized HPC benchmarks container.

- **HPL-AI**: Mixed-precision tensor core throughput (FP16/FP8)
- **HPCG**: GPU-accelerated sparse linear algebra

Primary metrics: TFLOPS, GFLOP/s, efficiency % vs theoretical peak

## Node Context
- Node: g138 (10.15.21.81)
- GPUs: 8x NVIDIA H100 80GB HBM3 SXM5
- CUDA: 12.6
- H100 FP16 theoretical peak: 1979 TFLOPS
- Docker: available

## Project Structure
~/vp-benchmarks/results/hpl_ai/
├── hpcg_gpu_<timestamp>.txt
├── hpl_ai_<timestamp>.txt
└── hpl_ai_summary.md

## Setup
```bash
mkdir -p ~/vp-benchmarks/results/hpl_ai
cd ~/vp-benchmarks

# Pull the container first
docker pull nvcr.io/nvidia/hpc-benchmarks:24.3
```

If pull fails try: `docker pull nvcr.io/nvidia/hpc-benchmarks:latest`

## Step 1: GPU-Optimized HPCG
```bash
TS=$(date +%Y%m%d_%H%M%S)

docker run --rm --gpus all \
  --cap-add=SYS_NICE \
  -v ~/vp-benchmarks/results/hpl_ai:/results \
  nvcr.io/nvidia/hpc-benchmarks:24.3 \
  bash -c "hpcg.sh --nx 256 --ny 256 --nz 256 --rt 60 \
  2>&1 | tee /results/hpcg_gpu_${TS}.txt"
```

If hpcg.sh flags differ, try:
```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/hpc-benchmarks:24.3 \
  hpcg.sh --help
```
And adjust flags accordingly.

### What to record
- GFLOP/s rating (expect 1000-4000 GFLOP/s vs 2 GFLOP/s CPU result)
- Convergence status

## Step 2: HPL-AI (Tensor Core Benchmark)
```bash
TS=$(date +%Y%m%d_%H%M%S)

docker run --rm --gpus all \
  --cap-add=SYS_NICE \
  -v ~/vp-benchmarks/results/hpl_ai:/results \
  nvcr.io/nvidia/hpc-benchmarks:24.3 \
  bash -c "mpirun -np 8 --allow-run-as-root \
  hpl.sh --xhpl-ai --dat /workspace/HPL.dat \
  2>&1 | tee /results/hpl_ai_${TS}.txt"
```

If HPL.dat is needed, create one with these settings for 8x H100 80GB:
```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
200000       Ns
1            # of NBs
1024         NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
2            Ps
4            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
4            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
1            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
1            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

### What to record
- Peak TFLOPS
- Efficiency % vs 1979 TFLOPS theoretical

## Step 3: Write Summary
Write to ~/vp-benchmarks/results/hpl_ai/hpl_ai_summary.md:

# HPL-AI / HPCG Results
Date: <timestamp>
Node: g138
GPUs: 8x NVIDIA H100 80GB HBM3 SXM5
CUDA: 12.6
Container: nvcr.io/nvidia/hpc-benchmarks:24.3
H100 FP16 Tensor Core theoretical peak: 1979 TFLOPS

## HPCG — GPU-Accelerated Sparse Compute
| Metric | Value |
|--------|-------|
| GFLOP/s (GPU) | ... |
| GFLOP/s (CPU baseline) | 2.05 |
| GPU Speedup | ...x |
| Convergence | PASSED / FAILED |
| Grid size | 256x256x256 |

## HPL-AI — Tensor Core Utilization
| Metric | Value |
|--------|-------|
| Peak TFLOPS | ... |
| Theoretical peak | 1979 TFLOPS (FP16) |
| Efficiency % | ...% |
| Problem size (N) | ... |
| Block size (NB) | ... |

## Analysis
- GPU HPCG vs CPU HPCG demonstrates GPU acceleration ratio
- HPL-AI efficiency reflects tensor core utilization health
- Note any deviations from expected performance

## Raw Log Files
- hpcg_gpu_<timestamp>.txt
- hpl_ai_<timestamp>.txt

## Step 4: Commit
```bash
cd ~/vp-benchmarks
git add results/hpl_ai/
git commit -m "results: HPL-AI/HPCG GPU benchmark $(date +%Y%m%d_%H%M%S)"
git push origin master
```

## Error Handling
- If docker pull fails: try nvcr.io/nvidia/hpc-benchmarks:latest
- If hpcg.sh flags are wrong: run with --help to find correct syntax
- If HPL-AI needs a different HPL.dat: check container docs with docker run --rm nvcr.io/nvidia/hpc-benchmarks:24.3 ls /workspace/
- If efficiency <50%: check all 8 GPUs visible with nvidia-smi and NVLink active
- Keep the CPU HPCG result (2.05 GFLOP/s) in summary as baseline comparison

## Done
When hpl_ai_summary.md is written and committed, stop and report results.
Do not run any other benchmarks.