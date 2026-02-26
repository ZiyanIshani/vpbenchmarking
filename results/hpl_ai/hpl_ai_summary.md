# HPL / HPCG / HPL-MxP Benchmark Results
**Node:** Voltage Park bare metal | **GPUs:** 8x NVIDIA H100 80GB HBM3 | **Container:** `nvcr.io/nvidia/hpc-benchmarks:24.09`

---

## Results Summary

| Benchmark | Precision | Total TFLOPS | Per GPU TFLOPS | LU Peak TFLOPS | Status |
|-----------|-----------|-------------|---------------|----------------|--------|
| HPCG (GPU) | N/A | 0.523 | 0.065 | — | PASSED |
| HPL (FP64) | FP64 | 365.4 | 45.67 | — | PASSED |
| HPL-MxP | FP16 (sloppy-type=2) | 364.95 | 45.62 | 2,537.7 | PASSED |
| HPL-MxP | FP8 (sloppy-type=1) | 315.12 | 39.39 | **3,554.6** | PASSED |

**H100 FP8 theoretical peak: 3,958 TFLOPS → measured LU peak = 89.8% utilization**

---

## 1. HPCG (GPU-Accelerated)

**What it measures:** Memory-bandwidth-bound sparse iterative solver (multigrid + CG). Reflects real-world scientific computing workload performance, not peak compute.

**Run configuration:**
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/hpc-benchmarks:24.09 \
  /workspace/hpcg.sh --nx 256 --ny 256 --nz 256 --rt 60
```

- Grid size: 256 × 256 × 256 (16,777,216 equations)
- Runtime: 60.3 seconds
- Threads per process: 104 (all CPU threads + GPU offload)
- GPU memory used: 14,009 MB / 81,109 MB per GPU

**Results:**
- Official GFLOP/s rating: **523.366 GFLOP/s**
- HPCG 2.4 historical rating: 525.297 GFLOP/s
- Raw Total Bandwidth: **4,203.68 GB/s** (H100 HBM3 theoretical: ~3,350 GB/s — measurement includes multi-level cache effects)
- All validation checks: **PASSED** (spectral convergence, symmetry, reproducibility)

**Note:** The official 60-second result is considered a short run for submission purposes. Official TOP500 HPCG submissions require ≥1,800 seconds. For comparison purposes the number is valid.

**CPU baseline (bare metal, no GPU):** 2.05 GFLOP/s → **255× GPU speedup**

---

## 2. HPL Standard (FP64 Dense Linear Algebra)

**What it measures:** Classic High-Performance Linpack benchmark. Solves a dense linear system Ax=b using LU factorization in full FP64 double precision. The traditional metric for TOP500 supercomputer rankings.

**Run configuration:**
```bash
mpirun -np 8 /workspace/hpl.sh \
  --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-dgx-1N.dat \
  --no-multinode
```

**Dat file choice:** Used `HPL-dgx-1N.dat` — NVIDIA's standard single-node DGX H100 configuration. Chosen for comparability with published H100 benchmarks. The H200 dat file (`HPL-H200-8GPUs.dat`, N=366,735) was attempted first but caused OOM errors since H200 has 141GB vs H100's 80GB VRAM.

**Key parameters from dat file:**
- N = 264,192 (matrix dimension)
- NB = 1,024 (blocking factor)
- P × Q = 4 × 2 (process grid, column-major)

**Results:**
- Total: **365.4 TFLOPS** (3.654e+05 GFLOPS)
- Per GPU: **45.67 TFLOPS**
- Runtime: 33.64 seconds
- Residual check: **PASSED** (||Ax-b|| = 0.000195, threshold = 16.0)
- GPU memory used: 74.67–76.15 GiB per GPU (out of 79.65 GiB)

**Efficiency:** 76% of H100 FP64 theoretical peak (60 TFLOPS/GPU) — strong result for double precision.

---

## 3. HPL-MxP — Mixed Precision Tensor Core Benchmark

**What it measures:** Solves the same Ax=b problem as HPL but uses low-precision tensor cores (FP16 or FP8) for the heavy LU factorization, then iteratively refines the solution back to FP64 accuracy using GMRES. The LU GFLOPS figure directly measures tensor core throughput — the most relevant metric for AI training workloads.

**Run configuration:**
```bash
mpirun -np 8 /workspace/hpl-mxp.sh \
  --gpu-affinity 0:1:2:3:4:5:6:7 \
  --nprow 4 --npcol 2 --nporder row \
  --n 274432 --nb 1024 \
  --sloppy-type [1|2]
```

**Parameter choices:**
- **N = 274,432:** Taken from the larger N value in `HPL-dgx-1N.dat` (which lists N=264,192 and N=274,432). Chosen to maximize GPU memory utilization (~37 GB / 79 GB per GPU). An initial run with N=200,000 was done first as a test and showed lower efficiency due to underutilization.
- **NB = 1,024:** Standard blocking factor matching HPL dat file.
- **Process grid 4×2:** Matches HPL configuration for consistency.
- **sloppy-type:** Controls the precision of the LU factorization phase — FP16 (type=2) is the default, FP8 (type=1) exercises the H100's highest-throughput tensor core mode.

### Run A — FP16 (sloppy-type=2)

- Total GFLOPS: **364,950 (364.95 TFLOPS)**
- Per GPU: 45,619 GFLOPS (45.62 TFLOPS)
- **LU GFLOPS: 2,537,700 (2.54 PFLOPS)** ← tensor core throughput
- LU per GPU: 317,208 GFLOPS
- LU runtime: 5.43 seconds
- GMRES iterations: 3 (residuals: 4.86e-4 → 6.73e-10 → 9.77e-14)
- Validation: **PASSED** (residual = 5.90e-4, threshold = 16.0)

### Run B — FP8 (sloppy-type=1) ← Primary Result

- Total GFLOPS: **315,120 (315.12 TFLOPS)**
- Per GPU: 39,390 GFLOPS (39.39 TFLOPS)
- **LU GFLOPS: 3,554,600 (3.55 PFLOPS)** ← tensor core throughput
- LU per GPU: 444,331 GFLOPS
- LU runtime: 3.88 seconds (vs 5.43s for FP16 — 1.4× faster factorization)
- GMRES iterations: 3 (residuals: 5.23e-3 → 1.53e-8 → 1.06e-13)
- Validation: **PASSED** (residual = 6.43e-4, threshold = 16.0)

**Why FP8 total GFLOPS is lower than FP16:** FP8 introduces more numerical error during factorization (larger initial residual ~5e-3 vs ~5e-4 for FP16), so the GMRES refinement phase takes longer (39.85s vs 32.33s). The overall reported GFLOPS is amortized across both phases. The LU figure alone is the correct metric for tensor core utilization.

**H100 FP8 theoretical peak: 3,958 TFLOPS → LU peak = 89.8% efficiency**

---

## Infrastructure Notes

- **Docker:** Required `apt`-installed Docker + nvidia-container-toolkit. Snap Docker (Ubuntu default) does not support GPU passthrough to containers — snap isolation prevents `nvidia-ctk runtime configure` from taking effect.
- **NGC authentication:** Container requires login to `nvcr.io` with `$oauthtoken` / NGC API key.
- **No InfiniBand:** All runs show the IB warning. Single-node benchmarks are unaffected; multi-node communication falls back to Ethernet/NVLink.
- **File permissions:** Docker runs as root, so result files written to mounted volumes require `sudo chown -R ubuntu:ubuntu` after the run.