# Stage 1B: WMMA Block-Tiled SMEM Reuse

## Scope

`01_wmma_block_tiled` is a controlled shared-memory reuse experiment after Stage 1A:

- one CTA contains 4 warps with `blockDim = (32, 4)`;
- warps are arranged as a 2x2 grid;
- each warp still computes exactly one 16x16 C tile;
- one CTA computes a 32x32 C tile;
- each K step stages `A[32x16]` and `B[16x32]` through shared memory;
- each staged A half-tile is reused by two warp columns;
- each staged B half-tile is reused by two warp rows.

This stage tests whether explicit CTA-level shared-memory reuse improves operand supply compared with Stage 1A direct global WMMA fragment loads.

## Expected Traffic Change

For the same 32x32 output CTA tile and one `K=16` step:

| Kernel | A input elements | B input elements | Reuse |
|---|---:|---:|---|
| `01_wmma_4warp_independent` | `4 * 16 * 16` | `4 * 16 * 16` | none |
| `01_wmma_block_tiled` | `32 * 16` | `16 * 32` | 2-way A and B reuse |

Stage 1B should reduce global input traffic by about 2x for A and B at this CTA granularity. It adds cooperative GMEM-to-SMEM stores, WMMA shared-memory loads, and two barriers per `K=16` step.

## Acceptance Checklist

Stage 1B is considered valid when:

- the kernel uses 4 warps/block with 2x2 warp arrangement;
- every warp computes one 16x16 C tile;
- the staged shared tiles are `A[32x16]` and `B[16x32]`;
- cooperative GMEM-to-SMEM loading is correct;
- A and B are explicitly reused by two warps each;
- all correctness tests pass;
- PTX/SASS show shared-memory load/store paths;
- `ptxas` reports no spills;
- measured global input traffic is clearly lower than Stage 1A;
- benchmark results are compared against Stage 0, Stage 1A, and cuBLAS on the same device;
- Nsight Compute either proves or disproves the hypothesis that shared-memory reuse improves operand supply.

A performance improvement is not required. The experiment is still useful if Nsight Compute shows that lower global traffic and lower long-scoreboard stalls are offset by barrier or shared-memory overhead.

## Suggested Commands

Correctness and benchmark:

```bash
./build/bin/01_wmma_block_tiled_test 256 256 256
./build/bin/01_wmma_block_tiled_bench 1024 1024 1024
./build/bin/01_wmma_block_tiled_bench 4096 4096 4096
```

PTX/SASS evidence:

```bash
mkdir -p results/H100_CUDA12.4/ptx results/H100_CUDA12.4/sass results/H100_CUDA12.4/ptxas

nvcc -O3 -std=c++17 -lineinfo -arch=sm_90 -Iinclude \
  -ptx kernels/01_wmma_block_tiled.cu \
  -o results/H100_CUDA12.4/ptx/01_wmma_block_tiled_sm90.ptx

nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -arch=sm_90 -Iinclude \
  -c kernels/01_wmma_block_tiled.cu \
  -o /tmp/01_wmma_block_tiled_sm90.o \
  2> results/H100_CUDA12.4/ptxas/01_wmma_block_tiled_sm90.txt

cuobjdump --dump-sass build/bin/01_wmma_block_tiled_bench \
  > results/H100_CUDA12.4/sass/01_wmma_block_tiled_sm90.sass
```

Nsight Compute text and `.ncu-rep` export:

```bash
mkdir -p results/H100_CUDA12.4/ncu

ncu \
  --section SpeedOfLight \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --target-processes all \
  --kernel-name regex:tensorcore_gemm \
  --export results/H100_CUDA12.4/ncu/01_wmma_block_tiled_4096_core \
  --force-overwrite \
  --page details \
  ./build/bin/01_wmma_block_tiled_bench 4096 4096 4096 \
  > results/H100_CUDA12.4/ncu/01_wmma_block_tiled_4096_core.txt
```

## Metrics To Compare With Stage 1A

Primary evidence:

- DRAM/L2/global load traffic for A and B;
- `L1/TEX Cache Throughput`;
- long scoreboard stalls;
- short scoreboard or MIO stalls;
- barrier stalls;
- eligible warps per scheduler;
- issue rate;
- compute throughput;
- registers/thread, static shared memory, and spills from `ptxas`.

Expected interpretation:

- If global traffic and long scoreboard go down, shared-memory reuse worked as intended.
- If throughput does not improve, check whether short scoreboard, MIO, shared-memory bank conflicts, barriers, or reduced occupancy offset the benefit.

## Current Observation: H100 / CUDA 12.4

The first Nsight Compute run for the `4096x4096x4096` case shows that the smaller `A[32x16] + B[16x32]` shared-memory staging version changes the bottleneck profile substantially compared with Stage 1A.

| Kernel | Grid | CTA tile | SMEM/block | L1/TEX throughput | DRAM throughput | Compute throughput | Active warps/scheduler | Eligible warps/scheduler | Issue rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `01_wmma_4warp_independent` | `256x256` | four independent `16x16` tiles | 0 B | 99.86% | low | 17.19% | 14.96 | 0.13 | 0.11 |
| `01_wmma_block_tiled` | `128x128` | one `32x32` CTA tile | 2 KB | ~70.6% | ~5.7% | ~76.4% | 9.76 | 2.30 | 0.77 |

Interpretation:

- The grid changed from `256x256` independent 16x16 tiles to `128x128` 32x32 CTA tiles, confirming the intended CTA mapping.
- Global input traffic should be about 2x lower for A and B because each CTA loads `A[32x16]` and `B[16x32]` once instead of four independent `16x16` A/B pairs.
- Eligible warps per scheduler improved from `0.13` to `2.30`, and issue rate improved from `0.11` to `0.77`, so operand supply is much healthier.
- L1/TEX throughput dropped from the saturated Stage 1A level to about `70%`, indicating that direct global operand loads are no longer the same dominant pressure point.
- The kernel now pays for shared-memory load/store and two barriers per K tile. If benchmark speedup is limited, that overhead is the primary tradeoff to explain.

This result supports the Stage 1B hypothesis: explicit shared-memory reuse improves operand supply. Correctness, benchmark, NCU, `ptxas`, PTX, and SASS evidence are saved under `results/H100_CUDA12.4/`.

## Benchmark Result: H100 / CUDA 12.4

Correctness passed for the `256x256x256` test suite, including padded leading-dimension mapping cases.

Same-device benchmark comparison:

| Kernel | Shape | Median ms | Median TFLOPS | cuBLAS median TFLOPS |
|---|---:|---:|---:|---:|
| `00_wmma_1warp` | 1024^3 | 0.0840 | 25.575 | 136.678 |
| `01_wmma_4warp_independent` | 1024^3 | 0.0832 | 25.801 | 137.801 |
| `01_wmma_block_tiled` | 1024^3 | 0.1204 | 17.843 | 121.135 |
| `00_wmma_1warp` | 4096^3 | 4.8165 | 28.535 | 677.013 |
| `01_wmma_4warp_independent` | 4096^3 | 4.8647 | 28.252 | 676.906 |
| `01_wmma_block_tiled` | 4096^3 | 5.9065 | 23.269 | 637.330 |

Interpretation:

- Stage 1B is slower in wall-clock benchmark than Stage 0 and Stage 1A.
- Nsight Compute still shows much better scheduler readiness and compute-pipeline utilization than Stage 1A.
- The added GMEM-to-SMEM stores, WMMA shared-memory loads, and two barriers per K tile outweigh the 2-way operand reuse in this simple WMMA implementation.
- This is a valid Stage 1B negative performance result: the shared-memory reuse hypothesis improves operand supply, but the implementation has enough synchronization and shared-memory overhead that total runtime regresses.

## PTX/SASS Evidence

`ptxas` resource report for `sm_90`:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 48 registers, 2048 bytes smem
```

The 2048-byte static shared-memory allocation matches `A[32x16] + B[16x32]` with FP16 elements:

```text
32 * 16 * 2 + 16 * 32 * 2 = 2048 bytes
```

PTX evidence shows explicit shared-memory storage and WMMA shared-memory fragment loads:

```text
.shared .align 32 .b8 ... a_shm[1024]
.shared .align 32 .b8 ... b_shm[1024]
st.shared.u16
wmma.load.a.sync.aligned.row.m16n16k16.shared.f16
wmma.load.b.sync.aligned.row.m16n16k16.shared.f16
```

SASS evidence shows shared-memory stores and matrix shared-memory loads:

```text
STS.U16
LDSM.16.M88.4
LDSM.16.MT88.4
```

## Stage 1B Conclusion

Stage 1B passes the intended acceptance criteria:

- 4 warps/block with a 2x2 warp arrangement;
- each warp computes exactly one 16x16 output tile;
- the CTA stages `A[32x16]` and `B[16x32]` through shared memory;
- A and B are explicitly reused by two warps each;
- correctness tests pass;
- `ptxas` reports no spills;
- PTX/SASS confirm the shared-memory path;
- benchmark and Nsight Compute are compared against Stage 0, Stage 1A, and cuBLAS on H100/CUDA 12.4.

The experiment validates the operand-supply hypothesis but not the performance-improvement hypothesis. Compared with Stage 1A, scheduler readiness and compute throughput improve sharply, but wall-clock time regresses because cooperative stores, shared-memory WMMA loads, and barriers add more cost than this simple 2-way reuse removes.
