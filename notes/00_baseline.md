# Stage 0 Baseline: 00_wmma_1warp

## Scope

This note records the first reproducible baseline for the single-warp WMMA GEMM kernel:

- kernel: `kernels/00_wmma_1warp.cu`
- test app: `00_wmma_1warp_test`
- benchmark app: `00_wmma_1warp_bench`
- input layout: A row-major FP16, B row-major FP16
- accumulation/output: FP32
- baseline: cuBLAS `cublasGemmEx`

The kernel computes one 16x16 output tile per warp. Each warp loads A/B fragments directly from global memory for every K tile, performs WMMA MMA, and stores the FP32 output tile to global memory. There is no CTA-level tiling, shared-memory staging, or inter-warp data reuse in this version.

## Environment

Results are saved under `results/H100_CUDA12.4/`.

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| Driver | 550.54.15 |
| CUDA runtime reported by driver | 12.4 |
| nvcc | CUDA 12.4, V12.4.131 |
| CMake | 3.22.1 |
| Compile target | sm_90 |

## Correctness

Command:

```bash
./bin/00_wmma_1warp_test 256 256 256
```

All test patterns passed:

| Pattern | max_abs_error | max_rel_error | normalized_error |
|---|---:|---:|---:|
| zeros | 0 | 0 | 0 |
| ones | 0 | 0 | 0 |
| identity_like | 0 | 0 | 0 |
| deterministic_random | 0.00000477 | 0.01771144 | 4.37161173e-07 |
| signed_pattern | 0 | 0 | 0 |

The deterministic-random relative error is large only where the reference value is small. The normalized error is low enough for this FP16-input/FP32-accumulate baseline.

## Benchmark

Commands:

```bash
./bin/00_wmma_1warp_bench 1024 1024 1024
./bin/00_wmma_1warp_bench 4096 4096 4096
```

| Shape | Kernel avg ms | Kernel avg TFLOPS | cuBLAS avg ms | cuBLAS avg TFLOPS | Kernel vs cuBLAS error |
|---|---:|---:|---:|---:|---:|
| 1024x1024x1024 | 0.0839 | 25.586 | 0.0156 | 137.223 | 0 |
| 4096x4096x4096 | 4.8330 | 28.438 | 0.2031 | 676.836 | 0 |

This establishes the expected low baseline: the single-warp WMMA kernel is correct but far below cuBLAS throughput.

## PTX, SASS, and Resource Evidence

Saved files:

- `results/H100_CUDA12.4/ptx/00_wmma_1warp_sm90.ptx`
- `results/H100_CUDA12.4/sass/00_wmma_1warp_sm90.sass`
- `results/H100_CUDA12.4/ptxas/00_wmma_1warp_sm90.txt`

The generated PTX contains WMMA load/MMA/store instructions, and the SASS contains HMMA instructions. The `ptxas` report shows:

```text
0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 32 registers
```

So the baseline has no spills and modest register usage. The performance gap is not explained by local-memory spilling.

## Nsight Compute Summary

Saved reports:

- `results/H100_CUDA12.4/ncu/00_wmma_1warp_1024_core.txt`
- `results/H100_CUDA12.4/ncu/00_wmma_1warp_4096_core.txt`

Key metrics:

| Shape | Duration | Compute throughput | DRAM throughput | L1/TEX throughput | L2 throughput | Issue slots busy | Active warps/scheduler | Eligible warps/scheduler |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | 101.41 us | 16.41% | 1.24% | 98.71% | 32.34% | 10.94% | 7.69 | 0.13 |
| 4096x4096x4096 | 6.15 ms | 17.26% | 6.27% | 99.79% | 47.63% | 10.81% | 7.91 | 0.11 |

Interpretation:

- The kernel is not DRAM-bandwidth bound. DRAM throughput is low.
- L1/TEX throughput is near saturated, and Nsight Compute reports long waits around L1TEX operations.
- There are many active warps, but almost no eligible warps per scheduler. Issue slots are busy only about 11%.
- Tensor Core compute is therefore poorly fed. The bottleneck is the direct global-memory fragment loading pattern and lack of data reuse, not the raw HMMA instruction itself.

## Conclusion

Stage 0 is complete as a baseline:

- correctness harness is in place;
- benchmark harness is in place;
- cuBLAS baseline is in place;
- PTX/SASS/resource artifacts are saved;
- Nsight Compute reports identify the first bottleneck.

The next stage should improve data reuse with a multi-warp CTA tile and shared-memory staging. The target is not only higher TFLOPS, but also a visible shift in profiler evidence: higher issue efficiency, more eligible warps per scheduler, lower L1/TEX pressure per unit of work, and better Tensor Core utilization.
