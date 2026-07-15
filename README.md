# tensorcore-learning

一个以 **NVIDIA Tensor Core GEMM** 为主线的渐进式学习与性能工程仓库。

项目从最小的 `nvcuda::wmma` 单 warp GEMM 出发，逐步进入显式 `mma.sync`、`ldmatrix`、shared-memory layout、`cp.async` 多阶段流水线、CUTLASS/CuTe，以及后期 Hopper 的 WGMMA、TMA 与 warp specialization。最终目标是把这些能力迁移到 fused GEMM 和 LLM kernel。

> 当前状态：**Stage 0 — 仓库初始化与基线基础设施**。代码、实验结果和笔记将按照学习阶段增量加入，不预先创建空占位文件。

## 项目目标

### 1. 建立可推导的 Tensor Core 心智模型

能够从 instruction tile、warp tile 和 CTA tile 推导：

- thread/lane 与矩阵元素的映射；
- `GMEM → SMEM → registers → Tensor Core → epilogue` 数据路径；
- MMA、load/store、barrier 和 pipeline 的指令数量；
- global/shared-memory 流量与算术强度；
- registers、shared memory、occupancy 和并行度之间的权衡；
- PTX、SASS、Nsight Compute 指标与源码设计之间的对应关系。

### 2. 独立实现并分析高性能 GEMM

主线演进路径：

```text
WMMA single-warp GEMM
→ WMMA multi-warp CTA tiling
→ mma.sync microkernel
→ ldmatrix + shared-memory layout
→ explicit mma.sync CTA GEMM
→ cp.async multi-stage pipeline
→ tuned and engineered GEMM
→ WGMMA + TMA + warp specialization
```

### 3. 形成可复现的工程仓库

每个关键版本都应保留：

- 可运行 kernel；
- deterministic correctness tests；
- 统一 benchmark；
- cuBLAS/cuBLASLt baseline；
- PTX/SASS 证据；
- Nsight Compute 分析；
- 设计假设、失败实验和结论笔记。

## 硬件路线

GPU 按需租用，不把 Hopper 作为前置条件。

| GPU | 编译目标 | 计划中的用途 | 不作为主线的内容 |
|---|---:|---|---|
| V100 | `sm_70` | Stage 0–1、WMMA、可选 Volta MMA 与跨代对照 | `ldmatrix`、`cp.async`、WGMMA/TMA |
| RTX 4090 | `sm_89` | Stage 0–7、Stage 9A、可移植 fused GEMM | Hopper WGMMA/TMA |
| A800 | `sm_80` | Stage 0–7 的 Ampere 主线、Stage 9A、数据中心性能基线 | Hopper WGMMA/TMA |
| H800/H100 | `sm_90` / `sm_90a` | Stage 0–7 portable kernel 使用 `sm_90`；Stage 8/9B Hopper WGMMA/TMA 使用 `sm_90a` | 非前置依赖，后期按需短租 |

不同 GPU 的结果使用独立目录和独立 cuBLAS/cuBLASLt baseline。原始 TFLOPS、occupancy 和 stall 数据只在同一设备、同一软件栈与相近运行状态下纵向比较。

## 学习阶段

| Stage | 主题 | 主要产物 |
|---:|---|---|
| 0 | 基线与实验基础设施 | correctness、benchmark、cuBLAS/cuBLASLt、PTX/SASS、Nsight Compute |
| 1 | WMMA 解剖与多 warp CTA bridge | CTA tiling、SMEM 复用、P1 基线 |
| 2 | `mma.sync` 指令级微内核 | lane/register fragment mapping |
| 3 | `ldmatrix` 与 shared-memory layout | plain、padding、XOR swizzle 对照 |
| 4 | 显式 `mma.sync` 多 warp CTA GEMM | 同步式 block-tiled GEMM |
| 5 | `cp.async` 多阶段流水线 | 2-stage、3-stage、register double buffering |
| 6 | 通用高性能 GEMM 工程化 | 参数化、epilogue、tail、dispatch、P2 |
| 7 | 性能建模、SASS 与 Nsight 分析 | 静态模型与证据驱动的调优流程 |
| 8 | Hopper WGMMA、TMA 与 warp specialization | Hopper 专项，当前 deferred |
| 9A | Ampere/Ada CUTLASS/CuTe 映射 | `Layout`、`TiledCopy`、`TiledMMA`、collective |
| 9B | Hopper CUTLASS/CuTe 映射 | WGMMA/TMA collective 与 schedule，deferred |
| 10 | 迁移到 LLM kernel | GEMM + bias + activation、Attention/MLP/MoE 扩展 |

## 计划中的仓库结构

`kernels/` 使用**一个阶段对应一个 `.cu` 文件**的扁平结构，便于直接比较版本差异。`include/` 同样保持扁平，只存放已经被多个实验复用的头文件。

```text
tensorcore-learning/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/
│   ├── cuda_check.cuh
│   ├── timer.cuh
│   ├── correctness.cuh
│   ├── benchmark_result.h
│   ├── environment.h
│   ├── kernels.cuh
│   ├── mma.cuh
│   ├── ldmatrix.cuh
│   ├── cp_async.cuh
│   ├── wgmma.cuh
│   ├── tma.cuh
│   ├── gemm_config.cuh
│   ├── dispatch.cuh
│   └── epilogue.cuh
├── kernels/
│   ├── 00_wmma_1warp.cu
│   ├── 01_wmma_cta_tiled.cu
│   ├── 02_mma_sync_micro.cu
│   ├── 03_mma_sync_single_warp.cu
│   ├── 04_mma_sync_cta.cu
│   ├── 05_smem_swizzle.cu
│   ├── 06_cp_async_2stage.cu
│   ├── 07_cp_async_multistage.cu
│   ├── 08_tuned_mma_gemm.cu
│   ├── 09_wgmma_micro.cu
│   ├── 10_wgmma_tma.cu
│   ├── 11_warp_specialized.cu
│   ├── 12_cute_gemm.cu
│   └── 13_fused_gemm.cu
├── apps/
│   ├── test_gemm.cu
│   └── bench_gemm.cu
├── configs/
├── tools/
├── notes/
├── results/<GPU>_<CUDA_VERSION>/
└── third_party/
```

上述是目标结构，不代表所有文件会在 Stage 0 一次性创建。

## 实验原则

1. **先正确，再流水化，再调参。** 同一次受控实验尽量只改变一个主要机制。
2. **先预测，再测量。** 运行前计算 tile、bytes、instruction count、registers 和 SMEM；运行后用工具验证。
3. **保留对照组。** 例如 plain/padded/swizzled、同步/2-stage/3-stage、direct/staged epilogue。
4. **以同设备 cuBLAS/cuBLASLt 为主要性能基线。** 自定义 kernel 与 baseline 必须具有相同 dtype、layout、output、alpha/beta 和计时范围。
5. **不静默处理不支持的配置。** 未实现的 shape、tail、layout 或架构应明确返回 `unsupported`。
6. **不同架构分别编译与调参。** 不把某台 GPU 的最佳 CTA tile、stage 数或 occupancy 结论直接迁移到另一台 GPU。

## Correctness 计划

至少覆盖：

- 全 0；
- 全 1；
- identity-like pattern；
- 固定 seed 的 deterministic random；
- 正负值和不同数量级混合；
- 用唯一编号构造的 mapping/permutation 测试。

统一记录：

```text
max_abs_error
max_rel_error
normalized_error
first_mismatch_index
reference_value
kernel_value
```

初期主线语义：

```text
A: FP16 row-major
B: FP16 row-major
accumulate: FP32
C: FP32 或明确指定的 FP16 output
```

## Benchmark 计划

- 使用 CUDA events 测量 kernel 时间；
- allocation、初始化、H2D/D2H 和 correctness copy-back 不进入 kernel latency；
- 预热后运行多次，至少记录 median、min 和离散程度；
- 记录 GPU、compute capability、driver、CUDA、cuBLAS/cuBLASLt、Nsight Compute、时钟、功耗和温度；
- 保存 registers/thread、SMEM/block、spill、PTX/SASS 与主要 profiler 指标。

TFLOPS 定义：

```text
TFLOPS = 2 × M × N × K / elapsed_seconds / 1e12
```

初始 shape：

```text
1024 × 1024 × 1024
4096 × 4096 × 4096
```

后续增加 aligned square、rectangular 和 tail shape 集合。

## 性能里程碑

- **P1**：同一设备上，多 warp CTA kernel 明显超过单 warp WMMA baseline。
- **P2**：A800 或 RTX 4090 上，`mma.sync + cp.async` 在选定 aligned shape 达到约 60%–75% 同设备 cuBLAS/cuBLASLt，并能解释主要差距。
- **P3**：获得 H800/H100 后，`WGMMA + TMA` 在至少三个选定 aligned shape 达到 80% 以上同设备 cuBLAS/cuBLASLt。Hopper 不可用时该里程碑保持 deferred，不阻塞其他阶段。

这些数字是学习项目的验收线，不是对所有 shape、语义和设备的性能承诺。

## 当前工作：Stage 0

- [x] 保存当前实现为 `kernels/00_wmma_1warp.cu`；
- [x] 建立统一 correctness harness；
- [x] 建立统一 benchmark 与 cuBLAS baseline；
- [x] 收集环境、`ptxas` 资源报告、PTX 和 SASS；
- [x] 对 `1024³` 与 `4096³` 建立首份结果；
- [x] 完成第一次 Nsight Compute 基线分析；
- [x] 编写 `notes/00_baseline.md`。

Stage 0 的 H100/CUDA 12.4 结果保存在 `results/H100_CUDA12.4/`，总结见 `notes/00_baseline.md`。

## 构建状态

Stage 0 当前提供单 warp WMMA kernel、correctness app，以及包含 cuBLAS baseline 的 benchmark app：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --config Release
.\build\bin\Release\00_wmma_1warp_test.exe 256 256 256
.\build\bin\Release\00_wmma_1warp_bench.exe 1024 1024 1024
```

Linux 上推荐显式指定 Release 和目标架构：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
./build/bin/00_wmma_1warp_test 256 256 256
./build/bin/00_wmma_1warp_bench 1024 1024 1024
```

也可以直接用 `nvcc` 做快速验证：

```bash
nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -arch=sm_80 -Iinclude -DTC_KERNEL_NAME=\"00_wmma_1warp\" -DTC_KERNEL_LAUNCH=launch_wmma_1warp kernels/00_wmma_1warp.cu apps/test_gemm.cu -o 00_wmma_1warp_test
nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v -arch=sm_80 -Iinclude -DTC_KERNEL_NAME=\"00_wmma_1warp\" -DTC_KERNEL_LAUNCH=launch_wmma_1warp kernels/00_wmma_1warp.cu apps/bench_gemm.cu -lcublas -o 00_wmma_1warp_bench
./00_wmma_1warp_test 256 256 256
./00_wmma_1warp_bench 1024 1024 1024
```

常用架构覆盖：V100 使用 `70`，A800 使用 `80`，RTX 4090 使用 `89`，H100/H800 使用 `90`。如果使用单配置生成器，运行路径通常是 `./build/bin/00_wmma_1warp_test` 和 `./build/bin/00_wmma_1warp_bench`。WMMA 需要 `sm_70` 或更新架构，不能用默认的旧架构目标编译。如果已有 build 目录缓存了旧值，例如 `52`，请删除 build 目录或重新配置时显式传入 `-DCMAKE_CUDA_ARCHITECTURES=90`。
