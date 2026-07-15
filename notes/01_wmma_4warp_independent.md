# Stage 1A: 4-Warp Independent WMMA

## Scope

`01_wmma_4warp_independent` is a controlled Stage 1A experiment:

- one CTA contains 4 warps;
- each warp computes one independent 16x16 C tile;
- A/B fragments are still loaded directly from global memory;
- there is no shared-memory staging and no CTA-level A/B reuse yet.

This isolates the effect of changing CTA granularity from one warp/block to four warps/block.

## Current Observation

The 1024 benchmark is almost unchanged from Stage 0. Nsight Compute for the 4096 case shows why the experiment is still useful:

| Kernel | Shape | Active warps/scheduler | Eligible warps/scheduler | Issue rate | L1/TEX throughput | Compute throughput |
|---|---:|---:|---:|---:|---:|---:|
| 00_wmma_1warp | 4096x4096x4096 | 7.91 | 0.11 | 0.11 issued warp/scheduler | 99.79% | 17.26% |
| 01_wmma_4warp_independent | 4096x4096x4096 | 14.96 | 0.13 | 0.11 issued warp/scheduler | 99.86% | 17.19% |

The active warp count rises close to the architectural maximum of 16 warps/scheduler, so the one-warp-per-block block-slot cap has been removed for this larger shape. However, eligible warps and issue rate do not improve.

Nsight Compute reports that each warp spends about 125.4 cycles waiting on L1TEX scoreboard dependencies, about 96.1% of the issue interval. This confirms that simply packing more independent warps into each CTA is not enough. The dominant bottleneck remains operand supply from direct global-memory WMMA fragment loads.

## Conclusion

Stage 1A is a negative but useful result:

- CTA granularity alone does not improve throughput materially;
- the kernel is still limited by L1TEX scoreboard stalls;
- Stage 1B should introduce shared-memory staging and CTA-level A/B reuse.
