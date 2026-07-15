#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

struct GemmShape {
    size_t M;
    size_t N;
    size_t K;

    size_t lda;
    size_t ldb;
    size_t ldc;
};

cudaError_t launch_wmma_1warp(const half *A, const half *B, float *C,
                              const GemmShape &shape, cudaStream_t stream = nullptr);

#endif // KERNELS_CUH
