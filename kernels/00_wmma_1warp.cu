#ifndef __CUDACC__
#error "00_wmma_1warp.cu must be compiled by nvcc or another CUDA compiler, not a host C++ compiler."
#endif

#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <climits>
#include <cstdint>

namespace wmma = nvcuda::wmma;

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr std::uintptr_t kWmmaAlignmentBytes = 32;

bool is_aligned_32(const void *ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % kWmmaAlignmentBytes == 0;
}

// One warp computes one 16x16 C tile. Stage 0 intentionally keeps this kernel
// minimal so later stages can isolate each tiling and memory-system change.
template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void tensorcore_gemm(const half *A, const half *B, float *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    const int row = blockIdx.y * WMMA_M;
    const int col = blockIdx.x * WMMA_N;

    if (row >= M || col >= N) {
        return;
    }

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + row * lda + k, lda);
        wmma::load_matrix_sync(b_frag, B + k * ldb + col, ldb);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + row * ldc + col, c_frag, ldc, wmma::mem_row_major);
}

} // namespace

cudaError_t launch_wmma_1warp(const half *A, const half *B, float *C, const GemmShape &shape,
                              cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!is_aligned_32(A) || !is_aligned_32(B) || !is_aligned_32(C)) {
        return cudaErrorInvalidValue;
    }
    if (shape.M == 0 || shape.N == 0 || shape.K == 0 || shape.lda < shape.K ||
        shape.ldb < shape.N || shape.ldc < shape.N) {
        return cudaErrorInvalidValue;
    }
    if (shape.lda % 8 != 0 || shape.ldb % 8 != 0 || shape.ldc % 4 != 0) {
        return cudaErrorInvalidValue;
    }
    if (shape.M % kWmmaM != 0 || shape.N % kWmmaN != 0 || shape.K % kWmmaK != 0) {
        return cudaErrorInvalidValue;
    }
    if (shape.M > INT_MAX || shape.N > INT_MAX || shape.K > INT_MAX || shape.lda > INT_MAX ||
        shape.ldb > INT_MAX || shape.ldc > INT_MAX) {
        return cudaErrorInvalidValue;
    }

    const dim3 block(32);
    const dim3 grid(static_cast<unsigned>(shape.N / kWmmaN),
                    static_cast<unsigned>(shape.M / kWmmaM));

    tensorcore_gemm<kWmmaM, kWmmaN, kWmmaK><<<grid, block, 0, stream>>>(
        A, B, C, static_cast<int>(shape.M), static_cast<int>(shape.N), static_cast<int>(shape.K),
        static_cast<int>(shape.lda), static_cast<int>(shape.ldb), static_cast<int>(shape.ldc));
    return cudaGetLastError();
}
