#ifndef __CUDACC__
#error "01_wmma_cta_tiled.cu must be compiled by nvcc or another CUDA compiler, not a host C++ compiler."
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
constexpr int kWarpsPerBlock = 4;
constexpr std::uintptr_t kWmmaAlignmentBytes = 32;

bool is_aligned_32(const void *ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % kWmmaAlignmentBytes == 0;
}

// Stage 1A: one CTA contains four independent warps. Each warp computes one
// 16x16 C tile and still loads A/B fragments directly from global memory.
template <int WMMA_M, int WMMA_N, int WMMA_K, int WARPS_PER_BLOCK>
__global__ void tensorcore_gemm(const half *A, const half *B, float *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
    const int warp_id = threadIdx.y;
    const int tile_m = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    const int tile_n = blockIdx.x;

    const int row = tile_m * WMMA_M;
    const int col = tile_n * WMMA_N;

    if (row >= M || col >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + row * lda + k, lda);
        wmma::load_matrix_sync(b_frag, B + k * ldb + col, ldb);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + row * ldc + col, c_frag, ldc, wmma::mem_row_major);
}

} // namespace

cudaError_t launch_wmma_cta_tiled(const half *A, const half *B, float *C,
                                  const GemmShape &shape, cudaStream_t stream) {
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

    const int tile_rows = static_cast<int>(shape.M / kWmmaM);
    const int tile_cols = static_cast<int>(shape.N / kWmmaN);

    const dim3 block(32, kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>(tile_cols),
                    static_cast<unsigned>((tile_rows + kWarpsPerBlock - 1) / kWarpsPerBlock));

    tensorcore_gemm<kWmmaM, kWmmaN, kWmmaK, kWarpsPerBlock><<<grid, block, 0, stream>>>(
        A, B, C, static_cast<int>(shape.M), static_cast<int>(shape.N), static_cast<int>(shape.K),
        static_cast<int>(shape.lda), static_cast<int>(shape.ldb), static_cast<int>(shape.ldc));
    return cudaGetLastError();
}