#ifndef __CUDACC__
#error "01_wmma_block_tiled.cu must be compiled by nvcc or another CUDA compiler, not a host C++ compiler."
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
constexpr int kCtaM = 32;
constexpr int kCtaN = 32;
constexpr int kCtaK = 16;
constexpr int kWarpCols = 2;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock = kThreadsPerWarp * kWarpsPerBlock;
constexpr std::uintptr_t kWmmaAlignmentBytes = 32;

bool is_aligned_32(const void *ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % kWmmaAlignmentBytes == 0;
}

// Stage 1B: one CTA computes a 32x32 C tile. Four warps are arranged as a 2x2
// grid, each warp computes one 16x16 output tile, and the CTA cooperatively
// stages A[32x16] and B[16x32] through shared memory for explicit 2-way reuse.
template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void tensorcore_gemm(const half *A, const half *B, float *C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
    __shared__ __align__(32) half a_shm[kCtaM][kCtaK];
    __shared__ __align__(32) half b_shm[kCtaK][kCtaN];

    const int warp_id = threadIdx.y;
    const int tid = warp_id * blockDim.x + threadIdx.x;

    const int warp_m = warp_id / kWarpCols;
    const int warp_n = warp_id % kWarpCols;

    const int cta_row = blockIdx.y * kCtaM;
    const int cta_col = blockIdx.x * kCtaN;
    const int warp_row = warp_m * WMMA_M;
    const int warp_col = warp_n * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += kCtaK) {
        for (int idx = tid; idx < kCtaM * kCtaK; idx += kThreadsPerBlock) {
            const int row = idx / kCtaK;
            const int k = idx % kCtaK;
            const int global_row = cta_row + row;
            const int global_k = k0 + k;
            a_shm[row][k] = (global_row < M && global_k < K) ? A[global_row * lda + global_k]
                                                              : __float2half(0.0f);
        }

        for (int idx = tid; idx < kCtaK * kCtaN; idx += kThreadsPerBlock) {
            const int k = idx / kCtaN;
            const int col = idx % kCtaN;
            const int global_k = k0 + k;
            const int global_col = cta_col + col;
            b_shm[k][col] = (global_k < K && global_col < N) ? B[global_k * ldb + global_col]
                                                             : __float2half(0.0f);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, &a_shm[warp_row][0], kCtaK);
        wmma::load_matrix_sync(b_frag, &b_shm[0][warp_col], kCtaN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    const int row = cta_row + warp_row;
    const int col = cta_col + warp_col;
    if (row < M && col < N) {
        wmma::store_matrix_sync(C + row * ldc + col, c_frag, ldc, wmma::mem_row_major);
    }
}

} // namespace

cudaError_t launch_wmma_block_tiled(const half *A, const half *B, float *C,
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

    const dim3 block(kThreadsPerWarp, kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((shape.N + kCtaN - 1) / kCtaN),
                    static_cast<unsigned>((shape.M + kCtaM - 1) / kCtaM));

    tensorcore_gemm<kWmmaM, kWmmaN, kWmmaK><<<grid, block, 0, stream>>>(
        A, B, C, static_cast<int>(shape.M), static_cast<int>(shape.N), static_cast<int>(shape.K),
        static_cast<int>(shape.lda), static_cast<int>(shape.ldb), static_cast<int>(shape.ldc));
    return cudaGetLastError();
}