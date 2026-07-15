#include "correctness.cuh"
#include "cublas_check.cuh"
#include "cuda_check.cuh"
#include "kernels.cuh"
#include "timer.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#ifndef TC_KERNEL_NAME
#define TC_KERNEL_NAME "unknown_kernel"
#endif

#ifndef TC_KERNEL_LAUNCH
#error "TC_KERNEL_LAUNCH must be defined by CMake for this benchmark target."
#endif
namespace {

struct BenchmarkStats {
    float min_ms = 0.0f;
    float median_ms = 0.0f;
    float avg_ms = 0.0f;
};

int parse_positive_arg(char **argv, int argc, int index, int fallback) {
    if (index >= argc) {
        return fallback;
    }

    const int value = std::atoi(argv[index]);
    if (value <= 0) {
        std::fprintf(stderr, "argument %d must be positive, got %d\n", index, value);
        std::exit(EXIT_FAILURE);
    }
    return value;
}

BenchmarkStats summarize(std::vector<float> samples_ms) {
    std::sort(samples_ms.begin(), samples_ms.end());

    BenchmarkStats stats;
    stats.min_ms = samples_ms.front();
    stats.median_ms = samples_ms[samples_ms.size() / 2];
    stats.avg_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0f) /
                   static_cast<float>(samples_ms.size());
    return stats;
}

double tflops_for(float elapsed_ms, int M, int N, int K) {
    return 2.0 * M * N * K / (elapsed_ms * 1.0e-3) / 1.0e12;
}

void print_stats(const char *name, const BenchmarkStats &stats, int M, int N, int K) {
    std::printf("%s:\n", name);
    std::printf("  min_ms: %.4f, median_ms: %.4f, avg_ms: %.4f\n", stats.min_ms,
                stats.median_ms, stats.avg_ms);
    std::printf("  best_tflops: %.3f, median_tflops: %.3f, tflops_from_avg_ms: %.3f\n",
                tflops_for(stats.min_ms, M, N, K), tflops_for(stats.median_ms, M, N, K),
                tflops_for(stats.avg_ms, M, N, K));
}

void run_cublas_row_major(cublasHandle_t handle, const half *A, const half *B, float *C,
                          const GemmShape &shape) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. Row-major C[M,N] = A[M,K] * B[K,N] is equivalent to
    // column-major C^T[N,M] = B^T[N,K] * A^T[K,M] using the same memory buffers.
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(shape.N),
                              static_cast<int>(shape.M), static_cast<int>(shape.K), &alpha, B,
                              CUDA_R_16F, static_cast<int>(shape.ldb), A, CUDA_R_16F,
                              static_cast<int>(shape.lda), &beta, C, CUDA_R_32F,
                              static_cast<int>(shape.ldc), CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

BenchmarkStats benchmark_custom(const half *A, const half *B, float *C, const GemmShape &shape,
                                int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        CUDA_CHECK(TC_KERNEL_LAUNCH(A, B, C, shape));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples_ms;
    samples_ms.reserve(static_cast<size_t>(bench_iters));
    tc::CudaEventTimer timer;
    for (int i = 0; i < bench_iters; ++i) {
        timer.start();
        CUDA_CHECK(TC_KERNEL_LAUNCH(A, B, C, shape));
        samples_ms.push_back(timer.stop());
    }

    return summarize(samples_ms);
}

BenchmarkStats benchmark_cublas(cublasHandle_t handle, const half *A, const half *B, float *C,
                                const GemmShape &shape, int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        run_cublas_row_major(handle, A, B, C, shape);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples_ms;
    samples_ms.reserve(static_cast<size_t>(bench_iters));
    tc::CudaEventTimer timer;
    for (int i = 0; i < bench_iters; ++i) {
        timer.start();
        run_cublas_row_major(handle, A, B, C, shape);
        samples_ms.push_back(timer.stop());
    }

    return summarize(samples_ms);
}

} // namespace

int main(int argc, char **argv) {
    const int M = parse_positive_arg(argv, argc, 1, 1024);
    const int N = parse_positive_arg(argv, argc, 2, 1024);
    const int K = parse_positive_arg(argv, argc, 3, 1024);
    const int warmup_iters = parse_positive_arg(argv, argc, 4, 5);
    const int bench_iters = parse_positive_arg(argv, argc, 5, 20);

    const GemmShape shape{static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K),
                          static_cast<size_t>(K), static_cast<size_t>(N), static_cast<size_t>(N)};

    if (shape.M % 16 != 0 || shape.N % 16 != 0 || shape.K % 16 != 0) {
        std::fprintf(stderr, "unsupported shape: M, N, and K must be multiples of 16\n");
        return EXIT_FAILURE;
    }

    const size_t a_elems = shape.M * shape.lda;
    const size_t b_elems = shape.K * shape.ldb;
    const size_t c_elems = shape.M * shape.ldc;

    std::vector<half> h_A(a_elems);
    std::vector<half> h_B(b_elems);

    tc::fill_matrix(h_A, 1234);
    tc::fill_matrix(h_B, 5678);

    half *d_A = nullptr;
    half *d_B = nullptr;
    float *d_C_custom = nullptr;
    float *d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_custom), c_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_cublas), c_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_custom, 0, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C_cublas, 0, c_elems * sizeof(float)));

    cublasHandle_t cublas = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    const BenchmarkStats custom_stats =
        benchmark_custom(d_A, d_B, d_C_custom, shape, warmup_iters, bench_iters);
    const BenchmarkStats cublas_stats =
        benchmark_cublas(cublas, d_A, d_B, d_C_cublas, shape, warmup_iters, bench_iters);

    std::vector<float> h_C_custom(c_elems, 0.0f);
    std::vector<float> h_C_cublas(c_elems, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, c_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, c_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const tc::ErrorStats cublas_diff =
        tc::compare_results(h_C_cublas, h_C_custom, 5.0e-1f, 5.0e-2f);
    const bool correct = cublas_diff.first_mismatch < 0;

    std::printf("shape: M=%d N=%d K=%d\n", M, N, K);
    std::printf("warmup_iters: %d, bench_iters: %d\n", warmup_iters, bench_iters);
    print_stats(TC_KERNEL_NAME, custom_stats, M, N, K);
    print_stats("cublas_gemm_ex", cublas_stats, M, N, K);
    std::printf("kernel_vs_cublas: max_abs_error=%.8f max_rel_error=%.8f normalized_error=%.8e\n",
                cublas_diff.max_abs, cublas_diff.max_rel, cublas_diff.normalized);
    if (!correct) {
        std::printf("kernel_vs_cublas first_mismatch_index=%d cublas=%.8f kernel=%.8f\n",
                    cublas_diff.first_mismatch, cublas_diff.reference_value,
                    cublas_diff.kernel_value);
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_custom));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
