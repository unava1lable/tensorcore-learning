#include "correctness.cuh"
#include "cuda_check.cuh"
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef TC_KERNEL_NAME
#define TC_KERNEL_NAME "unknown_kernel"
#endif

#ifndef TC_KERNEL_LAUNCH
#error "TC_KERNEL_LAUNCH must be defined by CMake for this test target."
#endif
namespace {

constexpr int kCaseNameWidth = 28;
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

void fill_constant(std::vector<half> &matrix, float value) {
    for (half &entry : matrix) {
        entry = __float2half(value);
    }
}

void fill_identity_a(std::vector<half> &A, const GemmShape &shape) {
    fill_constant(A, 0.0f);
    for (size_t m = 0; m < shape.M; ++m) {
        for (size_t k = 0; k < shape.K; ++k) {
            A[m * shape.lda + k] = __float2half(m == k ? 1.0f : 0.0f);
        }
    }
}

void fill_identity_b(std::vector<half> &B, const GemmShape &shape) {
    fill_constant(B, 0.0f);
    for (size_t k = 0; k < shape.K; ++k) {
        for (size_t n = 0; n < shape.N; ++n) {
            B[k * shape.ldb + n] = __float2half(k == n ? 1.0f : 0.0f);
        }
    }
}

void fill_identity_like(std::vector<half> &A, std::vector<half> &B, const GemmShape &shape) {
    fill_identity_a(A, shape);
    fill_identity_b(B, shape);
}

void fill_unique_a(std::vector<half> &A, const GemmShape &shape) {
    fill_constant(A, 0.0f);
    for (size_t m = 0; m < shape.M; ++m) {
        for (size_t k = 0; k < shape.K; ++k) {
            const int value = static_cast<int>((m * shape.K + k) % 32);
            A[m * shape.lda + k] = __float2half(static_cast<float>(value));
        }
    }
}

void fill_unique_b(std::vector<half> &B, const GemmShape &shape) {
    fill_constant(B, 0.0f);
    for (size_t k = 0; k < shape.K; ++k) {
        for (size_t n = 0; n < shape.N; ++n) {
            const int value = static_cast<int>((k * shape.N + n) % 32);
            B[k * shape.ldb + n] = __float2half(static_cast<float>(value));
        }
    }
}

void fill_signed_pattern(std::vector<half> &A, std::vector<half> &B, const GemmShape &shape) {
    fill_constant(A, 0.0f);
    fill_constant(B, 0.0f);
    for (size_t m = 0; m < shape.M; ++m) {
        for (size_t k = 0; k < shape.K; ++k) {
            const float sign = ((m + k) % 2 == 0) ? 1.0f : -1.0f;
            A[m * shape.lda + k] = __float2half(sign * (1.0f + static_cast<float>(k % 7)) / 8.0f);
        }
    }
    for (size_t k = 0; k < shape.K; ++k) {
        for (size_t n = 0; n < shape.N; ++n) {
            const float sign = ((k + n) % 3 == 0) ? -1.0f : 1.0f;
            B[k * shape.ldb + n] = __float2half(sign * (1.0f + static_cast<float>(n % 5)) / 16.0f);
        }
    }
}

bool run_case(const char *name, const GemmShape &shape, const std::vector<half> &A,
              const std::vector<half> &B, float c_initial = 0.0f) {
    const size_t a_bytes = A.size() * sizeof(half);
    const size_t b_bytes = B.size() * sizeof(half);
    const size_t c_elems = shape.M * shape.ldc;

    std::vector<float> C(c_elems, c_initial);
    std::vector<float> reference(c_elems, c_initial);

    half *d_A = nullptr;
    half *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), a_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), b_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), c_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), b_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), c_elems * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(TC_KERNEL_LAUNCH(d_A, d_B, d_C, shape));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    tc::reference_gemm(A, B, reference, shape.M, shape.N, shape.K, shape.lda, shape.ldb,
                       shape.ldc);
    const tc::ErrorStats stats = tc::compare_results(reference, C, 5.0e-1f, 5.0e-2f);

    const bool case_passed = stats.first_mismatch < 0;
    std::printf("[%-*s] max_abs_error=%10.8f max_rel_error=%10.8f normalized_error=%14.8e %s",
                kCaseNameWidth, name, stats.max_abs, stats.max_rel, stats.normalized,
                case_passed ? "PASSED" : "FAILED");
    if (!case_passed) {
        std::printf(" first_mismatch_index=%d reference=%.8f kernel=%.8f",
                    stats.first_mismatch, stats.reference_value, stats.kernel_value);
    }
    std::printf("\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return stats.first_mismatch < 0;
}

bool run_padded_mapping_cases() {
    const GemmShape padded_shape{32, 32, 32, 48, 48, 48};
    const size_t a_elems = padded_shape.M * padded_shape.lda;
    const size_t b_elems = padded_shape.K * padded_shape.ldb;

    std::vector<half> A(a_elems);
    std::vector<half> B(b_elems);

    bool passed = true;

    fill_identity_a(A, padded_shape);
    fill_unique_b(B, padded_shape);
    passed &= run_case("padded_identity_a_unique_b", padded_shape, A, B, -12345.0f);

    fill_unique_a(A, padded_shape);
    fill_identity_b(B, padded_shape);
    passed &= run_case("padded_unique_a_identity_b", padded_shape, A, B, -12345.0f);

    return passed;
}

} // namespace

int main(int argc, char **argv) {
    std::printf("kernel: %s\n", TC_KERNEL_NAME);
    const int M = parse_positive_arg(argv, argc, 1, 256);
    const int N = parse_positive_arg(argv, argc, 2, 256);
    const int K = parse_positive_arg(argv, argc, 3, 256);

    const GemmShape shape{static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K),
                          static_cast<size_t>(K), static_cast<size_t>(N), static_cast<size_t>(N)};

    if (shape.M % 16 != 0 || shape.N % 16 != 0 || shape.K % 16 != 0) {
        std::fprintf(stderr, "unsupported shape: M, N, and K must be multiples of 16\n");
        return EXIT_FAILURE;
    }

    const size_t a_elems = shape.M * shape.lda;
    const size_t b_elems = shape.K * shape.ldb;

    bool passed = true;

    std::vector<half> A(a_elems);
    std::vector<half> B(b_elems);

    fill_constant(A, 0.0f);
    fill_constant(B, 0.0f);
    passed &= run_case("zeros", shape, A, B);

    fill_constant(A, 1.0f);
    fill_constant(B, 1.0f);
    passed &= run_case("ones", shape, A, B);

    fill_identity_like(A, B, shape);
    passed &= run_case("identity_like", shape, A, B);

    tc::fill_matrix(A, 1234);
    tc::fill_matrix(B, 5678);
    passed &= run_case("deterministic_random", shape, A, B);

    fill_signed_pattern(A, B, shape);
    passed &= run_case("signed_pattern", shape, A, B);

    fill_identity_a(A, shape);
    fill_unique_b(B, shape);
    passed &= run_case("identity_a_unique_b", shape, A, B);

    fill_unique_a(A, shape);
    fill_identity_b(B, shape);
    passed &= run_case("unique_a_identity_b", shape, A, B);

    passed &= run_padded_mapping_cases();

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
