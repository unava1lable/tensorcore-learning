#ifndef CORRECTNESS_CUH
#define CORRECTNESS_CUH

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace tc {

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double normalized = 0.0;
    int first_mismatch = -1;
    float reference_value = 0.0f;
    float kernel_value = 0.0f;
};

inline void fill_matrix(std::vector<half> &matrix, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (half &value : matrix) {
        value = __float2half(dist(rng));
    }
}

inline void reference_gemm(const std::vector<half> &A, const std::vector<half> &B,
                           std::vector<float> &C, size_t M, size_t N, size_t K,
                           size_t lda = 0, size_t ldb = 0, size_t ldc = 0) {
    const size_t actual_lda = lda == 0 ? K : lda;
    const size_t actual_ldb = ldb == 0 ? N : ldb;
    const size_t actual_ldc = ldc == 0 ? N : ldc;

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += __half2float(A[m * actual_lda + k]) * __half2float(B[k * actual_ldb + n]);
            }
            C[m * actual_ldc + n] = acc;
        }
    }
}

inline ErrorStats compare_results(const std::vector<float> &reference,
                                  const std::vector<float> &actual,
                                  float abs_tol, float rel_tol) {
    if (reference.size() != actual.size()) {
        throw std::invalid_argument("reference and actual sizes differ");
    }

    ErrorStats stats;
    double sum_sq_error = 0.0;
    double sum_sq_ref = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        const float ref = reference[i];
        const float got = actual[i];

        if (!std::isfinite(ref) || !std::isfinite(got)) {
            const bool exact_nonfinite_match = (ref == got);
            if (!exact_nonfinite_match) {
                stats.max_abs = std::numeric_limits<float>::infinity();
                stats.max_rel = std::numeric_limits<float>::infinity();
                if (stats.first_mismatch < 0) {
                    stats.first_mismatch = static_cast<int>(i);
                    stats.reference_value = ref;
                    stats.kernel_value = got;
                }
            }
            continue;
        }

        const float abs_error = std::fabs(ref - got);
        const float rel_error = abs_error / std::max(std::fabs(ref), 1.0e-6f);
        const float allowed_error = abs_tol + rel_tol * std::fabs(ref);

        stats.max_abs = std::max(stats.max_abs, abs_error);
        stats.max_rel = std::max(stats.max_rel, rel_error);

        if (stats.first_mismatch < 0 && abs_error > allowed_error) {
            stats.first_mismatch = static_cast<int>(i);
            stats.reference_value = ref;
            stats.kernel_value = got;
        }

        sum_sq_error += static_cast<double>(abs_error) * abs_error;
        sum_sq_ref += static_cast<double>(ref) * ref;
    }

    stats.normalized = std::sqrt(sum_sq_error / std::max(sum_sq_ref, 1.0e-30));
    return stats;
}

} // namespace tc

#endif // CORRECTNESS_CUH
