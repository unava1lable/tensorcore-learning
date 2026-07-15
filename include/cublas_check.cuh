#ifndef CUBLAS_CHECK_CUH
#define CUBLAS_CHECK_CUH

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

inline const char *cublas_status_string(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        cublasStatus_t status = (call);                                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                                             \
            std::fprintf(stderr, "cuBLAS error at %s:%d for %s: %s\n", __FILE__, __LINE__, \
                         #call, cublas_status_string(status));                             \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)

#endif // CUBLAS_CHECK_CUH
