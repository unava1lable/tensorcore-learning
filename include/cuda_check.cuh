#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", __FILE__, \
                         __LINE__, #call, cudaGetErrorString(status));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#endif