#ifndef TIMER_CUH
#define TIMER_CUH

#include "cuda_check.cuh"

#include <cuda_runtime.h>

namespace tc {

class CudaEventTimer {
  public:
    CudaEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    CudaEventTimer(const CudaEventTimer &) = delete;
    CudaEventTimer &operator=(const CudaEventTimer &) = delete;

    ~CudaEventTimer() {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

  private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

} // namespace tc

#endif // TIMER_CUH
