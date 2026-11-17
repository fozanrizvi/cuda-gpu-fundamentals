// Shared helpers for the real-CUDA (.cu) examples. Needs nvcc.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err__ = (call);                                                       \
    if (err__ != cudaSuccess) {                                                       \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),     \
                   __FILE__, __LINE__);                                               \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

// Times a callable on the GPU with CUDA events; returns milliseconds.
template <class F>
float time_gpu_ms(F&& f, int reps = 10) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  f();  // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) f();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / reps;
}
