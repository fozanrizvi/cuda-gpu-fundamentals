// Real CUDA: explicit copies vs managed memory (with/without prefetch), touching only part of a buffer.
// Build: nvcc -O2 kernel.cu -o unified_memory   (page-fault migration needs Linux + Pascal or newer)
#include <chrono>
#include <cstdio>
#include "../../common/cuda_check.cuh"

__global__ void touch(float* x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}

template <class F>
static double wall_ms(F f) {
  auto t0 = std::chrono::steady_clock::now();
  f();
  CUDA_CHECK(cudaDeviceSynchronize());
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  const size_t n = 1ull << 28;  // 1 GB of floats
  const size_t used = n / 4;    // kernel touches 25%
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  float* h = (float*)malloc(n * 4);
  for (size_t i = 0; i < n; ++i) h[i] = 0;
  float* d;
  CUDA_CHECK(cudaMalloc(&d, n * 4));
  double t_explicit = wall_ms([&] {
    CUDA_CHECK(cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice));
    touch<<<(used + 255) / 256, 256>>>(d, used);
  });

  float* m;
  CUDA_CHECK(cudaMallocManaged(&m, n * 4));
  for (size_t i = 0; i < n; ++i) m[i] = 0;  // pages start resident on the CPU
  double t_managed = wall_ms([&] { touch<<<(used + 255) / 256, 256>>>(m, used); });

  for (size_t i = 0; i < n; ++i) m[i] = 0;  // pull everything back to the CPU
  double t_prefetch = wall_ms([&] {
    CUDA_CHECK(cudaMemPrefetchAsync(m, used * 4, dev, 0));  // CUDA 13 changed this signature (cudaMemLocation)
    touch<<<(used + 255) / 256, 256>>>(m, used);
  });

  std::printf("explicit full copy: %.1f ms   managed (faults): %.1f ms   managed + prefetch: %.1f ms\n", t_explicit,
              t_managed, t_prefetch);
}
