// Real CUDA: time a divergent branch vs. a warp-uniform branch doing identical total work.
// Build: nvcc -O2 kernel.cu -o warp_divergence
#include <cstdio>
#include "../../common/cuda_check.cuh"

__device__ float work_a(float x) { for (int i = 0; i < 2000; ++i) x = x * 1.0001f + 0.5f; return x; }
__device__ float work_b(float x) { for (int i = 0; i < 2000; ++i) x = x * 0.9999f - 0.5f; return x; }

__global__ void divergent(float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = (threadIdx.x % 2 == 0) ? work_a(i) : work_b(i);  // both paths run inside every warp
}
__global__ void uniform(float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = ((threadIdx.x / 32) % 2 == 0) ? work_a(i) : work_b(i);  // whole warps agree
}

int main() {
  const int n = 1 << 20;
  float* out;
  CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
  float t1 = time_gpu_ms([&] { divergent<<<n / 256, 256>>>(out); });
  float t2 = time_gpu_ms([&] { uniform<<<n / 256, 256>>>(out); });
  std::printf("divergent: %.3f ms   warp-uniform: %.3f ms   slowdown: %.2fx\n", t1, t2, t1 / t2);
  cudaFree(out);
}
