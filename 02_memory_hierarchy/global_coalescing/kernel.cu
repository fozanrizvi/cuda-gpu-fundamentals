// Real CUDA: effective bandwidth of strided reads. Build: nvcc -O2 kernel.cu -o global_coalescing
#include <cstdio>
#include "../../common/cuda_check.cuh"

__global__ void strided_copy(const float* in, float* out, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[(long long)i * stride % n];  // stride 1 = coalesced
}

int main() {
  const int n = 1 << 24;
  float *in, *out;
  CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(in, 0, n * sizeof(float)));
  for (int stride : {1, 2, 4, 8, 16, 32}) {
    float ms = time_gpu_ms([&] { strided_copy<<<n / 256, 256>>>(in, out, n, stride); });
    std::printf("stride %2d: %7.3f ms  useful bandwidth %7.1f GB/s\n", stride, ms, 2.0 * n * sizeof(float) / ms / 1e6);
  }
  cudaFree(in); cudaFree(out);
}
