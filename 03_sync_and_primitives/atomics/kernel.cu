// Real CUDA: histogram with global atomics vs shared-memory privatisation. Build: nvcc -O2 kernel.cu -o atomics
#include <cstdio>
#include <vector>
#include "../../common/cuda_check.cuh"

#define BINS 16

__global__ void hist_global(const unsigned char* data, unsigned* hist, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) atomicAdd(&hist[data[i] % BINS], 1u);
}
__global__ void hist_shared(const unsigned char* data, unsigned* hist, int n) {
  __shared__ unsigned local[BINS];
  if (threadIdx.x < BINS) local[threadIdx.x] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) atomicAdd(&local[data[i] % BINS], 1u);
  __syncthreads();
  if (threadIdx.x < BINS) atomicAdd(&hist[threadIdx.x], local[threadIdx.x]);
}

int main() {
  const int n = 1 << 26;
  std::vector<unsigned char> h(n);
  for (int i = 0; i < n; ++i) h[i] = (unsigned char)((i * 2654435761u) >> 24);
  unsigned char* d; unsigned* hist;
  CUDA_CHECK(cudaMalloc(&d, n)); CUDA_CHECK(cudaMalloc(&hist, BINS * 4));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
  float a = time_gpu_ms([&] { cudaMemset(hist, 0, BINS * 4); hist_global<<<1024, 256>>>(d, hist, n); });
  float b = time_gpu_ms([&] { cudaMemset(hist, 0, BINS * 4); hist_shared<<<1024, 256>>>(d, hist, n); });
  std::printf("global atomics: %.3f ms   shared privatised: %.3f ms   speedup %.2fx\n", a, b, a / b);
}
