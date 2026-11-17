// Real CUDA version of sim.cpp. Build: nvcc -O2 kernel.cu -o thread_hierarchy
#include <cstdio>
#include <vector>
#include "../../common/cuda_check.cuh"

__global__ void map_pixels(int* owner, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  owner[y * W + x] = blockIdx.y * gridDim.x + blockIdx.x;  // linear block id
}

int main() {
  const int W = 6, H = 5;
  dim3 block(4, 2);
  dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  int* d;
  CUDA_CHECK(cudaMalloc(&d, W * H * sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0xff, W * H * sizeof(int)));
  map_pixels<<<grid, block>>>(d, W, H);
  CUDA_CHECK(cudaGetLastError());
  std::vector<int> h(W * H);
  CUDA_CHECK(cudaMemcpy(h.data(), d, W * H * sizeof(int), cudaMemcpyDeviceToHost));
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) std::printf(" %d", h[y * W + x]);
    std::printf("\n");
  }
  cudaFree(d);
}
