// Real CUDA: query occupancy with the runtime API. Build: nvcc -O2 kernel.cu -o occupancy
#include <cstdio>
#include "../../common/cuda_check.cuh"

__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  int maxWarps = p.maxThreadsPerMultiProcessor / p.warpSize;
  std::printf("%s: %d SMs, max %d warps/SM\n", p.name, p.multiProcessorCount, maxWarps);
  for (int tpb : {32, 64, 128, 256, 512, 1024}) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, saxpy, tpb, 0));
    int warps = blocks * ((tpb + 31) / 32);
    std::printf("block %4d: %2d blocks/SM -> %2d warps/SM = %3.0f%% occupancy\n", tpb, blocks, warps, 100.0 * warps / maxWarps);
  }
  int minGrid = 0, best = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &best, saxpy, 0, 0));
  std::printf("runtime suggests block size %d (min grid to fill device: %d)\n", best, minGrid);
}
