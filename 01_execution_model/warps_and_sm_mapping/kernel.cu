// Real CUDA: ask the hardware which SM each block really ran on (PTX %smid) and lane/warp ids.
// Build: nvcc -O2 kernel.cu -o warps_and_sm_mapping
#include <cstdio>
#include <vector>
#include "../../common/cuda_check.cuh"

__device__ unsigned smid() { unsigned v; asm volatile("mov.u32 %0, %%smid;" : "=r"(v)); return v; }
__device__ unsigned laneid() { unsigned v; asm volatile("mov.u32 %0, %%laneid;" : "=r"(v)); return v; }
__device__ unsigned warpid() { unsigned v; asm volatile("mov.u32 %0, %%warpid;" : "=r"(v)); return v; }

__global__ void record(unsigned* sm, unsigned* warp, unsigned* lane) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sm[i] = smid();
  warp[i] = warpid();  // hardware warp slot on the SM (not tid/32!)
  lane[i] = laneid();
}

int main() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("%s: %d SMs\n", p.name, p.multiProcessorCount);
  const int blocks = 12, tpb = 100, n = blocks * tpb;
  unsigned *sm, *warp, *lane;
  CUDA_CHECK(cudaMallocManaged(&sm, n * 4));
  CUDA_CHECK(cudaMallocManaged(&warp, n * 4));
  CUDA_CHECK(cudaMallocManaged(&lane, n * 4));
  record<<<blocks, tpb>>>(sm, warp, lane);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int b = 0; b < blocks; ++b)
    std::printf("block %2d ran on SM %u (thread 0: hw warp slot %u, lane %u; thread 99: warp slot %u, lane %u)\n", b,
                sm[b * tpb], warp[b * tpb], lane[b * tpb], warp[b * tpb + 99], lane[b * tpb + 99]);
  // Observe: consecutive blocks do NOT map to consecutive SMs; several blocks can share an SM.
  cudaFree(sm); cudaFree(warp); cudaFree(lane);
}
