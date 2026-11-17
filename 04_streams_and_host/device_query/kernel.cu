// Real CUDA: print the properties of every GPU. Build: nvcc -O2 kernel.cu -o device_query
#include <cstdio>
#include "../../common/cuda_check.cuh"

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, i));
    std::printf("Device %d: %s (sm_%d%d)\n", i, p.name, p.major, p.minor);
    std::printf("  SMs: %d\n  warp size: %d\n", p.multiProcessorCount, p.warpSize);
    std::printf("  max threads/SM: %d, max threads/block: %d, max blocks/SM: %d\n", p.maxThreadsPerMultiProcessor,
                p.maxThreadsPerBlock, p.maxBlocksPerMultiProcessor);
    std::printf("  registers/SM: %d, registers/block: %d\n", p.regsPerMultiprocessor, p.regsPerBlock);
    std::printf("  shared mem/SM: %zu KB, shared mem/block: %zu KB\n", p.sharedMemPerMultiprocessor / 1024,
                p.sharedMemPerBlock / 1024);
    std::printf("  constant mem: %zu KB, L2: %d KB, global mem: %.1f GB\n", p.totalConstMem / 1024, p.l2CacheSize / 1024,
                p.totalGlobalMem / 1e9);
    std::printf("  max grid: %d x %d x %d, max block: %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2],
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("  async engines (copy/compute overlap): %d, concurrent kernels: %d, managed memory: %d\n", p.asyncEngineCount,
                p.concurrentKernels, p.managedMemory);
  }
}
