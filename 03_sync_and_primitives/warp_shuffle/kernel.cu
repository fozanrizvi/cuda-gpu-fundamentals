// Real CUDA: warp-level sum with __shfl_down_sync, then combine warps. Build: nvcc -O2 kernel.cu -o warp_shuffle
#include <cstdio>
#include <vector>
#include "../../common/cuda_check.cuh"

__inline__ __device__ int warp_sum(int v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;  // valid in lane 0
}

__global__ void block_sum(const int* in, int* out) {
  __shared__ int per_warp[32];
  int v = warp_sum(in[blockIdx.x * blockDim.x + threadIdx.x]);
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) per_warp[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < blockDim.x / 32) ? per_warp[lane] : 0;
    v = warp_sum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

int main() {
  const int blocks = 8, tpb = 256, n = blocks * tpb;
  std::vector<int> h(n, 1), r(blocks);
  int *d, *o;
  CUDA_CHECK(cudaMalloc(&d, n * 4)); CUDA_CHECK(cudaMalloc(&o, blocks * 4));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
  block_sum<<<blocks, tpb>>>(d, o);
  CUDA_CHECK(cudaMemcpy(r.data(), o, blocks * 4, cudaMemcpyDeviceToHost));
  for (int b = 0; b < blocks; ++b) std::printf("block %d sum = %d (expected %d)\n", b, r[b], tpb);
}
