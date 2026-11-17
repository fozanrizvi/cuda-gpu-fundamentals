// Real CUDA: reverse within a block via shared memory. Remove the __syncthreads() to see the race
// (results become nondeterministic for blocks > 32 threads, since multiple warps are involved).
// Build: nvcc -O2 kernel.cu -o syncthreads
#include <cstdio>
#include <vector>
#include "../../common/cuda_check.cuh"

template <bool SYNC>
__global__ void reverse_block(const int* in, int* out) {
  __shared__ int s[256];
  int t = threadIdx.x, base = blockIdx.x * blockDim.x;
  s[t] = in[base + t];
  if (SYNC) __syncthreads();
  out[base + t] = s[blockDim.x - 1 - t];
}

int main() {
  const int blocks = 4096, tpb = 256, n = blocks * tpb;
  std::vector<int> h(n), r(n);
  for (int i = 0; i < n; ++i) h[i] = i;
  int *din, *dout;
  CUDA_CHECK(cudaMalloc(&din, n * 4)); CUDA_CHECK(cudaMalloc(&dout, n * 4));
  CUDA_CHECK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
  for (int sync = 1; sync >= 0; --sync) {
    CUDA_CHECK(cudaMemset(dout, 0, n * 4));
    if (sync) reverse_block<true><<<blocks, tpb>>>(din, dout); else reverse_block<false><<<blocks, tpb>>>(din, dout);
    CUDA_CHECK(cudaMemcpy(r.data(), dout, n * 4, cudaMemcpyDeviceToHost));
    long wrong = 0;
    for (int b = 0; b < blocks; ++b)
      for (int t = 0; t < tpb; ++t) wrong += r[b * tpb + t] != h[b * tpb + tpb - 1 - t];
    std::printf("%s __syncthreads: %ld wrong elements\n", sync ? "with   " : "without", wrong);
  }
}
