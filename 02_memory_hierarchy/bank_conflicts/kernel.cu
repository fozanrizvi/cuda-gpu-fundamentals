// Real CUDA: transpose through shared memory with and without padding. Build: nvcc -O2 kernel.cu -o bank_conflicts
#include <cstdio>
#include "../../common/cuda_check.cuh"

#define T 32

template <int PAD>
__global__ void transpose(const float* in, float* out, int N) {
  __shared__ float tile[T][T + PAD];
  int x = blockIdx.x * T + threadIdx.x, y = blockIdx.y * T + threadIdx.y;
  tile[threadIdx.y][threadIdx.x] = in[y * N + x];       // row write: conflict-free
  __syncthreads();
  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;
  out[y * N + x] = tile[threadIdx.x][threadIdx.y];      // column read: 32-way conflict when PAD == 0
}

int main() {
  const int N = 4096;
  float *in, *out;
  CUDA_CHECK(cudaMalloc(&in, (size_t)N * N * 4));
  CUDA_CHECK(cudaMalloc(&out, (size_t)N * N * 4));
  CUDA_CHECK(cudaMemset(in, 0, (size_t)N * N * 4));
  dim3 block(T, T), grid(N / T, N / T);
  float t0 = time_gpu_ms([&] { transpose<0><<<grid, block>>>(in, out, N); });
  float t1 = time_gpu_ms([&] { transpose<1><<<grid, block>>>(in, out, N); });
  std::printf("no padding: %.3f ms   padded [32][33]: %.3f ms   speedup %.2fx\n", t0, t1, t0 / t1);
  // Profile with: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared ./bank_conflicts
}
