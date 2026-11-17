// Real CUDA: naive vs. tiled (shared-memory) matmul. Build: nvcc -O2 kernel.cu -o shared_memory
#include <cstdio>
#include "../../common/cuda_check.cuh"

#define TILE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0;
  for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];  // 2 global loads per FMA
  C[row * N + col] = acc;
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float sA[TILE][TILE], sB[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0;
  for (int t = 0; t < N / TILE; ++t) {
    sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
    sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }
  C[row * N + col] = acc;
}

int main() {
  const int N = 1024;
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, N * N * 4)); CUDA_CHECK(cudaMalloc(&B, N * N * 4)); CUDA_CHECK(cudaMalloc(&C, N * N * 4));
  CUDA_CHECK(cudaMemset(A, 0, N * N * 4)); CUDA_CHECK(cudaMemset(B, 0, N * N * 4));
  dim3 block(TILE, TILE), grid(N / TILE, N / TILE);
  float t1 = time_gpu_ms([&] { matmul_naive<<<grid, block>>>(A, B, C, N); });
  float t2 = time_gpu_ms([&] { matmul_tiled<<<grid, block>>>(A, B, C, N); });
  std::printf("naive: %.3f ms   tiled: %.3f ms   speedup %.2fx\n", t1, t2, t1 / t2);
}
