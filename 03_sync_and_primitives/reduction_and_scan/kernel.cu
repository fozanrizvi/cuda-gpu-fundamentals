// Real CUDA: shared-memory tree reduction + Hillis-Steele scan within one block.
// Build: nvcc -O2 kernel.cu -o reduction_and_scan
#include <cstdio>
#include <vector>
#include "../../common/cuda_check.cuh"

#define N 256

__global__ void reduce_and_scan(const int* in, int* sum, int* scan) {
  __shared__ int s[N];
  int t = threadIdx.x;
  s[t] = in[t];
  __syncthreads();
  for (int stride = N / 2; stride > 0; stride >>= 1) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) *sum = s[0];

  s[t] = in[t];
  __syncthreads();
  for (int d = 1; d < N; d <<= 1) {
    int add = (t >= d) ? s[t - d] : 0;
    __syncthreads();
    s[t] += add;
    __syncthreads();
  }
  scan[t] = s[t];
}

int main() {
  std::vector<int> h(N, 1), scan(N);
  int *din, *dsum, *dscan, sum = 0;
  CUDA_CHECK(cudaMalloc(&din, N * 4)); CUDA_CHECK(cudaMalloc(&dsum, 4)); CUDA_CHECK(cudaMalloc(&dscan, N * 4));
  CUDA_CHECK(cudaMemcpy(din, h.data(), N * 4, cudaMemcpyHostToDevice));
  reduce_and_scan<<<1, N>>>(din, dsum, dscan);
  CUDA_CHECK(cudaMemcpy(&sum, dsum, 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(scan.data(), dscan, N * 4, cudaMemcpyDeviceToHost));
  std::printf("sum = %d (expected %d), scan[N-1] = %d (expected %d)\n", sum, N, scan[N - 1], N);
}
