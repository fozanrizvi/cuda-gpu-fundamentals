// Real CUDA: constant memory broadcast vs. divergent access, and how to see register usage / spills.
// Build: nvcc -O2 --ptxas-options=-v kernel.cu -o constant_and_registers
//   ptxas prints "Used N registers, M bytes spill stores" for each kernel.
//   Try: nvcc -O2 -maxrregcount=32 --ptxas-options=-v ...  to force spills.
#include <cstdio>
#include "../../common/cuda_check.cuh"

__constant__ float c_coef[256];

__global__ void uniform_read(float* out, int n) {  // every lane reads the same constant -> broadcast
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0;
  for (int k = 0; k < 256; ++k) acc += c_coef[k] * i;
  if (i < n) out[i] = acc;
}
__global__ void divergent_read(float* out, int n) {  // each lane reads a different constant -> serialised
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0;
  for (int k = 0; k < 256; ++k) acc += c_coef[(k + threadIdx.x) & 255] * i;
  if (i < n) out[i] = acc;
}

__global__ void register_heavy(float* out, const int* idx, int n) {  // dynamically indexed array -> local memory
  float scratch[64];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < 64; ++k) scratch[k] = i * k;
  if (i < n) out[i] = scratch[idx[i] & 63];
}

int main() {
  float h[256];
  for (int i = 0; i < 256; ++i) h[i] = 1.0f / (i + 1);
  CUDA_CHECK(cudaMemcpyToSymbol(c_coef, h, sizeof(h)));
  const int n = 1 << 22;
  float* out; int* idx;
  CUDA_CHECK(cudaMalloc(&out, n * 4)); CUDA_CHECK(cudaMalloc(&idx, n * 4)); CUDA_CHECK(cudaMemset(idx, 0, n * 4));
  float a = time_gpu_ms([&] { uniform_read<<<n / 256, 256>>>(out, n); });
  float b = time_gpu_ms([&] { divergent_read<<<n / 256, 256>>>(out, n); });
  float c = time_gpu_ms([&] { register_heavy<<<n / 256, 256>>>(out, idx, n); });
  std::printf("constant uniform: %.3f ms   constant divergent: %.3f ms   local-memory kernel: %.3f ms\n", a, b, c);
}
