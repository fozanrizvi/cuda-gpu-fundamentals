// Real CUDA: chunked pipeline on 1 vs 4 streams, timed with events. Build: nvcc -O2 kernel.cu -o streams_events
#include <cstdio>
#include "../../common/cuda_check.cuh"

__global__ void heavy(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; for (int k = 0; k < 500; ++k) v = v * 1.0001f + 0.1f; x[i] = v; }
}

static float run(int streams, float* h, float* d, int n, int chunks) {
  cudaStream_t s[8];
  for (int i = 0; i < streams; ++i) CUDA_CHECK(cudaStreamCreate(&s[i]));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  int cn = n / chunks;
  CUDA_CHECK(cudaEventRecord(a));
  for (int c = 0; c < chunks; ++c) {
    cudaStream_t st = s[c % streams];
    float *hp = h + (size_t)c * cn, *dp = d + (size_t)c * cn;
    CUDA_CHECK(cudaMemcpyAsync(dp, hp, cn * 4, cudaMemcpyHostToDevice, st));
    heavy<<<(cn + 255) / 256, 256, 0, st>>>(dp, cn);
    CUDA_CHECK(cudaMemcpyAsync(hp, dp, cn * 4, cudaMemcpyDeviceToHost, st));
  }
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  for (int i = 0; i < streams; ++i) CUDA_CHECK(cudaStreamDestroy(s[i]));
  return ms;
}

int main() {
  const int n = 1 << 24, chunks = 8;
  float *h, *d;
  CUDA_CHECK(cudaMallocHost(&h, n * 4));  // pinned memory: required for real async copies
  CUDA_CHECK(cudaMalloc(&d, n * 4));
  for (int i = 0; i < n; ++i) h[i] = 1.0f;
  run(1, h, d, n, chunks);  // warm-up
  float t1 = run(1, h, d, n, chunks), t4 = run(4, h, d, n, chunks);
  std::printf("1 stream: %.2f ms   4 streams: %.2f ms   speedup %.2fx\n", t1, t4, t1 / t4);
}
