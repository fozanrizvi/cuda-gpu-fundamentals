// Shared memory: tiled matrix multiply. Each block stages TILE x TILE sub-blocks of A and B in
// on-chip shared memory, so every global value is loaded once per tile instead of once per use.
#include <cmath>
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  const int N = 32, TILE = 8;
  std::vector<float> A(N * N), B(N * N), C(N * N, 0), ref(N * N, 0);
  for (int i = 0; i < N * N; ++i) { A[i] = float(i % 7) - 3; B[i] = float(i % 5) - 2; }
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c)
      for (int k = 0; k < N; ++k) ref[r * N + c] += A[r * N + k] * B[k * N + c];

  long global_loads = 0;
  sim::launch(sim::dim3(N / TILE, N / TILE), sim::dim3(TILE, TILE), [&](sim::ThreadCtx& c) {
    float* sA = c.shared<float>();            // TILE*TILE floats
    float* sB = sA + TILE * TILE;             // TILE*TILE floats
    int row = c.blockIdx.y * TILE + c.threadIdx.y, col = c.blockIdx.x * TILE + c.threadIdx.x;
    float acc = 0;
    long loads = 0;
    for (int t = 0; t < N / TILE; ++t) {
      sA[c.threadIdx.y * TILE + c.threadIdx.x] = A[row * N + t * TILE + c.threadIdx.x];   // 1 global load each
      sB[c.threadIdx.y * TILE + c.threadIdx.x] = B[(t * TILE + c.threadIdx.y) * N + col];
      loads += 2;
      c.sync();                                  // tile fully loaded before anyone reads it
      for (int k = 0; k < TILE; ++k) acc += sA[c.threadIdx.y * TILE + k] * sB[k * TILE + c.threadIdx.x];  // shared reads
      c.sync();                                  // everyone done before the tile is overwritten
    }
    C[row * N + col] = acc;
    sim::atomicAdd(&global_loads, loads);
  }, 2 * TILE * TILE * sizeof(float));

  double maxerr = 0;
  for (int i = 0; i < N * N; ++i) maxerr = std::fmax(maxerr, std::fabs(C[i] - ref[i]));
  long naive = 2L * N * N * N;  // naive kernel: every product loads one A and one B element from global memory
  std::printf("max |tiled - reference| = %g\n", maxerr);
  std::printf("global loads: naive = %ld, tiled = %ld  (%.0fx fewer, = TILE)\n", naive, global_loads, double(naive) / global_loads);
  std::printf("shared memory per block: %zu bytes\n", 2 * TILE * TILE * sizeof(float));
}
