// __syncthreads(): a barrier for all threads of ONE block. Without it, threads that read shared memory
// written by other threads race with those writes. Here half the block is made "slow" to expose the race
// deterministically (on hardware, different warps progress at different speeds).
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "cuda_sim.hpp"

static int run(bool use_sync) {
  const int N = 64;
  std::vector<int> in(N), out(N, 0);
  for (int i = 0; i < N; ++i) in[i] = i;
  sim::launch(sim::dim3(1), sim::dim3(N), [&](sim::ThreadCtx& c) {
    int* s = c.shared<int>();
    if (c.tid == 0)
      for (int i = 0; i < N; ++i) s[i] = -1;  // sentinel = "not written yet"
    c.sync();
    if (c.tid >= N / 2) std::this_thread::sleep_for(std::chrono::milliseconds(20));  // slow warps
    s[c.tid] = in[c.tid];                      // stage into shared memory
    if (use_sync) c.sync();                    // <-- the barrier
    out[c.tid] = s[N - 1 - c.tid];             // read a value written by a *different* thread
  }, N * sizeof(int));
  int wrong = 0;
  for (int i = 0; i < N; ++i) wrong += (out[i] != in[N - 1 - i]);
  return wrong;
}

int main() {
  std::printf("reverse a 64-element array through shared memory\n");
  std::printf("  with __syncthreads():    %2d wrong elements\n", run(true));
  std::printf("  without __syncthreads(): %2d wrong elements (read the -1 sentinel => data race)\n", run(false));
  std::printf("\nRules: every thread in the block must reach the same __syncthreads(); never call it in a branch\n"
              "that only some threads take. It synchronises a block only - blocks cannot barrier with each other\n"
              "inside a kernel (that is why a grid-wide reduction needs a second kernel launch or atomics).\n");
}
