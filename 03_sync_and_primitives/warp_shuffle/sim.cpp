// Warp shuffle: lanes of a warp exchange registers directly, no shared memory and no __syncthreads.
// Classic use: a warp-wide sum in log2(32) = 5 steps with __shfl_down_sync.
// (The simulator emulates the warp's lockstep with a block barrier, see cuda_sim.hpp.)
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  const unsigned TPB = 64;  // two warps
  std::vector<int> in(TPB), warp_sum(TPB / 32, 0), trace;
  for (unsigned i = 0; i < TPB; ++i) in[i] = int(i) + 1;

  sim::launch(sim::dim3(1), sim::dim3(TPB), [&](sim::ThreadCtx& c) {
    int v = in[c.tid];
    for (unsigned off = 16; off > 0; off >>= 1) {
      v += c.shfl_down(v, off);                    // v += value held by lane (lane + off)
      if (c.tid == 0) trace.push_back(v);          // watch lane 0 of warp 0
    }
    if (c.lane == 0) warp_sum[c.warp] = v;         // lane 0 now holds the full warp sum
  });

  std::printf("lane 0 of warp 0 after offsets 16, 8, 4, 2, 1:");
  for (int v : trace) std::printf(" %d", v);
  std::printf("\n");
  for (unsigned w = 0; w < TPB / 32; ++w) {
    int ref = 0;
    for (unsigned l = 0; l < 32; ++l) ref += in[w * 32 + l];
    std::printf("warp %u sum = %d (reference %d) %s\n", w, warp_sum[w], ref, warp_sum[w] == ref ? "OK" : "MISMATCH");
  }
}
