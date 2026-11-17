// Atomics: a histogram. Many threads increment the same bin -> need atomic read-modify-write.
// Optimisation: privatise the histogram in shared memory per block, then merge once into global memory.
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  const int N = 4096, BINS = 16, BLOCKS = 8, TPB = 128;
  std::vector<unsigned> data(N), ref(BINS, 0);
  unsigned x = 12345;
  for (int i = 0; i < N; ++i) { x = x * 1664525u + 1013904223u; data[i] = (x >> 16) % BINS; ++ref[data[i]]; }

  // Version 1: every thread does atomicAdd on the global histogram.
  std::vector<unsigned> g1(BINS, 0);
  long global_atomics_1 = 0;
  sim::launch(sim::dim3(BLOCKS), sim::dim3(TPB), [&](sim::ThreadCtx& c) {
    for (int i = c.global_x(); i < N; i += BLOCKS * TPB) {
      sim::atomicAdd(&g1[data[i]], 1u);
      sim::atomicAdd(&global_atomics_1, 1L);
    }
  });

  // Version 2: shared-memory private histogram per block, merged at the end.
  std::vector<unsigned> g2(BINS, 0);
  long global_atomics_2 = 0;
  sim::launch(sim::dim3(BLOCKS), sim::dim3(TPB), [&](sim::ThreadCtx& c) {
    unsigned* local = c.shared<unsigned>();
    if (c.tid < BINS) local[c.tid] = 0;
    c.sync();
    for (int i = c.global_x(); i < N; i += BLOCKS * TPB) sim::atomicAdd(&local[data[i]], 1u);  // cheap, on-chip
    c.sync();
    if (c.tid < BINS) {
      sim::atomicAdd(&g2[c.tid], local[c.tid]);       // one global atomic per bin per block
      sim::atomicAdd(&global_atomics_2, 1L);
    }
  }, BINS * sizeof(unsigned));

  std::printf("global-only histogram correct:   %s  (%ld global atomics)\n", g1 == ref ? "yes" : "NO", global_atomics_1);
  std::printf("shared-privatised correct:       %s  (%ld global atomics)\n", g2 == ref ? "yes" : "NO", global_atomics_2);
  std::printf("=> %.0fx fewer contended global atomics\n", double(global_atomics_1) / global_atomics_2);
}
