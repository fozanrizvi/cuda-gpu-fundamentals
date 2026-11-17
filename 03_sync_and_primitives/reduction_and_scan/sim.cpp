// Block-level reduction (tree) and inclusive prefix scan (Hillis-Steele) in shared memory.
// Both are built from "each thread does a little work, __syncthreads, repeat".
#include <cstdio>
#include <numeric>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  const unsigned N = 128;
  std::vector<int> in(N);
  for (unsigned i = 0; i < N; ++i) in[i] = int(i % 10) + 1;

  // ---- Reduction: halve the active threads every step (sequential addressing => no bank conflicts)
  int red = 0, steps = 0;
  sim::launch(sim::dim3(1), sim::dim3(N), [&](sim::ThreadCtx& c) {
    int* s = c.shared<int>();
    s[c.tid] = in[c.tid];
    c.sync();
    for (unsigned stride = N / 2; stride > 0; stride >>= 1) {
      if (c.tid < stride) s[c.tid] += s[c.tid + stride];
      c.sync();
      if (c.tid == 0) ++steps;
    }
    if (c.tid == 0) red = s[0];
  }, N * sizeof(int));
  int ref_sum = std::accumulate(in.begin(), in.end(), 0);
  std::printf("reduction: %d (reference %d) in %d steps for %u elements  %s\n", red, ref_sum, steps, N,
              red == ref_sum ? "OK" : "MISMATCH");

  // ---- Scan: step d adds the element d positions to the left. Needs two syncs per step (read, then write).
  std::vector<int> scan(N), ref(N);
  std::partial_sum(in.begin(), in.end(), ref.begin());
  sim::launch(sim::dim3(1), sim::dim3(N), [&](sim::ThreadCtx& c) {
    int* s = c.shared<int>();
    s[c.tid] = in[c.tid];
    c.sync();
    for (unsigned d = 1; d < N; d <<= 1) {
      int add = (c.tid >= d) ? s[c.tid - d] : 0;  // read phase
      c.sync();
      s[c.tid] += add;                            // write phase
      c.sync();
    }
    scan[c.tid] = s[c.tid];
  }, N * sizeof(int));
  std::printf("scan:      %s; first 10 = ", scan == ref ? "matches std::partial_sum" : "MISMATCH");
  for (int i = 0; i < 10; ++i) std::printf("%d ", scan[i]);
  std::printf("\n(work: reduction does N-1 adds; Hillis-Steele scan does ~N log2 N adds but only log2 N steps)\n");
}
