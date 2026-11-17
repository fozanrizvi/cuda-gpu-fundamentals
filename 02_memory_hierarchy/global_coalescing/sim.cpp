// Global-memory coalescing: the 32 lanes of a warp issue one load together. The memory system
// fetches 32-byte sectors (128-byte lines). Contiguous addresses -> few sectors; strided -> many.
#include <cstdint>
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  std::printf("%-26s %12s %10s %12s\n", "access pattern (float)", "sectors/warp", "lines", "efficiency");
  struct P { const char* name; unsigned stride, offset; };
  P pats[] = {{"in[i]            stride 1", 1, 0}, {"in[i+1]          misaligned", 1, 1},
              {"in[2*i]          stride 2", 2, 0}, {"in[4*i]          stride 4", 4, 0},
              {"in[8*i]          stride 8", 8, 0}, {"in[32*i]        stride 32", 32, 0}};
  for (auto p : pats) {
    std::vector<uint64_t> addr(128);
    sim::launch(sim::dim3(1), sim::dim3(128), [&](sim::ThreadCtx& c) {
      addr[c.tid] = uint64_t(c.tid * p.stride + p.offset) * sizeof(float);  // byte address this thread loads
    });
    int sectors = 0, lines = 0, warps = 0;
    sim::for_each_warp(addr, [&](size_t, const std::vector<uint64_t>& w) {
      sectors += sim::sectors_touched(w);
      lines += sim::lines_touched(w);
      ++warps;
    });
    double s = double(sectors) / warps;
    // useful bytes = 32 lanes * 4 B = 128 B; bytes fetched = sectors * 32 B
    std::printf("%-26s %12.1f %10.1f %11.0f%%\n", p.name, s, double(lines) / warps, 100.0 * 128 / (s * 32));
  }
  std::printf("\nStride-1 reads 4 sectors (128 B useful of 128 B fetched). Stride-32 pulls 32 sectors = 1 KB\n"
              "to use 128 B: ~12%% efficiency, so a memory-bound kernel runs ~8x slower.\n");
}
