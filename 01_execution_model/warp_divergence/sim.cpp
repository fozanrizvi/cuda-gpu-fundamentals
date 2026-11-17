// Warp divergence: a warp executes one instruction stream. If lanes take different branch paths,
// the hardware runs each path in turn with the other lanes masked off. Cost ~ number of distinct paths.
#include <cstdio>
#include <functional>
#include <set>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  struct Pattern { const char* name; std::function<int(const sim::ThreadCtx&)> path; };
  std::vector<Pattern> patterns = {
      {"if (lane % 2 == 0)         -> 2 paths inside every warp", [](const sim::ThreadCtx& c) { return int(c.lane % 2); }},
      {"if ((tid / 32) % 2 == 0)   -> branch is warp-uniform   ", [](const sim::ThreadCtx& c) { return int((c.tid / 32) % 2); }},
      {"switch (lane % 4)          -> 4 paths inside every warp", [](const sim::ThreadCtx& c) { return int(c.lane % 4); }},
      {"if (lane < 8)              -> 2 paths, uneven split    ", [](const sim::ThreadCtx& c) { return int(c.lane < 8); }},
  };

  const unsigned threads = 128;
  for (auto& p : patterns) {
    std::vector<int> taken(threads);
    sim::launch(sim::dim3(1), sim::dim3(threads), [&](sim::ThreadCtx& c) { taken[c.tid] = p.path(c); });

    int total_passes = 0, warps = 0;
    unsigned example_mask = 0;
    sim::for_each_warp(taken, [&](size_t w, const std::vector<int>& slice) {
      std::set<int> distinct(slice.begin(), slice.end());
      total_passes += int(distinct.size());
      ++warps;
      if (w == 0) {  // active mask of the *first* path executed by warp 0
        int first = slice[0];
        for (size_t l = 0; l < slice.size(); ++l)
          if (slice[l] == first) example_mask |= 1u << l;
      }
    });
    std::printf("%s passes/warp = %.1f  (warp0 first-path active mask = 0x%08x)\n", p.name,
                double(total_passes) / warps, example_mask);
  }
  std::printf("\nA warp-uniform branch costs 1 pass; k divergent paths cost ~k passes (all paths' latency adds up).\n");
}
