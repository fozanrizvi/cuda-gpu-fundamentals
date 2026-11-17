// Constant memory and registers.
//  (1) Constant memory is cached and *broadcasts* when all lanes of a warp read the same address;
//      reads of k distinct addresses in one warp are serialised into k passes.
//  (2) Registers live in the SM's register file. More registers per thread => fewer resident warps.
#include <cstdio>
#include <set>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  sim::banner("Constant memory: distinct addresses per warp read");
  struct P { const char* name; unsigned (*idx)(unsigned lane); };
  P pats[] = {{"coef[0]           (all lanes same)", [](unsigned) { return 0u; }},
              {"coef[lane / 8]    (4 distinct)", [](unsigned l) { return l / 8; }},
              {"coef[lane]        (32 distinct)", [](unsigned l) { return l; }}};
  for (auto p : pats) {
    std::vector<unsigned> idx(32);
    sim::launch(sim::dim3(1), sim::dim3(32), [&](sim::ThreadCtx& c) { idx[c.tid] = p.idx(c.lane); });
    std::set<unsigned> distinct(idx.begin(), idx.end());
    std::printf("  %-36s -> %2zu pass(es)\n", p.name, distinct.size());
  }

  sim::banner("Registers per thread vs. residency (256 threads/block)");
  std::printf("  %-10s %-10s %-12s %s\n", "regs/thr", "blocks/SM", "warps/SM", "occupancy");
  for (int regs : {16, 32, 64, 96, 128, 255}) {
    auto o = sim::occupancy(256, regs, 0);
    std::printf("  %-10d %-10d %-12d %.0f%%  (%s)\n", regs, o.blocks_per_sm, o.warps_per_sm, o.fraction * 100, o.limiter);
  }
  std::printf("\nIf the compiler needs more than 255 registers (or you cap them with -maxrregcount) the excess\n"
              "'spills' to local memory - which is really off-chip global memory, cached in L1/L2.\n");
}
