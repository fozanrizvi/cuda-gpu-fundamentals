// Warps and SM mapping: blocks are handed to SMs; each block is split into warps of 32 threads.
// A block size that is not a multiple of 32 leaves lanes of the last warp idle (wasted hardware).
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  sim::device().sm_count = 4;  // shrink the GPU so the table is readable

  struct Rec { unsigned block, warp, lane, sm; };
  for (unsigned threads_per_block : {96u, 100u}) {
    sim::banner(threads_per_block == 96 ? "block = 96 threads (3 full warps)"
                                        : "block = 100 threads (4 warps, last one has 4 lanes)");
    const unsigned blocks = 6;
    std::vector<Rec> rec(blocks * threads_per_block);
    sim::launch(sim::dim3(blocks), sim::dim3(threads_per_block), [&](sim::ThreadCtx& c) {
      rec[c.block_linear * threads_per_block + c.tid] = {c.block_linear, c.warp, c.lane, c.sm};
    });

    std::printf("block -> SM (round-robin illustration; real hardware assigns dynamically as SMs free up)\n");
    for (unsigned b = 0; b < blocks; ++b) std::printf("  block %u -> SM %u\n", b, rec[b * threads_per_block].sm);

    unsigned warps = (threads_per_block + 31) / 32;
    std::printf("warps per block: %u\n", warps);
    for (unsigned w = 0; w < warps; ++w) {
      unsigned active = 0;
      for (unsigned t = 0; t < threads_per_block; ++t)
        if (rec[t].warp == w) ++active;
      std::printf("  warp %u: %2u/32 lanes active (%.0f%% utilisation)\n", w, active, 100.0 * active / 32);
    }
  }
}
