// Occupancy: how many warps can be resident on one SM, given the block's resource demands.
// Each SM has fixed pools: thread slots, block slots, registers, shared memory. The scarcest one wins.
#include <cstdio>
#include "cuda_sim.hpp"

int main() {
  const auto& d = sim::device();
  std::printf("Device: %s\n  per SM: %d threads (%d warps), %d blocks, %d regs, %d KB smem\n", d.name.c_str(),
              d.max_threads_per_sm, d.max_threads_per_sm / 32, d.max_blocks_per_sm, d.regs_per_sm, d.smem_per_sm / 1024);

  struct Cfg { int tpb, regs, smem; };
  Cfg cfgs[] = {
      {256, 32, 0},     {256, 64, 0},      {256, 128, 0},   {256, 255, 0},
      {64, 32, 0},      {32, 32, 0},       {1024, 32, 0},   {256, 32, 48 * 1024},
      {256, 32, 100 * 1024}, {128, 40, 16 * 1024},
  };
  std::printf("\n%8s %8s %10s | %10s %10s %10s  limited by\n", "thr/blk", "regs/thr", "smem/blk", "blocks/SM", "warps/SM", "occupancy");
  for (auto c : cfgs) {
    auto o = sim::occupancy(c.tpb, c.regs, c.smem);
    std::printf("%8d %8d %9dB | %10d %10d %9.0f%%  %s\n", c.tpb, c.regs, c.smem, o.blocks_per_sm, o.warps_per_sm,
                o.fraction * 100, o.limiter);
  }
  std::printf("\nHigher occupancy helps hide memory latency (the SM switches to another warp for free),\n"
              "but it is not the goal itself: kernels with lots of ILP can run fast at low occupancy.\n");
}
