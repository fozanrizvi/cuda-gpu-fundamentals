// Device query: print the hardware hierarchy a CUDA program is mapped onto (simulated A100-like spec).
// On a real GPU, kernel.cu prints the same fields from cudaGetDeviceProperties.
#include <cstdio>
#include "cuda_sim.hpp"

int main() {
  const auto& d = sim::device();
  std::printf("Device: %s (compute capability %d.%d)\n\n", d.name.c_str(), d.cc_major, d.cc_minor);
  std::printf("GPU\n"
              " +-- %d SMs (streaming multiprocessors)\n"
              "      +-- up to %d resident threads = %d warps (warp = 32 threads, runs in lockstep)\n"
              "      +-- up to %d resident blocks\n"
              "      +-- register file: %d x 32-bit regs (%d KB) shared by all resident threads\n"
              "      +-- shared memory / L1: %d KB (carved up among resident blocks)\n"
              "      +-- 4 warp schedulers: each cycle each issues one instruction from a ready warp\n\n",
              d.sm_count, d.max_threads_per_sm, d.max_threads_per_sm / 32, d.max_blocks_per_sm, d.regs_per_sm,
              d.regs_per_sm * 4 / 1024, d.smem_per_sm / 1024);
  std::printf("Per block limit: %d threads\n", d.max_threads_per_block);
  std::printf("Threads in flight on the whole GPU: %d x %d = %d\n", d.sm_count, d.max_threads_per_sm,
              d.sm_count * d.max_threads_per_sm);
  std::printf("\nCUDA concept        ->  hardware\n"
              "  thread            ->  one lane of a warp (executes on a CUDA core/ALU)\n"
              "  warp (32 threads) ->  the unit the scheduler issues instructions for\n"
              "  block             ->  resident on exactly ONE SM until it finishes\n"
              "  grid              ->  all blocks; spread across SMs by the hardware block scheduler\n"
              "  shared memory     ->  on-chip SRAM inside the SM, one slice per resident block\n"
              "  registers         ->  SM register file, private per thread\n"
              "  global memory     ->  off-chip DRAM (HBM/GDDR), cached in L2 (and L1)\n");
}
