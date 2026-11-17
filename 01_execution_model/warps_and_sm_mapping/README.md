# Warps and SM mapping

**Concept.** Blocks are distributed to SMs; inside a block, consecutive 32 threads form a warp (`tid / 32`, lane = `tid % 32`).

**Hardware mapping.** A block lives on one SM for its whole life. The SM schedules warps, not threads, so a 100-thread block still occupies 4 warps.

**What to observe.** Block-to-SM table and the 12% utilisation of the last warp. On a real GPU, `kernel.cu` reads `%smid` to show the true (non-round-robin) placement.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run warps_and_sm_mapping`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
