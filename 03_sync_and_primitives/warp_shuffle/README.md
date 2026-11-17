# Warp shuffle

**Concept.** `__shfl_down_sync` reads a register of another lane in the same warp.

**Hardware mapping.** Lanes exchange data through the register crossbar: no shared memory, no block barrier.

**What to observe.** Five steps (16, 8, 4, 2, 1) reduce 32 values; lane 0 holds the sum.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run warp_shuffle`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
