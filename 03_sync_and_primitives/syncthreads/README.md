# __syncthreads

**Concept.** A barrier across one block, guaranteeing earlier shared/global writes are visible to the whole block.

**Hardware mapping.** Warps of a block progress independently on the SM; the barrier makes them wait for each other. There is no barrier across blocks.

**What to observe.** The reversal is correct with the barrier and corrupt without it.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run syncthreads`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
