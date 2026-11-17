# Global memory coalescing

**Concept.** A warp's 32 loads are merged into 32-byte sectors / 128-byte lines. Adjacent lanes touching adjacent addresses need the fewest.

**Hardware mapping.** DRAM and L2 serve whole sectors; uncoalesced access wastes bandwidth, the usual bottleneck.

**What to observe.** Sectors per warp rises from 4 to 32 as stride grows; efficiency falls from 100% to 12%.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run global_coalescing`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
