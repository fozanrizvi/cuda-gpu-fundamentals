# Shared memory (tiling)

**Concept.** A block-local scratchpad explicitly managed by the program. Tiled matmul loads each global element once per tile instead of once per use.

**Hardware mapping.** On-chip SRAM inside the SM, ~100x lower latency than DRAM; its size limits resident blocks.

**What to observe.** Global loads drop by a factor of TILE; the result matches the CPU reference.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run shared_memory`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
