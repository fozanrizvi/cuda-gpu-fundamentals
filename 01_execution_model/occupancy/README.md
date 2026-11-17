# Occupancy

**Concept.** Resident warps per SM / maximum warps per SM. Limited by threads, blocks, registers or shared memory per SM, whichever runs out first.

**Hardware mapping.** The SM has fixed pools (thread slots, register file, shared memory). Each resident block carves out a piece.

**What to observe.** The limiting resource for each configuration. More registers or smem per block => fewer resident blocks.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run occupancy`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
