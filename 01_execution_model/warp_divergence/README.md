# Warp divergence

**Concept.** All 32 lanes of a warp share one program counter. If a branch splits the lanes, each path runs in turn with the others masked off.

**Hardware mapping.** SIMT execution on the SM: cost = sum of the paths taken in a warp. Branches that are uniform per warp are free.

**What to observe.** Passes per warp for several branch conditions and the active masks.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run warp_divergence`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
