# Shared memory bank conflicts

**Concept.** Shared memory is 32 banks x 4 bytes. Lanes hitting different words in one bank are serialised.

**Hardware mapping.** Banks are parallel SRAM ports; conflicts reduce throughput by the conflict degree.

**What to observe.** Column access of a 32x32 tile is 32-way; padding to 33 makes it 1-way. Same-word reads broadcast.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run bank_conflicts`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
