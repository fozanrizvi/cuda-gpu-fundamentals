# Unified memory

**Concept.** `cudaMallocManaged` gives one pointer usable from CPU and GPU; pages migrate on demand.

**Hardware mapping.** Page faults over PCIe/NVLink trigger migrations; prefetching moves data ahead of use.

**What to observe.** Explicit copy moves everything; managed moves only touched pages. (The sim is an approximate cost model.)

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run unified_memory`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
