# Constant memory and registers

**Concept.** Constant memory (64 KB) is cached and broadcasts one address to a warp. Registers are private per thread and come from the SM register file.

**Hardware mapping.** Many registers/thread => fewer resident warps. Overflow spills to "local" memory, which is actually DRAM.

**What to observe.** Passes grow with distinct constant addresses per warp; occupancy falls as regs/thread rise.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run constant_and_registers`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
