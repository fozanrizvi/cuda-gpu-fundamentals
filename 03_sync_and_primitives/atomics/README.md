# Atomics

**Concept.** Read-modify-write operations that cannot be interleaved by other threads. Needed when many threads update one location.

**Hardware mapping.** Executed in L2 (global) or the SM (shared). Same-address atomics serialise.

**What to observe.** Privatising the histogram in shared memory cuts global atomics 32x here.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run atomics`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
