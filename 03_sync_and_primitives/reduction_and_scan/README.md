# Reduction and prefix scan

**Concept.** Parallel patterns built from shared memory plus `__syncthreads` in log2(N) steps.

**Hardware mapping.** Each step halves (reduction) or doubles (scan) the active work per block; steps are separated by barriers.

**What to observe.** Both results match the sequential CPU answers.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run reduction_and_scan`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
