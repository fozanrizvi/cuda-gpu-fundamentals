# Thread hierarchy

**Concept.** A launch is a grid of blocks of threads (up to 3D each). A thread finds its work from `blockIdx`, `blockDim`, `threadIdx`; the grid is rounded up, so kernels need a bounds guard.

**Hardware mapping.** Software-only naming: the grid is the whole job, a block is the unit handed to one SM, a thread is one lane of a warp.

**What to observe.** Which block owns each pixel, and how many extra threads the ceil-division launches.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run thread_hierarchy`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
