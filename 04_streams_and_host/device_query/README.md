# Device query

**Concept.** `cudaGetDeviceProperties` exposes the limits that drive every launch decision.

**Hardware mapping.** Shows the GPU > SM > warp scheduler hierarchy with sizes of each resource.

**What to observe.** The concept-to-hardware table; the simulator prints an A100-like spec.

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run device_query`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
