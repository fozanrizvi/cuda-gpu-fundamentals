# Streams and events

**Concept.** A stream is an ordered queue of GPU work. Different streams can overlap; events timestamp or order them.

**Hardware mapping.** Copy engines (H2D, D2H) and the compute engine run concurrently if work is in different streams and host memory is pinned.

**What to observe.** The pipelined timeline is faster than the serial one. (The sim is an approximate scheduling model, not measured time.)

**Run.** `sim.cpp` runs anywhere: `../../build_sim.sh run streams_events`. `kernel.cu` needs an NVIDIA GPU: `nvcc -O2 kernel.cu -o demo && ./demo`.
