# CUDA → GPU Fundamentals

A hands-on C++ project for understanding how CUDA's programming model maps onto GPU hardware. Each concept lives in its own folder with three files:

| file | purpose |
|---|---|
| `README.md` | the concept, its hardware mapping, what to look for |
| `sim.cpp` | **runs on any machine** (including a Mac): a CPU simulator of the CUDA execution model that prints the observable effect (warp masks, sectors, bank conflicts, occupancy, ...) |
| `kernel.cu` | the same idea as real CUDA, for an NVIDIA GPU (Colab, cloud, local) |

## Quick start (no GPU needed)
```bash
./build_sim.sh run               # build and run every simulator
./build_sim.sh run coalescing    # just one concept (substring match)
# or with CMake: cmake -S . -B build && cmake --build build
```
On an NVIDIA machine, CMake also builds each `kernel.cu` as `cuda_<name>`; or compile one by hand: `nvcc -O2 kernel.cu -o demo`.

## Reading order
1. **01_execution_model** – `thread_hierarchy` → `warps_and_sm_mapping` → `warp_divergence` → `occupancy`
2. **02_memory_hierarchy** – `global_coalescing` → `shared_memory` → `bank_conflicts` → `constant_and_registers`
3. **03_sync_and_primitives** – `syncthreads` → `atomics` → `warp_shuffle` → `reduction_and_scan`
4. **04_streams_and_host** – `device_query` → `streams_events` → `unified_memory`

## Software ↔ hardware cheat sheet
| CUDA | Hardware |
|---|---|
| thread | one lane of a warp, runs on an ALU/CUDA core |
| warp (32 threads) | unit issued by a warp scheduler; one program counter (SIMT) |
| block | scheduled onto exactly one SM and stays there |
| grid | all blocks, distributed over SMs by the hardware block scheduler |
| registers | SM register file, private per thread |
| shared memory | on-chip SRAM in the SM, per block, 32 banks |
| global / local memory | off-chip DRAM via L2 (and L1) |
| constant memory | cached read-only region, broadcast to a warp |
| stream | ordered work queue; copy and compute engines overlap across streams |

## About the simulator (`common/sim/cuda_sim.hpp`)
- Each CUDA thread is a real `std::thread`; blocks run one after another; `ctx.sync()` is a real barrier.
- Warp = 32 consecutive thread ids; block→SM is round-robin (real hardware assigns dynamically).
- It reports **counts** (passes, sectors, conflicts, occupancy), never performance. Warp shuffles are emulated with a block barrier (all threads of the block must call them). Keep block sizes small (a thread per CUDA thread).
- `streams_events` and `unified_memory` are explicitly approximate models; use their `kernel.cu` for real behavior.
- The `.cu` files were written for CUDA 12 but have not been compiled in this repo's dev environment (no NVIDIA toolchain on macOS); expect to fix small issues on first build.
