// cuda_sim.hpp - a tiny CPU simulator of the CUDA execution model.
//
// Purpose: let you *observe* how CUDA concepts map onto GPU hardware without a GPU.
//   - a kernel is a callable taking a ThreadCtx&  (the "thread" sees threadIdx, blockIdx, ...)
//   - every CUDA thread of a block is a real std::thread, so __syncthreads() (ctx.sync()) is a real barrier
//   - blocks run one after another (a real GPU runs many at once, on different SMs)
//   - each thread knows its warp (32 consecutive threads), lane, and the SM its block was assigned to
//
// What it is NOT: a performance model. Timings are meaningless; counts (transactions,
// bank conflicts, passes, occupancy) are the interesting output.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace sim {

constexpr unsigned kWarpSize = 32;

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
  unsigned count() const { return x * y * z; }
};

// ---- Hardware description -------------------------------------------------------------
struct DeviceSpec {
  std::string name;
  int sm_count;               // streaming multiprocessors
  int max_threads_per_sm;     // resident threads per SM
  int max_blocks_per_sm;      // resident blocks per SM
  int max_threads_per_block;
  int regs_per_sm;            // 32-bit registers in the SM's register file
  int smem_per_sm;            // bytes of shared memory per SM
  int cc_major, cc_minor;     // compute capability
};

inline DeviceSpec a100_like() {
  return {"Simulated A100-like (sm_80)", 108, 2048, 32, 1024, 65536, 164 * 1024, 8, 0};
}

// Global "current device" so examples can tweak it (e.g. shrink to 4 SMs for readable output).
inline DeviceSpec& device() {
  static DeviceSpec d = a100_like();
  return d;
}

// ---- Occupancy calculator (ignores allocation granularity, so numbers are slightly optimistic)
struct Occupancy {
  int blocks_per_sm;
  int warps_per_sm;
  double fraction;      // resident warps / max warps
  const char* limiter;  // which resource ran out first
};

inline Occupancy occupancy(int threads_per_block, int regs_per_thread, int smem_per_block) {
  const DeviceSpec& d = device();
  const int kInf = 1 << 30;
  int by_threads = d.max_threads_per_sm / threads_per_block;
  int by_blocks = d.max_blocks_per_sm;
  int by_regs = regs_per_thread > 0 ? d.regs_per_sm / (regs_per_thread * threads_per_block) : kInf;
  int by_smem = smem_per_block > 0 ? d.smem_per_sm / smem_per_block : kInf;
  int best = std::min(std::min(by_threads, by_blocks), std::min(by_regs, by_smem));
  const char* why = best == by_threads ? "threads/SM" : best == by_blocks ? "blocks/SM"
                  : best == by_regs ? "registers" : "shared memory";
  int warps_per_block = (threads_per_block + kWarpSize - 1) / kWarpSize;
  int warps = best * warps_per_block;
  int max_warps = d.max_threads_per_sm / kWarpSize;
  return {best, warps, double(warps) / max_warps, why};
}

// ---- Block / thread execution ----------------------------------------------------------
class Barrier {
 public:
  explicit Barrier(unsigned n) : count_(n) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    unsigned g = gen_;
    if (++waiting_ == count_) {
      waiting_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return g != gen_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  unsigned count_, waiting_ = 0, gen_ = 0;
};

struct BlockState {
  BlockState(unsigned nthreads, size_t smem_bytes)
      : barrier(nthreads), smem(smem_bytes + 8, 0), xchg(nthreads, 0) {}
  Barrier barrier;
  std::vector<unsigned char> smem;   // the block's shared memory (one per block, as on hardware)
  std::vector<uint64_t> xchg;        // scratch used to emulate warp shuffles
};

struct ThreadCtx {
  dim3 threadIdx, blockIdx, blockDim, gridDim;
  unsigned tid = 0;           // linear thread index inside the block
  unsigned block_linear = 0;  // linear block index inside the grid
  unsigned warp = 0;          // warp index inside the block (tid / 32)
  unsigned lane = 0;          // lane inside the warp (tid % 32)
  unsigned sm = 0;            // SM this block was assigned to (round-robin illustration)
  BlockState* blk = nullptr;

  // __syncthreads()
  void sync() { blk->barrier.wait(); }

  // extern __shared__ T smem[]
  template <class T>
  T* shared() { return reinterpret_cast<T*>(blk->smem.data()); }

  // __shfl_down_sync(full_mask, v, delta). Emulated with a block barrier, so *every* thread of the
  // block must call it (on hardware only the warp must, because a warp runs in lockstep).
  template <class T>
  T shfl_down(T v, unsigned delta) {
    static_assert(sizeof(T) <= 8, "shuffle value too large");
    std::memcpy(&blk->xchg[tid], &v, sizeof(T));
    sync();
    unsigned src = (lane + delta < kWarpSize && tid + delta < blockDim.x * blockDim.y * blockDim.z) ? tid + delta : tid;
    T r;
    std::memcpy(&r, &blk->xchg[src], sizeof(T));
    sync();
    return r;
  }

  unsigned global_x() const { return blockIdx.x * blockDim.x + threadIdx.x; }
};

// <<<grid, block, smem_bytes>>> kernel(...)
template <class Kernel>
void launch(dim3 grid, dim3 block, Kernel kernel, size_t smem_bytes = 0) {
  const unsigned nb = grid.count(), nt = block.count();
  for (unsigned b = 0; b < nb; ++b) {
    BlockState st(nt, smem_bytes);
    std::vector<std::thread> threads;
    threads.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
      threads.emplace_back([&, t, b] {
        ThreadCtx c;
        c.gridDim = grid;
        c.blockDim = block;
        c.blockIdx = dim3(b % grid.x, (b / grid.x) % grid.y, b / (grid.x * grid.y));
        c.threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
        c.tid = t;
        c.block_linear = b;
        c.warp = t / kWarpSize;
        c.lane = t % kWarpSize;
        c.sm = b % device().sm_count;
        c.blk = &st;
        kernel(c);
      });
    }
    for (auto& th : threads) th.join();
  }
}

// ---- Atomics (global lock; fine for a teaching simulator) -----------------------------
inline std::mutex& atomic_mutex() {
  static std::mutex m;
  return m;
}
template <class T>
T atomicAdd(T* addr, T v) {
  std::lock_guard<std::mutex> g(atomic_mutex());
  T old = *addr;
  *addr = old + v;
  return old;
}

// ---- Memory-system analyzers (operate on the addresses one warp touches) ---------------
// Global memory: a warp's 32 requests are served in 32-byte sectors (grouped in 128-byte lines).
inline int sectors_touched(const std::vector<uint64_t>& byte_addrs) {
  std::set<uint64_t> s;
  for (auto a : byte_addrs) s.insert(a / 32);
  return int(s.size());
}
inline int lines_touched(const std::vector<uint64_t>& byte_addrs) {
  std::set<uint64_t> s;
  for (auto a : byte_addrs) s.insert(a / 128);
  return int(s.size());
}
// Shared memory: 32 banks, 4 bytes wide. Degree N = the access is serviced in N passes.
// Several lanes reading the *same* word is a broadcast and costs nothing extra.
inline int bank_conflict_degree(const std::vector<unsigned>& word_idx) {
  std::map<unsigned, std::set<unsigned>> per_bank;
  for (auto w : word_idx) per_bank[w % 32].insert(w);
  size_t deg = 1;
  for (auto& kv : per_bank) deg = std::max(deg, kv.second.size());
  return int(deg);
}

// Split per-thread records into warps and apply `f(warp_index, slice_of_32)`.
template <class T, class F>
void for_each_warp(const std::vector<T>& per_thread, F f) {
  for (size_t w = 0; w * kWarpSize < per_thread.size(); ++w) {
    size_t lo = w * kWarpSize, hi = std::min(per_thread.size(), lo + kWarpSize);
    f(w, std::vector<T>(per_thread.begin() + lo, per_thread.begin() + hi));
  }
}

inline void banner(const char* title) { std::printf("\n=== %s ===\n", title); }

}  // namespace sim
