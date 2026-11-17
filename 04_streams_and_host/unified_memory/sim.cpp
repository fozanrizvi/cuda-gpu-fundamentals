// Unified (managed) memory (APPROXIMATE cost model, not real timing).
// cudaMallocManaged returns one pointer valid on CPU and GPU. Pages migrate on demand: a GPU touching a
// page that lives on the host triggers a page fault and a migration. Explicit cudaMemcpy moves everything.
#include <cstdio>
#include <initializer_list>

int main() {
  const double MB = 1 << 20;
  const double bytes = 1024 * MB;      // 1 GB buffer
  const double page = 2 * MB;          // migration granularity (real: 4 KB - 2 MB depending on access pattern)
  const double pcie_gbps = 25;         // PCIe Gen4 x16 effective
  const double fault_us = 20;          // per-fault handling overhead
  auto xfer_ms = [&](double b) { return b / (pcie_gbps * 1e9) * 1e3; };

  std::printf("1 GB buffer; kernel touches a fraction of it. Model: PCIe %.0f GB/s, %.0f us/fault, %.0f MB pages\n\n",
              pcie_gbps, fault_us, page / MB);
  std::printf("%-12s %-18s %-22s %-22s\n", "touched", "explicit cudaMemcpy", "managed (fault-driven)", "managed + prefetch");
  for (double frac : {1.0, 0.5, 0.25, 0.05}) {
    double touched = bytes * frac;
    double pages = touched / page;
    double t_explicit = xfer_ms(bytes);                              // programmer copies the whole thing
    double t_managed = xfer_ms(touched) + pages * fault_us / 1e3;    // only touched pages move, each pays a fault
    double t_prefetch = xfer_ms(touched);                            // cudaMemPrefetchAsync: bulk move, no faults
    std::printf("%3.0f%%         %-12.1f ms    %-14.1f ms    %-14.1f ms\n", frac * 100, t_explicit, t_managed, t_prefetch);
  }
  std::printf("\nTakeaways: managed memory simplifies code and moves only what is used, but fault handling costs\n"
              "time when most of the buffer is touched. cudaMemPrefetchAsync / cudaMemAdvise recover the speed.\n");
}
