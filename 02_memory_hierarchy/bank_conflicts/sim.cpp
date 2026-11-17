// Shared-memory bank conflicts: shared memory has 32 banks, each 4 bytes wide; word w lives in bank w % 32.
// If several lanes of a warp touch *different* words in the same bank, the access is serialised.
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

static int degree_for(unsigned (*index)(unsigned lane)) {
  std::vector<unsigned> idx(32);
  sim::launch(sim::dim3(1), sim::dim3(32), [&](sim::ThreadCtx& c) { idx[c.tid] = index(c.lane); });
  return sim::bank_conflict_degree(idx);
}

int main() {
  std::printf("%-46s %s\n", "access (one warp, float tile in shared mem)", "conflict degree (passes)");
  std::printf("%-46s %d\n", "tile[0][lane]        row read, stride 1", degree_for([](unsigned l) { return l; }));
  std::printf("%-46s %d\n", "tile[0][2*lane]      stride 2", degree_for([](unsigned l) { return 2 * l; }));
  std::printf("%-46s %d\n", "tile[0][4*lane]      stride 4", degree_for([](unsigned l) { return 4 * l; }));
  std::printf("%-46s %d\n", "tile[lane][0]  tile[32][32]  column read", degree_for([](unsigned l) { return l * 32; }));
  std::printf("%-46s %d\n", "tile[lane][0]  tile[32][33]  padded column", degree_for([](unsigned l) { return l * 33; }));
  std::printf("%-46s %d\n", "tile[0][5]           all lanes same word", degree_for([](unsigned) { return 5u; }));
  std::printf("\nColumn access of a 32-wide tile hits bank 0 on every lane (32-way conflict). Padding each row by one\n"
              "word shifts successive rows by one bank, making the column access conflict-free.\n"
              "Same-word reads are a broadcast, not a conflict.\n");
}
