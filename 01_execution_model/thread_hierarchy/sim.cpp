// Thread hierarchy: map a 2D grid of 2D blocks onto a W x H image. Shows which block owns
// each pixel and why we need a bounds check when the image is not a multiple of the block size.
#include <cstdio>
#include <vector>
#include "cuda_sim.hpp"

int main() {
  const int W = 6, H = 5;
  sim::dim3 block(4, 2);                                     // 8 threads per block
  sim::dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);  // ceil-div
  std::printf("image %dx%d, block (%u,%u), grid (%u,%u) => %u blocks, %u threads launched for %d pixels\n",
              W, H, block.x, block.y, grid.x, grid.y, grid.count(), grid.count() * block.count(), W * H);

  std::vector<int> owner(W * H, -1);  // which linear block id wrote each pixel
  int idle = 0;
  std::mutex m;
  sim::launch(grid, block, [&](sim::ThreadCtx& c) {
    int x = c.blockIdx.x * c.blockDim.x + c.threadIdx.x;
    int y = c.blockIdx.y * c.blockDim.y + c.threadIdx.y;
    if (x >= W || y >= H) {  // the guard: these threads exist but have no work
      std::lock_guard<std::mutex> g(m);
      ++idle;
      return;
    }
    owner[y * W + x] = c.block_linear;
  });

  std::printf("\nblock id that processed each pixel:\n");
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) std::printf(" %d", owner[y * W + x]);
    std::printf("\n");
  }
  std::printf("\n%d threads were launched but fell outside the image (masked by the bounds check)\n", idle);
}
