// Streams (APPROXIMATE timeline model, not real timing).
// A GPU has independent engines: host->device copy, compute, device->host copy. Work in ONE stream is
// ordered; work in DIFFERENT streams may overlap if it uses different engines. We split a job into chunks
// and schedule them with a simple "earliest start" rule to show the pipeline.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

struct Op { int chunk, stream; const char* engine; double start, end; };

static double simulate(int chunks, int streams, double th, double tk, double td, bool print) {
  double free_engine[3] = {0, 0, 0};            // H2D, compute, D2H
  std::vector<double> stream_ready(streams, 0); // in-stream ordering
  std::vector<Op> ops;
  const char* names[3] = {"H2D ", "KERN", "D2H "};
  double dur[3] = {th, tk, td};
  double finish = 0;
  for (int c = 0; c < chunks; ++c) {
    int s = c % streams;
    double t = stream_ready[s];
    for (int e = 0; e < 3; ++e) {
      double start = std::max(t, free_engine[e]);
      double end = start + dur[e];
      free_engine[e] = end;
      t = end;
      ops.push_back({c, s, names[e], start, end});
    }
    stream_ready[s] = t;
    finish = std::max(finish, t);
  }
  if (print) {
    for (auto& o : ops)
      std::printf("  stream %d chunk %d %s  %5.1f -> %5.1f\n", o.stream, o.chunk, o.engine, o.start, o.end);
  }
  return finish;
}

int main() {
  const int chunks = 4;
  const double th = 2, tk = 3, td = 2;  // ms per chunk on each engine
  std::printf("4 chunks, each: H2D %.0f ms, kernel %.0f ms, D2H %.0f ms\n\n", th, tk, td);
  std::printf("1 stream (everything serialised):\n");
  double t1 = simulate(chunks, 1, th, tk, td, true);
  std::printf("  total = %.1f ms\n\n4 streams (copies of one chunk overlap compute of another):\n", t1);
  double t4 = simulate(chunks, 4, th, tk, td, true);
  std::printf("  total = %.1f ms  (%.2fx faster)\n", t4, t1 / t4);
  std::printf("\nCUDA events (cudaEventRecord) are timestamps placed in a stream; cudaStreamWaitEvent lets one\n"
              "stream wait for another. Overlap needs pinned host memory (cudaMallocHost) + cudaMemcpyAsync.\n");
}
