#!/usr/bin/env bash
# Build (and optionally run) every CPU-simulator example. No CUDA or CMake required.
#   ./build_sim.sh          build all into ./build/
#   ./build_sim.sh run      build all and run each
#   ./build_sim.sh run occupancy   build+run only paths matching "occupancy"
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
mode="${1:-build}"
filter="${2:-}"
mkdir -p build
status=0
for src in [0-9][0-9]_*/*/sim.cpp; do
  [[ -n "$filter" && "$src" != *"$filter"* ]] && continue
  name="$(basename "$(dirname "$src")")"
  if ! "$CXX" -std=c++17 -O1 -Wall -Wextra -pthread -Icommon/sim "$src" -o "build/$name"; then
    echo "BUILD FAILED: $src"; status=1; continue
  fi
  if [[ "$mode" == "run" ]]; then
    echo; echo "################ $src"
    "build/$name" || { echo "RUN FAILED: $name"; status=1; }
  fi
done
exit $status
